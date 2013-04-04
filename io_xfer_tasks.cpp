#include "sortio.h"

// --------------------------------------------------------------------
// Transfer_Tasks_Work(): work manager for data transfer tasks
// 
// This method configured to run on IO_COMM
// --------------------------------------------------------------------

void sortio_Class::Transfer_Tasks_Work()
{
  assert(initialized);

  const int USLEEP_INTERVAL = 10000;
  //const int USLEEP_INTERVAL = 1000;
  //const int USLEEP_INTERVAL = 100;
  int numTransferedFiles = 0;

  std::vector<int> fullQueueCounts(nio_tasks,0);

  // Start main data transfer loop; we continue until all files have
  // been read from IO_COMM and are transferred to SORT_COMM

  int count = 0;
  int *buf_nums;
  MPI_Request requestHandle;
  std::vector<int>::iterator itMax;
  int procMax;
  int maxCount;
  unsigned char *bufferRecv;
  int destRank = nio_tasks;
  bool waitFlag;

  buf_nums     = new int[MAX_READ_BUFFERS];

  bufferRecv = (unsigned char*) calloc(2*MAX_FILE_SIZE_IN_MBS*1024*1024,sizeof(unsigned char));

  nextDestRank_ = nio_tasks;

  while (numTransferedFiles < num_files_total)
    {
      // Check which processor has the most data available

      int localCount = fullQueue_.size();

      assert (MPI_Gather(&localCount,1,MPI_INTEGER,fullQueueCounts.data(),1,MPI_INTEGER,0,IO_COMM) == MPI_SUCCESS);

      if(master_io)
	{
	  itMax   = std::max_element(fullQueueCounts.begin(), fullQueueCounts.end());;
	  procMax = std::distance   (fullQueueCounts.begin(), itMax);

	  maxCount = *itMax;

	  grvy_printf(INFO,"[sortio][IO/XFER] Max full queue elements = %2i (at proc %.4i) -> iter = %i\n",
		      maxCount,procMax,count);
	}

      assert( MPI_Bcast(&maxCount,1,MPI_INTEGER,0,IO_COMM) == MPI_SUCCESS );
      assert( MPI_Bcast(&procMax, 1,MPI_INTEGER,0,IO_COMM) == MPI_SUCCESS );

      count++;

      // Distribute data from the processor with the most data on hand;
      // if no processors are ready yet, iterate and check again

      if(maxCount > 0)
	{
	  // step 1: let all the xfer tasks know who will be scattering data
#if 1
	  assert( MPI_Bcast(&procMax,1,MPI_INTEGER,0,IO_COMM) == MPI_SUCCESS );
#else
	  assert( MPI_Bcast(&procMax,1,MPI_INTEGER,0,XFER_COMM) == MPI_SUCCESS );
#endif

	  if(io_rank == procMax)
	    {
	      // step 1: verify that we have no remaining outstanding messages
	      // on this processor (wait if we do)

	      checkForSendCompletion(waitFlag=true);

	      grvy_printf(INFO,"[sortio][IO/XFER][%.4i] outstanding sends = %i\n",io_rank,messageQueue_.size());
	      assert(messageQueue_.size() == 0);

	      // step 2: lock the data transfer buffers on this processor

	      //int buf_num;

              #pragma omp critical (IO_XFER_UPDATES_lock) // Thread-safety: all queue updates are locked
	      {
		for(int i=0;i<maxCount;i++)
		  {
		    buf_nums[i] = fullQueue_.front();
		    fullQueue_.pop_front();
		    grvy_printf(INFO,"[sortio][IO/XFER][%.4i] removed %i buff from fullQueue\n",io_rank,buf_nums[i]);
		  }
	      }

	      grvy_printf(INFO,"[sortio][IO/XFER][%.4i] Sending %i buffers...\n",io_rank,maxCount);

	      assert(buffers[buf_nums[0]] != NULL);
	      assert(buf_nums[0] < MAX_READ_BUFFERS);

	      // step3: send buffers to XFER ranks asynchronously

	      const int tagXFER = 1000;

	      for(int i=0;i<maxCount;i++)
		{
		  grvy_printf(INFO,"[sortio][IO/XFER][%.4i] issuing iSend to rank %i\n",io_rank,nextDestRank_);
		  MPI_Isend(&buffers[buf_nums[i]],100,MPI_UNSIGNED_CHAR,CycleDestRank(),
			    tagXFER,XFER_COMM,&requestHandle);
		  
		  // marsk these messages as being in flight

		  MsgRecord message(buf_nums[i],requestHandle);
		  messageQueue_.push_back(message);
		}
	    } 
	  else
	    {
	      for(int i=0;i<maxCount;i++)
		CycleDestRank();
	    }

	  // All IO ranks keep track of total number of files transferred

	  numTransferedFiles += maxCount;
	  destRank           += maxCount;

	  // Check for any completed messages prior to next iteration

	  checkForSendCompletion(waitFlag=false);

	}
      else
	usleep(USLEEP_INTERVAL);

    } //  end xfer of all files

  // Send notice to xfer tasks to let them know we are all done

  int signalCompletion = -1;

  assert( MPI_Bcast(&signalCompletion,1,MPI_INTEGER,0,XFER_COMM) == MPI_SUCCESS );

  grvy_printf(INFO,"[sortio][IO/XFER][%.4i]: data XFER completed\n",io_rank);

  delete [] buf_nums;

  return;
}


// --------------------------------------------------------------------
// SendDataToXFERTasks(): asynchronously distribute buffer(s) to 
// receiving tasks in XFER_COMM
// --------------------------------------------------------------------

void sortio_Class::SendDataToXFERTasks(int numBuffers,int destination)
{

  // To begin, we impose a single src->destination xfer rule
  // (ie. pause if a previous set of sends from this src to this
  // destination have not completed.

  int src = io_rank;


}

// Check on messages in flight and free up data transfer buffers for
// any which have completed.

void sortio_Class::checkForSendCompletion(bool waitFlag)
{

  // a NOOP if we have no outstanding messages

  if(messageQueue_.size() == 0)
    return;

  std::list<MsgRecord>::iterator it = messageQueue_.begin();

  while(it != messageQueue_.end() )
    {
      int messageCompleted, bufNum;
      MPI_Status status;
      MPI_Request handle;

      bufNum = it->getBufNum();
      handle = it->getHandle();

      // check if iSend has completed

      MPI_Test(&handle,&messageCompleted,&status);

      if(messageCompleted)
	{
	  grvy_printf(INFO,"[sortio][IO/XFER][%.4i] message from buf %i complete \n",io_rank,bufNum);

	  it = messageQueue_.erase(it++);
	  
	  // re-enable this buffer for eligibility
	  
          #pragma omp critical (IO_XFER_UPDATES_lock) // Thread-safety: all queue updates are locked
	  {
	    emptyQueue_.push_back(bufNum);
	    grvy_printf(INFO,"[sortio][IO/XFER][%.4i] added %i buff back to emptyQueue\n",io_rank,bufNum);
	  }
	}
      else if(waitFlag)
	{
	  grvy_printf(INFO,"[sortio][IO/XFER][%.4i] Stalling for previously unfinished iSend (buf=%i)\n",
		      io_rank,bufNum);
	  MPI_Wait(&handle,&status);

	  it = messageQueue_.erase(it++);

          #pragma omp critical (IO_XFER_UPDATES_lock) // Thread-safety: all queue updates are locked
	  {
	    emptyQueue_.push_back(bufNum);
	    grvy_printf(INFO,"[sortio][IO/XFER][%.4i] added %i buff back to emptyQueue\n",io_rank,bufNum);
	  }
	}
      else
	++it;	// <-- message still active
    }

  return;
}

int sortio_Class::CycleDestRank()
{
  int startingDestRank = nextDestRank_;

  nextDestRank_++;

  if(nextDestRank_ >= nxfer_tasks)
    nextDestRank_ = nio_tasks;

  return(startingDestRank);
}
