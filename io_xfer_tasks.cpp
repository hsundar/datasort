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
  MPI_Request *sendRequests;
  std::vector<int>::iterator itMax;
  int procMax;
  int maxCount;
  unsigned char *bufferRecv;
  int numActiveSends = 0;
  int destRank = nio_tasks;
  //  std::vector<int> bufNumsPrev(MAX_READ_BUFFERS);
  //std::vector<int> numActiveSends(nio_tasks,0);

  buf_nums     = new int[MAX_READ_BUFFERS];
  sendRequests = new MPI_Request[MAX_READ_BUFFERS];

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

	      // verify no outstanding messages in flight and release
	      // xfer buffers from last send

	      int flag;
	      MPI_Status status;

	      grvy_printf(INFO,"[sortio][IO/XFER][%.4i] num active sends = %i\n",io_rank,numActiveSends);

	      const int numSendsPostedPreviously = numActiveSends;

	      for(int i=0;i<numSendsPostedPreviously;i++)
		{
		  grvy_printf(INFO,"[sortio][IO/XFER][%.4i] i = %i (max = %i)\n",io_rank,i,numActiveSends);

		  MPI_Test(&sendRequests[i],&flag,&status);
		  if(!flag)
		    {
		      grvy_printf(INFO,"[sortio][IO/XFER][%.4i] Stalling for previously unfinished iSend\n",io_rank);
		      MPI_Wait(&sendRequests[i],&status);
		    }
		  numActiveSends--;
		  grvy_printf(INFO,"[sortio][IO/XFER][%.4i] --> decremented to %i (max = %i)\n",
			      io_rank,numActiveSends);
		}

	      grvy_printf(INFO,"[sortio][IO/XFER][%.4i] outstanding sends = %i\n",io_rank,numActiveSends);
	      assert(numActiveSends == 0);

	      // step 2: distribute data from procMax using special
	      // scatter group with procmax as the leader

	      int buf_num;

              #pragma omp critical (IO_XFER_UPDATES_lock) // Thread-safety: all queue updates are locked
	      {
		// release previous buffers which have now had their
		// send completed

		for(int i=0;i<numSendsPostedPreviously)
		  fullQueue_.push(buf_nums[i]);

		// identify next set of data buffers to send.

		for(int i=0;i<maxCount;i++)
		  {
		    buf_nums[i] = fullQueue_.front();
		    fullQueue_.pop();
		    grvy_printf(INFO,"[sortio][IO/XFER][%.4i] removed %i buff from fullQueue\n",io_rank,buf_nums[i]);
		  }
	      }

	      grvy_printf(INFO,"[sortio][IO/XFER][%.4i] Sending %i buffers...\n",io_rank,maxCount);

	      //assert(procMax < Scatter_COMMS.size());
	      //MPI_Comm commScatter = Scatter_COMMS[procMax];

	      //const int numData = (1000000 / nxfer_tasks);
	      //const int numData = 10000;
	      //const int numData = (1000000 / nscatter_tasks);
	      //const int numData = 18250;
	      //const int numData = 100000;

	      assert(buffers[buf_nums[0]] != NULL);
	      assert(buf_nums[0] < MAX_READ_BUFFERS);
	      //assert(numData < MAX_FILE_SIZE_IN_MBS*1024*1024);
	      //	      assert(numData*nscatter_tasks <= 100*100*100);

#if 1
	      const int tagXFER = 1000;

	      for(int i=0;i<maxCount;i++)
		{
		  grvy_printf(INFO,"[sortio][IO/XFER][%.4i] issuing iSend to rank %i\n",io_rank,nextDestRank_);
		  MPI_Isend(&buffers[buf_nums[0]],100,MPI_UNSIGNED_CHAR,CycleDestRank(),
			    tagXFER,XFER_COMM,&sendRequests[i]);
		  numActiveSends++;
		}

	      //SendDataToXFERTasks(maxCount,procMax);
#else

	      MPI_Scatter(&buffers[buf_nums[0]],numData,MPI_UNSIGNED_CHAR,
			  bufferRecv,numData,MPI_UNSIGNED_CHAR,0,commScatter);

	      printf("[sortio][IO/XFER][%.4i] Just scattered %i (MB) of data\n",io_rank,
		     numData*nscatter_tasks/(1000*1000));
#endif

	      // fixme todo: need to cache buf_nums on a per rank
	      // basis; only release the buff once isend is complete

	      // step 3: flag this buffer as being eligible for read task to use again

              #pragma omp critical (IO_XFER_UPDATES_lock) // Thread-safety: all queue updates are locked
	      {
		for(int i=0;i<maxCount;i++)
		  {
		    emptyQueue_.push(buf_nums[i]);
		    grvy_printf(INFO,"[sortio][IO/XFER][%.4i] added %i buff back to emptyQueue\n",io_rank,buf_nums[i]);
		  }
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

	  // cyclic distribution to children in XFER_COMM
	  //	  if(destRank >= nxfer_tasks)
	  //	    destRank = nio_tasks;
	}
      else
	usleep(USLEEP_INTERVAL);

    } //  end xfer of all files

  // Send notice to xfer tasks to let them know we are all done

  int signalCompletion = -1;

  assert( MPI_Bcast(&signalCompletion,1,MPI_INTEGER,0,XFER_COMM) == MPI_SUCCESS );

#pragma omp critical (io_region_update)
  {
    // update region_flag here
  }

  grvy_printf(INFO,"[sortio][IO/XFER][%.4i]: data XFER completed\n",io_rank);

  //  delete [] fullQueueCounts;
  delete [] buf_nums;
  delete [] sendRequests;

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

// check on messages in flight and free up data transfer buffers for
// any which have completed.

void sortio_Class::CheckForSendCompletion(bool waitFlag)
{
  for(int i=0;i<numActiveSends_;i++)
    {
      if(

}

int sortio_Class::CycleDestRank()
{
  int startingDestRank = nextDestRank_;

  nextDestRank_++;

  if(nextDestRank_ >= nxfer_tasks)
    nextDestRank_ = nio_tasks;

  return(startingDestRank);
}
