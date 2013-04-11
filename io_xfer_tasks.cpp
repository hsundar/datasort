#include "sortio.h"

// --------------------------------------------------------------------
// Transfer_Tasks_Work(): work manager for data transfer tasks
// 
// This method configured to run on IO_COMM
// --------------------------------------------------------------------

void sortio_Class::Transfer_Tasks_Work()
{
  assert(initialized_);

  const int USLEEP_INTERVAL = 10000; // sleep interval if no data available to send
  int tagXFER               = 1000; // initial MPI message tag
  int numTransferredFiles   = 0;
  int count                 = 0;

  bool waitFlag;	      
  int tagLocal;
  int iter; 
  int bufNum;
  int messageSize;
  int destRank;
  MPI_Request requestHandle;

  std::vector<int> fullQueueCounts(numIoTasks_,0);
  std::vector<int> destRanks      (numIoTasks_,-1);
  std::vector<int> messageTags    (numIoTasks_,0);

  nextDestRank_ = numIoTasks_;	  // initialize first destination rank

  // before we begin main xfer loop, we wait for the first read to
  // occur on master_io rank and distribute the file size (assumed
  // constant for now)

  unsigned long int initialRecordsPerFile;

  if(isMasterIO_)
    {
      for(int iter=0;iter<20;iter++)
	{
	  if(isFirstRead_)
	    usleep(100000);
	  else
	    break;
	}

      if(isFirstRead_)
	MPI_Abort(MPI_COMM_WORLD,43);

      initialRecordsPerFile = recordsPerFile_;
    }

  // Note: # of records for 100 MB file = 1000000 

  assert( MPI_Bcast(&initialRecordsPerFile,1,MPI_UNSIGNED_LONG,0,XFER_COMM) == MPI_SUCCESS );

  messageSize = initialRecordsPerFile*REC_SIZE;

  assert(messageSize < MAX_FILE_SIZE_IN_MBS*1024*1024);

  if(isMasterIO_)
    grvy_printf(INFO,"[sortio][IO/XFER] Message size for XFERS = %i\n",messageSize);

  // Begin main data xfer loop -----------------------------------------------

  while (true)
    {
      // Transfer completed? We are done when we have received all the files *and* there
      // are no more outstanding messages active on any XFER sending processes

      if(numTransferredFiles == numFilesTotal_)
	{
	  // Before we stop, make sure all processes have empty
	  // message queues

	  int remainingLocal  = messageQueue_.size();
	  int remainingGlobal = 1;
	  
	  assert (MPI_Allreduce(&remainingLocal,&remainingGlobal,1,MPI_INTEGER,MPI_SUM,IO_COMM) == MPI_SUCCESS);

	  if(isMasterIO_)
	    grvy_printf(DEBUG,"[sortio][IO/XFER] All files sent, total # of outstanding messages = %i\n",
			remainingGlobal);
	    
	  if(remainingGlobal == 0)    // <-- finito
	    break;
	}

      // Gather up which processors have data available to send

      int localCount = fullQueue_.size();

      assert (MPI_Gather(&localCount,1,MPI_INTEGER,fullQueueCounts.data(),1,MPI_INTEGER,0,IO_COMM) == MPI_SUCCESS);

      // Assing destination ranks and message tags (master IO rank is
      // tasked with this bookkeeping)

      int numBuffersToTransfer = 0;

      if(isMasterIO_)
	{
	  for(int i=0;i<numIoTasks_;i++)
	    if(fullQueueCounts[i] >= 1)
	      {
		destRanks[i]   = CycleDestRank();
		messageTags[i] = ++tagXFER;

		numBuffersToTransfer++;
	      }
	}

      // distribute data from master_io on next set of destination ranks, message tags, etc

      assert( MPI_Bcast(&numBuffersToTransfer,1,MPI_INT,0,IO_COMM) == MPI_SUCCESS );

      if(numBuffersToTransfer > 0)
	{
	  MPI_Scatter(destRanks.data(),  1,MPI_INT,&destRank,1,MPI_INT,0,IO_COMM);
	  MPI_Scatter(messageTags.data(),1,MPI_INT,&tagLocal,1,MPI_INT,0,IO_COMM);
	}

      if( (numBuffersToTransfer > 0) && isMasterIO_)
	grvy_printf(DEBUG,"[sortio][IO/XFER] Number of hosts with data to send = %3i -> iter = %i\n",
		    numBuffersToTransfer,count);


      // Send one buffer's worth of data from each IO task that has it
      // available locally; if no processors are ready yet, iterate
      // and check again

      if( numBuffersToTransfer > 0)
	{
	  if(localCount > 0) 
	    {
	      // step 1: check that we do not have an overwhelming
	      // number of messages in flight from this host; if we
	      // are over a runtime-specified watermark, let's stall
	      // and flush the local message queue;
	      
	      grvy_printf(DEBUG,"[sortio][IO/XFER][%.4i] outstanding sends = %i\n",ioRank_,messageQueue_.size());
	      
	      checkForSendCompletion(waitFlag=true,MAX_MESSAGES_WATERMARK,iter=count);
	      
	      assert( (int)messageQueue_.size() <= MAX_MESSAGES_WATERMARK);
	      
	      // step 2: lock the oldest data transfer buffer on this processor
	      
              #pragma omp critical (IO_XFER_UPDATES_lock) // Thread-safety: all queue updates are locked
	      {
		bufNum = fullQueue_.front();
		fullQueue_.pop_front();
		grvy_printf(DEBUG,"[sortio][IO/XFER][%.4i] removed buff # %i from fullQueue\n",ioRank_,bufNum);
	      }

	      assert(buffers_[bufNum] != NULL);
	      
	      // step3: send buffers to XFER ranks asynchronously
	      
	      grvy_printf(DEBUG,"[sortio][IO/XFER][%.4i] issuing iSend to rank %i (tag = %i)\n",
			  ioRank_,destRank,tagLocal);
	      
	      MPI_Isend(&buffers_[bufNum][0],messageSize,MPI_UNSIGNED_CHAR,destRank,
			tagLocal,XFER_COMM,&requestHandle);
	      
	      // queue up these messages as being in flight
	  
	      MsgRecord message(bufNum,requestHandle);
	      messageQueue_.push_back(message);

	    }
      
	} // end if (numBuffersToTransfer > 0) 

      // All IO ranks keep track of total number of files transferred
      
      numTransferredFiles += numBuffersToTransfer;
  
      // Check for any completed messages prior to next iteration
      
      checkForSendCompletion(waitFlag=false,0,iter=count);
      
      // Sleep a bit on iterations which did not have any data to send
      
      if(numBuffersToTransfer == 0)
	usleep(USLEEP_INTERVAL);

      count++;
      
    } //  end xfer of all files

  if(isMasterIO_)
    grvy_printf(INFO,"[sortio][IO/XFER][%.4i]: data XFER COMPLETED\n",ioRank_);

  fflush(NULL);

  return;
}

// -------------------------------------------------------------------------
// checkForSendCompletion() Check on messages in flight and free up
// data transfer buffers for any which have completed. 
// -------------------------------------------------------------------------

void sortio_Class::checkForSendCompletion(bool waitFlag, int waterMark, int iter)
{

  // a NOOP if we have no outstanding messages (or if the number of
  // outstanding message is less than provided waterMark). To wait for
  // all outstanding messages, enable waitFlag and set waterMark=0
  // 
  // note that waterMark has no impact if !waitFlag

  if(messageQueue_.size() == 0 || ( (int)messageQueue_.size() <= waterMark) )
    return;

  std::list<MsgRecord>::iterator it = messageQueue_.begin();

  while(it != messageQueue_.end() )
    {
      int messageCompleted, bufNum;
      MPI_Status status;
      MPI_Request handle;

      bufNum = it->getBufNum();
      handle = it->getHandle();

      // check if send has completed

      MPI_Test(&handle,&messageCompleted,&status);

      if(messageCompleted)
	{
	  grvy_printf(DEBUG,"[sortio][IO/XFER][%.4i] message from buf %i complete (iter=%i)\n",ioRank_,bufNum,iter);
	  
	  it = messageQueue_.erase(it++);
	  
	  // re-enable this buffer for eligibility

	  addBuffertoEmptyQueue(bufNum);

	}
      else if(waitFlag)
	{
	  grvy_printf(INFO,"[sortio][IO/XFER][%.4i] Stalling for previously unfinished iSend (buf=%i,iter=%i)\n",
		      ioRank_,bufNum,iter);
	  MPI_Wait(&handle,&status);

	  it = messageQueue_.erase(it++);

	  addBuffertoEmptyQueue(bufNum);
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

  if(nextDestRank_ >= numXferTasks_)
    nextDestRank_ = numIoTasks_;

  return(startingDestRank);
}


