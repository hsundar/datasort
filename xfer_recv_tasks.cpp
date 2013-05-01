#include "sortio.h"

// --------------------------------------------------------------------
// beginRecvTransferProcess(): work manager for receiving data from IO
// tasks.
// 
// This routine is only meaningful on a subset of XFER_COMM (ie. the
// receiving tasks); the convention is that the first nio_tasks in
// XFER_COMM are responsible for sending the data to the receive tasks
// running aqui.
// 
// Please see io_xfer_tasks.cpp for the companion sending logic.
// --------------------------------------------------------------------

void sortio_Class::beginRecvTransferProcess()
{
  assert(initialized_);

  if(sortMode_ <= 0)		// no overlap in naive/read-only mode
    return;

  if(!isXFERTask_)		// rules out any tasks not in XFER_COMM
    return;
  if(xferRank_ < numIoTasks_)	// rules out the sending tasks in XFER_COMM
    return;

  int recvRank = numIoTasks_;
  int iter     = 0;
  int tagXFER  = 1000;
  unsigned long int recordsPerFile;
  int messageSize;
  const int MAX_FILES_PER_MESSAGE = 10;

  // before we begin main xfer loop, we receive the # of records per
  // file which is assumed constant

  assert( MPI_Bcast(&recordsPerFile,1,MPI_UNSIGNED_LONG,0,XFER_COMM) == MPI_SUCCESS );
  messageSize = recordsPerFile*REC_SIZE;

  assert(messageSize < MAX_FILE_SIZE_IN_MBS*1024*1024);

  // also before beginning main xfer loop, we init shared-memory
  // segments for transfer to first SORT_COMM rank on this same
  // host. The segments are created on the XFER_COMM side first and
  // then we send a handshake to the corresponding SORT_COMM rank on the
  // same host.

  int *syncFlags;		// read/write notification flags
  unsigned char *buffer;	// local buffer space to receive from IO ranks

  using namespace boost::interprocess;

  shared_memory_object::remove("syncFlags");
  shared_memory_object::remove("rawData");
  shared_memory_object::remove("syncFlags2");

  shared_memory_object sharedMem1(create_only,"syncFlags", read_write); 
  shared_memory_object sharedMem2(create_only,"rawData",   read_write);
  shared_memory_object sharedMem3(create_only,"syncFlags2",read_write);

  sharedMem1.truncate(2*sizeof(int));
  sharedMem2.truncate(MAX_FILES_PER_MESSAGE*MAX_FILE_SIZE_IN_MBS*1024*1024*sizeof(unsigned char));
  sharedMem3.truncate(sizeof(shmem_xfer_sync));

  mapped_region region1(sharedMem1,read_write);
  mapped_region region2(sharedMem2,read_write);
  mapped_region region3(sharedMem3,read_write);

  syncFlags = static_cast<int *          >(region1.get_address());
  buffer    = static_cast<unsigned char *>(region2.get_address());

  void *addr = region3.get_address();
  shmem_xfer_sync *syncFlags2 = new (addr) shmem_xfer_sync;

  syncFlags[0] = 0;		// master flag: 0=empty,1=full
  syncFlags[1] = messageSize;   //  extra data

  syncFlags2->isReadyForNewData = true;

  // initiate handshake

  int handshake = 1;
  grvy_printf(DEBUG,"[sortio][XFER/IPC][%.4i] posting IPC handshake (%i to %i)...\n",
	      xferRank_,numLocal_,localSortRank_);

  MPI_Send(&handshake,1,MPI_INTEGER,localSortRank_,1,GLOB_COMM);

  // Main Recv loop; each rank takes a turn receiving file
  // contents and distributing to local SORT_COMM processes for
  // subsequent sort; recv here is blocking but can match any source,
  // the corresponding send is non-blocking

  gt.BeginTimer("XFER/Recv");
  dataTransferred_ = 0;

  for(int ifile=0;ifile<numFilesTotal_;ifile++)
    {
      //tagXFER++;
      tagXFER += 2;

      grvy_printf(DEBUG,"[sortio][IO/Recv][%.4i] syncFlag[0] = %i, recvRank = %i (file = %i)\n",
		  xferRank_,syncFlags[0],recvRank,ifile);

      if(xferRank_ == recvRank)
	{

	  // possibly stall while we wait for last data transfer to
	  // local SORT rank to complete

	  {
	    scoped_lock<interprocess_mutex> lock(syncFlags2->mutex);
	    fflush(NULL);

	    if(!syncFlags2->isReadyForNewData)
	      {
		syncFlags2->condEmpty.wait(lock);
	      }
	  }
	      
	  MPI_Status status1;
	  MPI_Status status2;

	  grvy_printf(DEBUG,"[sortio][XFER/Recv][%.4i] initiating recv (iter=%i, tag=%i)\n",
		      xferRank_,iter,tagXFER);

	  // receive info regarding size (number) of messages, followed by raw data

	  int messageSizeIncoming;

	  MPI_Recv(&messageSizeIncoming,1,MPI_INT,MPI_ANY_SOURCE,tagXFER,XFER_COMM,&status1);
	  MPI_Recv(&buffer[0],messageSizeIncoming,MPI_UNSIGNED_CHAR,status1.MPI_SOURCE,tagXFER+1,XFER_COMM,&status2);

	  grvy_printf(DEBUG,"[sortio][XFER/Recv][%.4i] completed recv (iter=%i)\n",xferRank_,iter);

	  assert(messageSize == messageSizeIncoming);

	  // verifyMode = 2 -> dump data received in XFER_COMM to compare against input

	  if(verifyMode_ == 2)
	    {
	      char filename[1024];
	      sprintf(filename,"./partfromrecv%i",iter);
	      FILE *fp = fopen(filename,"wb");
	      assert(fp != NULL);
	      
	      fwrite(&buffer[0],sizeof(char),messageSize,fp);
	      fclose(fp);
	    }

	  // flag buffer as being eligible for transfer via IPC

	  syncFlags[0] = 1;

	  {
	    scoped_lock<interprocess_mutex> lock(syncFlags2->mutex);
	    syncFlags2->isReadyForNewData = false;
	  }

	} // end if(xferRank_ == recvRank)

      recvRank++;
      if(recvRank >= numXferTasks_)
	recvRank = numIoTasks_;

      iter++;
      dataTransferred_ += messageSize;
    }

  gt.EndTimer("XFER/Recv");

  // receive final handshake from sort task ( to guarantee SHM buffers stay in scope)

  gt.BeginTimer("XFER/Wait for final sort copy");
  handshake = -1;
  MPI_Status status;
  MPI_Recv(&handshake,1,MPI_INTEGER,localSortRank_,1,GLOB_COMM,&status);
  assert(handshake == 2);

  if(xferRank_ == numIoTasks_)
    {
      grvy_printf(INFO,"[sortio][XFER/Recv][%.4i]: ALL DONE\n",xferRank_);
      grvy_printf(INFO,"[sortio][XFER/Recv][%.4i]: Total received (bytes) = %zi\n",xferRank_,dataTransferred_);
    }

  gt.EndTimer("XFER/Wait for final sort copy");

  return;
}
