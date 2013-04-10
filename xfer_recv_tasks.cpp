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

  if(!isXFERTask_)		// rules out any tasks not in XFER_COMM
    return;
  if(xferRank_ < numIoTasks_)	// rules out the sending tasks in XFER_COMM
    return;

  int recvRank = numIoTasks_;
  int iter     = 0;
  int tagXFER  = 1000;
  unsigned long int recordsPerFile;
  int messageSize;

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

  shared_memory_object sharedMem1(create_only,"syncFlags",read_write);
  shared_memory_object sharedMem2(create_only,"rawData",  read_write);

  sharedMem1.truncate(2*sizeof(int));
  sharedMem2.truncate(MAX_FILE_SIZE_IN_MBS*1024*1024*sizeof(unsigned char));

  mapped_region region1(sharedMem1,read_write);
  mapped_region region2(sharedMem2,read_write);

  syncFlags = static_cast<int *          >(region1.get_address());
  buffer    = static_cast<unsigned char *>(region2.get_address());

  syncFlags[0] = 0;		// master flag: 0=empty,1=full
  syncFlags[1] = messageSize;   //  extra data

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
      tagXFER++;

      grvy_printf(DEBUG,"[sortio][IO/Recv][%.4i] syncFlag[0] = %i, recvRank = %i (file = %i)\n",
		  xferRank_,syncFlags[0],recvRank,ifile);

      if(xferRank_ == recvRank)
	{

	  // stall briefly if last data transfer to local SORT rank is
	  // incomplete on this host

	  const int usleepInterval = 100;

#if 1
	  if(syncFlags[0] != 0)
	    for(int i=1;i<=100000;i++)
	      {
		usleep(usleepInterval);
		if(syncFlags[0] == 0)
		  {
		    grvy_printf(INFO,"[sortio][IO/Recv/IPC][%.4i] buffer xfer incomplete, stalled for"
				" %9.4e secs (iter=%i)\n",xferRank_,1.0e-6*i*usleepInterval,ifile);
		    break;
		  }
	      }
#endif

	  assert(syncFlags[0] == 0);

	  MPI_Status status;
	  grvy_printf(DEBUG,"[sortio][XFER/Recv][%.4i] initiating recv (iter=%i, tag=%i)\n",
		      xferRank_,iter,tagXFER);

	  MPI_Recv(&buffer[0],messageSize,MPI_UNSIGNED_CHAR,MPI_ANY_SOURCE,tagXFER,XFER_COMM,&status);

	  grvy_printf(INFO,"[sortio][XFER/Recv][%.4i] completed recv (iter=%i)\n",xferRank_,iter);

	  // flag buffer as being eligible for transfer via IPC

	  syncFlags[0] = 1;
	} // end if(xferRank_ == recvRank)

      recvRank++;
      if(recvRank >= numXferTasks_)
	recvRank = numIoTasks_;

      iter++;
      dataTransferred_ += messageSize;
    }

  // receive final handshake from sort task ( to guarantee SHM buffers stay in scope)

  handshake = -1;
  MPI_Status status;
  MPI_Recv(&handshake,1,MPI_INTEGER,localSortRank_,1,GLOB_COMM,&status);
  assert(handshake == 2);

  gt.EndTimer("XFER/Recv");

  grvy_printf(INFO,"[sortio][XFER/Recv][%.4i]: ALL DONE\n",xferRank_);
  grvy_printf(INFO,"[sortio][XFER/Recv][%.4i]: Total received (bytes) = %zi\n",xferRank_,dataTransferred_);

  return;
}
