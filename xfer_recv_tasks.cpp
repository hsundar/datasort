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
  assert(initialized);

  if(!is_xfer_task)		// rules out any tasks not in XFER_COMM
    return;
  if(xfer_rank < nio_tasks)	// rules out the sending tasks in XFER_COMM
    return;

  int recvRank = nio_tasks;
  int iter     = 0;
  int tagXFER  = 1000;
  unsigned long int recordsPerFile;
  int messageSize;

  // init shared-memory segments for transfer to first SORT_COMM rank
  // on this same host. Note that we create the segments on the
  // XFER_COMM side and then send a handshake to the corresponding
  // SORT_COMM rank on the same host.

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
  syncFlags[1] = 0;		//  child flag: 0=not-xferred,1=xferred

  // initiate handshake

  const int handshake = 1;
  MPI_Send(&handshake,1,MPI_INTEGER,localSortRank_,1,GLOB_COMM);
  
  //  unsigned char *buffer;	// local buffer space for received data
  //  buffer = (unsigned char*) calloc(MAX_FILE_SIZE_IN_MBS*1024*1024,sizeof(unsigned char));

  // before we begin main xfer loop, we receive the # of records per
  // file which is assumed constant

  assert( MPI_Bcast(&recordsPerFile,1,MPI_UNSIGNED_LONG,0,XFER_COMM) == MPI_SUCCESS );
  messageSize = recordsPerFile*REC_SIZE;

  assert(messageSize < MAX_FILE_SIZE_IN_MBS*1024*1024);

  // Main Recv loop; each rank takes a turn receiving file
  // contents and distributing to local SORT_COMM processes for
  // subsequent sort; recv here is blocking but can match any source,
  // the corresponding send is non-blocking

  //  printf("initialized XFER/Recv on global rank %i\n",num_local);

  gt.BeginTimer("XFER/Recv");
  dataTransferred_ = 0;

  for(int ifile=0;ifile<num_files_total;ifile++)
    {
      tagXFER++;

      if(xfer_rank == recvRank)
	{

#if 0
	  // stall briefly if last data transfer is incomplete on this host

	  if(syncFlags[0] != 0)
	    for(int i=0;i<5000;i++)
	      {
		grvy_printf(INFO,"[sortio][IO/Recv/IPC][%.4i] buffer xfer incomplete, stalling....(iter=%i)\n",
			    xfer_rank,ifile);
		usleep(100000);
		if(syncFlags[0] == 0)
		  break;
	      }

	  assert(syncFlags[0] == 0);

#endif

	  MPI_Status status;
	  grvy_printf(INFO,"[sortio][XFER/Recv][%.4i] initiating recv (iter=%i, tag=%i)\n",
		      xfer_rank,iter,tagXFER);

	  MPI_Recv(&buffer[0],messageSize,MPI_UNSIGNED_CHAR,MPI_ANY_SOURCE,tagXFER,XFER_COMM,&status);

	  grvy_printf(INFO,"[sortio][XFER/Recv][%.4i] completed recv (iter=%i)\n",xfer_rank,iter);

	  // flag buffer as being eligible for transfer via IPC

	  syncFlags[0] = 1;
	}

      recvRank++;
      if(recvRank >= nxfer_tasks)
	recvRank = nio_tasks;

      iter++;
      dataTransferred_ += messageSize;
    }

  gt.EndTimer("XFER/Recv");

  grvy_printf(INFO,"[sortio][XFER/Recv][%.4i]: ALL DONE\n",xfer_rank);
  grvy_printf(INFO,"[sortio][XFER/Recv][%.4i]: Total received (bytes) = %zi\n",xfer_rank,dataTransferred_);

  return;
}
