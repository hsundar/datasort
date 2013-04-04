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

  unsigned char *buffer;

  buffer = (unsigned char*) calloc(MAX_FILE_SIZE_IN_MBS*1024*1024,sizeof(unsigned char));

  // Main Recv loop; each rank takes a turn receiving file
  // contents and distributing to local SORT_COMM processes for
  // subsequent sort; recv here is blocking but can match any source,
  // the corresponding send is non-blocking

  for(int ifile=0;ifile<num_files_total;ifile++)
    {
      if(xfer_rank == recvRank)
	{
	  MPI_Status status;
	  grvy_printf(DEBUG,"[sortio][XFER/Recv][%.4i]: initiating recv\n",xfer_rank);
	  MPI_Recv(&buffer[0],100,MPI_UNSIGNED_CHAR,MPI_ANY_SOURCE,1000,XFER_COMM,&status);
	  grvy_printf(INFO,"[sortio][XFER/Recv][%.4i]: completed recv\n",xfer_rank);
	}

      recvRank++;
      if(recvRank >= nxfer_tasks)
	recvRank = nio_tasks;
    }

  grvy_printf(INFO,"[sortio][XFER/Recv][%.4i]: ALL DONE\n",xfer_rank);
  MPI_Barrier(XFER_COMM);

  return;
}
