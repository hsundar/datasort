#include "sortio.h"

// --------------------------------------------------------------------
// Transfer_Tasks_Work(): work manager for data transfer tasks
// 
// This method configured to run on XFER_COMM
// --------------------------------------------------------------------

void sortio_Class::beginRecvTransferProcess()
{
  assert(initialized);

  // This routine is only meaningful on a subset of XFER_COMM (the
  // receiving tasks); the first nio_tasks in XFER_COMM are
  // responsible for scattering the data to these receive tasks. 
  // 
  // See file:io_xfer_tasks.cpp for the companion sending logic.

  if(!is_xfer_task)		// rules out any tasks not in XFER_COMM
    return;
  if(xfer_rank < nio_tasks)	// rules out the sending tasks in XFER_COMM
    return;

  int procMax;
  int count = 0;

  int recvRank = nio_tasks;

  unsigned char *buffer;

  buffer = (unsigned char*) calloc(MAX_FILE_SIZE_IN_MBS*1024*1024,sizeof(unsigned char));

  // Start of main Recv loop; each rank takes a turn receiving file
  // contents and distributing to local SORT_COMM processes for subsequent sort

  for(int ifile=0;ifile<num_files_total;ifile++)
    {

      if(xfer_rank == recvRank)
	grvy_printf(INFO,"[sortio][XFER/Recv][%.4i]: file %7i of %7i\n",ifile++,num_files_total);

      if(xfer_rank == recvRank)
	{
	  MPI_Status status;
	  
	  grvy_printf(INFO,"[sortio][XFER/Recv][%.4i]: initiating recv\n",xfer_rank);
	  MPI_Recv(&buffer[0],100,MPI_UNSIGNED_CHAR,MPI_ANY_SOURCE,1000,XFER_COMM,&status);
	  grvy_printf(INFO,"[sortio][XFER/Recv][%.4i]: completed recv\n",xfer_rank);

	  recvRank++;
	  if(recvRank >= nxfer_tasks)
	    recvRank = nio_tasks;
	}
      count++;
    }

  grvy_printf(INFO,"[sortio][XFER/Recv][%.4i]: ALL DONE\n",xfer_rank,count);
  MPI_Barrier(XFER_COMM);

  return;
}
