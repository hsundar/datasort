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
  int numTransferedFiles = 0;

  //  usleep(3000000); 
  int thread_id = omp_get_thread_num(); 
  //  printf("[%i]: thread id for Master thread = %i\n",io_rank,thread_id);

  std::vector<int> fullQueueCounts(nio_tasks,0);

  // Start main data transfer loop; we continue until all files have
  // been read from IO_COMM and are transferred to SORT_COMM

  int count = 0;
  std::vector<int>::iterator itMax;
  int procMax;
  int maxCount;

  while (numTransferedFiles < num_files_total)
    {

      // Check which processor has the most data available

      const int localCount = fullQueue_.size();

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

      count++;

      if(count > 10)
	numTransferedFiles = num_files_total;// hack

      // Scatter data from the processor with the most data on hand;
      // if no processors are ready yet, iterate and check again

      if(maxCount > 0)
	{
	  // step 1: let all the xfer tasks know who will be scattering data

	  assert( MPI_Bcast(&procMax, 1,MPI_INTEGER,0,XFER_COMM) == MPI_SUCCESS );

	  // step 2: distribute data from procMax using special
	  // scatter group with procmax as the leader

	}
      else
	usleep(USLEEP_INTERVAL);

    } //  end xfer of all files

  // Send notice to xfer tasks to let them know we are all done

  const int signalCompletion = -1;

  assert( MPI_Bcast(&signalCompletion, 1,MPI_INTEGER,0,XFER_COMM) == MPI_SUCCESS );

#pragma omp critical (io_region_update)
  {
    // update region_flag here
  }

  grvy_printf(INFO,"[sortio][IO/XFER][%.4i]: data XFER completed\n",io_rank);

  //  delete [] fullQueueCounts;

  return;
}
