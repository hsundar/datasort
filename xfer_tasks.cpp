#include "sortio.h"

// --------------------------------------------------------------------
// Transfer_Tasks_Work(): work manager for data transfer tasks
// --------------------------------------------------------------------

void sortio_Class::Transfer_Tasks_Work()
{
  assert(initialized);

  usleep(3000000); 
  int thread_id = omp_get_thread_num(); 
  printf("[%i]: thread id for Master thread = %i\n",io_rank,thread_id);

#pragma omp critical (io_region_update)
  {
    // update region_flag here
  }

  return;
}
