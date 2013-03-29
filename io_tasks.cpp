#include "sortio.h"

// --------------------------------------------------------------------
// IO_Tasks_Work(): work manager for IO tasks
// --------------------------------------------------------------------

void sortio_Class::IO_Tasks_Work()
{
  assert(initialized);

  int thread_id = omp_get_thread_num();
  grvy_printf(INFO,"[sortio]:IO[%i]: thread id for Master thread = %i\n",io_rank,thread_id);

    #pragma omp critical (io_region_update)
      {
	// update region_flag here
      }

  return;
}
