#include "sortio.h"

// --------------------------------------------------------------------
// IO_Tasks_Work(): work manager for IO tasks
// --------------------------------------------------------------------

void sortio_Class::IO_Tasks_Work()
{
  assert(initialized_);

  int thread_id = omp_get_thread_num();
  grvy_printf(INFO,"[sortio]:IO[%i]: thread id for Master thread = %i\n",ioRank_,thread_id);

  ReadFiles();

  grvy_printf(INFO,"[sortio][IO/Read][%.4i]: ALL DONE with Read\n",ioRank_);

  return;
}
