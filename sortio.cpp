//
// I/O class to aid in a large, distributed sort
//

#include "sortio.h"


sortio_Class::sortio_Class() 
{
  initialized  = 0;
  master       = false;
  nio_tasks    = 0;
}

void sortio_Class::Initialize(std::string ifile, MPI_Comm IO_COMM)
{
  assert(!initialized);

  MPI_Comm_size (IO_COMM, &nio_tasks);
  MPI_Comm_rank (IO_COMM, &io_rank  );

  // Read I/O runtime controls

  if(io_rank == 0)
    {

      GRVY::GRVY_Input_Class iparse;

      printf("[sortio]: # of MPI reader tasks = %4i\n",nio_tasks);

      assert(iparse.Open    ("input.dat")                         != 0);
      assert(iparse.Read_Var("sortio/num_files",&num_files_total) != 0);
      assert(iparse.Read_Var("sortio/input_dir",&indir)           != 0);

      printf("[sortio]: --> total number of files to read = %i\n",num_files_total);
      printf("[sortio]: --> input directory               = %s\n",indir.c_str());
    }

  initialized = true;

  return; 
}

