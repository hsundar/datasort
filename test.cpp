#include "sortio.h"

int main(int argc, char *argv[], char *env[])
{

  int num_procs;		// total # of MPI tasks available
  int num_local;		// rank of local MPI task

  sortio_Class sort_IO;		// IO subysystem 

  // [optional]: set total # of files to read via command-line;
  // otherwise, we read from input file

  if(argc > 1)
    sort_IO.Override_nFiles(atoi(argv[1]));

  sort_IO.Initialize("input.dat",MPI_COMM_WORLD);
  sort_IO.SplitComm();

  MPI_Finalize(); return 0;

  GRVY::GRVY_Input_Class iparse;

#if 0
  if(num_local == 0)
    {
      int flag;
      int nio_tasks;
      int nfiles;

      assert(iparse.Open    ("input.dat")                   != 0);
      assert(iparse.Read_Var("sortio/num_tasks",&nio_tasks) != 0);
      assert(nio_tasks <= num_procs);
    }

  sort_IO.ReadFiles();
  sort_IO.Summarize();
#endif

  MPI_Finalize();
  return 0;
}

