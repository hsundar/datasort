#include "mpi.h"
#include <stdio.h>
#include <grvy.h>
#include <assert.h>
#include "sortio.h"

int main(int argc, char *argv[], char *env[])
{

  int num_procs;		// total # of MPI tasks available
  int num_local;		// rank of local MPI task

  MPI_Init      (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank (MPI_COMM_WORLD, &num_local);

  sortio_Class sort_IO;		// IO subysystem 

  // [optional]: set total # of files to read via command-line;
  // otherwise, we read from input file

  if(argc > 1)
    sort_IO.Override_nFiles(atoi(argv[1]));

  sort_IO.Initialize("input.dat",MPI_COMM_WORLD);

  GRVY::GRVY_Input_Class iparse;

  if(num_local == 0)
    {
      int flag;
      int nio_tasks;
      int nfiles;

      assert(iparse.Open    ("input.dat")                   != 0);
      assert(iparse.Read_Var("sortio/num_tasks",&nio_tasks) != 0);

      //printf("nio_tasks = %i\n",nio_tasks);
      assert(nio_tasks <= num_procs);
    }

  sort_IO.ReadFiles();
  sort_IO.Summarize();

  MPI_Finalize();
  return 0;
}

