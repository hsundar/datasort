#include "mpi.h"
#include <stdio.h>
#include <grvy.h>
#include <assert.h>

int main(int argc, char *argv[], char *env[])
{

  int num_procs;		// total # of MPI tasks available
  int num_local;		// rank of local MPI task

  MPI_Init      (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank (MPI_COMM_WORLD, &num_local);

  // Read I/O runtime controls
  
  GRVY::GRVY_Input_Class iparse;

  if(num_local == 0)
    {
      assert(iparse.Open("input.dat") != 0);
    }

  


}

