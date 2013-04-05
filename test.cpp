#include "sortio.h"

int main(int argc, char *argv[], char *env[])
{

  sortio_Class sort_IO;	  

  // [optional]: set total # of files to read via command-line;
  // otherwise, we read from input file

  if(argc > 1)
    sort_IO.overrideNumFiles(atoi(argv[1]));
  if(argc > 2)
    sort_IO.overrideNumIOHosts(atoi(argv[2]));

  sort_IO.Initialize("input.dat",MPI_COMM_WORLD);
  sort_IO.SplitComm();
  MPI_Barrier(MPI_COMM_WORLD);
  sort_IO.Init_Read();  
  sort_IO.beginRecvTransferProcess();
  sort_IO.manageSortProcess();
  sort_IO.Summarize();

  return 0;
}

