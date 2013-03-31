#include "sortio.h"

int main(int argc, char *argv[], char *env[])
{

  sortio_Class sort_IO;	  

  // [optional]: set total # of files to read via command-line;
  // otherwise, we read from input file

  if(argc > 1)
    sort_IO.Override_nFiles(atoi(argv[1]));

  sort_IO.Initialize("input.dat",MPI_COMM_WORLD);
  sort_IO.SplitComm();
  sort_IO.Init_Read();  
  sort_IO.BeginTransferProcess();
  // sort_IO.ReadFiles();  
  // sort_IO.Summarize();

  return 0;
}

