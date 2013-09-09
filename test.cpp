#include "sortio.h"

int main(int argc, char *argv[], char *env[])
{

  sortio_Class sort_IO;	  

  // [optional]: set total # of files to read via command-line;
  // otherwise, we read from input file

  std::string input_file ("input.dat");

  if(argc > 1)
    input_file = argv[1];
  if(argc > 2)
    sort_IO.overrideNumFiles(atoi(argv[2]));
  if(argc > 3)
    sort_IO.overrideNumIOHosts(atoi(argv[3]));
  if(argc > 4)
    sort_IO.overrideNumSortThreads(atoi(argv[4]));
  if(argc > 5)
    sort_IO.overrideNumSortGroups(atoi(argv[5]));

  sort_IO.Initialize(input_file,MPI_COMM_WORLD);
  sort_IO.SplitComm();
  sort_IO.Init_Read();  
  sort_IO.beginRecvTransferProcess();
  sort_IO.manageSortProcess();
  sort_IO.Summarize();

  return 0;
}

