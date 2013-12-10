//-----------------------------------------------------------------------bl-
//--------------------------------------------------------------------------
// 
// datasort - an IO/data distribution utility for large data sorts.
//
// Copyright (C) 2013 Karl W. Schulz
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the Version 2.1 GNU Lesser General
// Public License as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc. 51 Franklin Street, Fifth Floor, 
// Boston, MA  02110-1301  USA
//
//-----------------------------------------------------------------------el-
// Top-level driver
//--------------------------------------------------------------------------

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

