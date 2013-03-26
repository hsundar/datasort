// -*-c++-*-
//
// I/O class to aid in a large, distributed sort
//

#include <string>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <sstream>
#include "mpi.h"
#include "grvy.h"

#define REC_SIZE 100

class sortio_Class {

 public:
  sortio_Class();
 ~sortio_Class();

  void Initialize(std::string inputfile, MPI_Comm IO_COMM);
  void Override_nFiles(int nfiles);
  void ReadFiles(); 
  void SplitComm(MPI_Comm INCOMM);
  void Summarize();

 private:
  bool master;			   // master task?
  bool initialized;		   // class initialized?
  bool override_numfiles;          // Override num_files setting?
  bool random_read_offset;         // Randomly change rank ordering for read to minimize cache effects?
  int  num_files_total;		   // total # of input files to sort
  unsigned long num_records_read;  // total # of records read locally
  std::string basename;		   // input file basename
  std::string indir;		   // input directory
  GRVY::GRVY_Timer_Class gt;       // performance timer

  // Dedicated I/O tasks 

  int      nio_tasks;		   // number of dedicated raw I/O tasks
  int      io_rank;		   // MPI rank of local I/O task
  MPI_Comm IO_COMM;		   // MPI communicator for raw I/O tasks

  // Data transfer 

  int      nxfer_tasks;		   // number of dedicated data transfer tasks
  int      xfer_rank;		   // MPI rank of local data transfer task
  MPI_Comm XFER_COMM;		   // MPI communicator for data transfer tasks

  // Data sort 

  int      nsort_tasks;		   // number of dedicated sort tasks
  int      sort_rank;		   // MPI rank of local sort task
  MPI_Comm SORT_COMM;		   // MPI communicator for data sort tasks

  unsigned char rec_buf[REC_SIZE];
  
};
