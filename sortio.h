// -*-c++-*-
//
// I/O class to aid in a large, distributed sort
//

#include <string>
#include <map>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <sstream>
#include "mpi.h"
#include "grvy.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define REC_SIZE 100
#define INFO  GRVY_INFO
#define DEBUG GRVY_DEBUG
#define ERROR GRVY_INFO

class sortio_Class {

 public:
  sortio_Class();
 ~sortio_Class();

  void Initialize(std::string inputfile, MPI_Comm IN_COMM);
  void Override_nFiles(int nfiles);
  void ReadFiles(); 
  void SplitComm();
  void Summarize();
  void Init_Read();
  void IO_Tasks_Work();
  void Transfer_Tasks_Work();

 private:
  bool master;			   // master task?
  bool initialized;		   // class initialized?
  bool override_numfiles;          // Override num_files setting?
  bool random_read_offset;         // Randomly change rank ordering for read to minimize cache effects?
  bool mpi_initialized_by_sortio;  // did we have to call MPI_Init()?
  int  num_files_total;		   // total # of input files to sort
  int  num_io_hosts;		   // total # of desired unique IO hosts

  unsigned long num_records_read;  // total # of records read locally
  std::string basename;		   // input file basename
  std::string indir;		   // input directory
  GRVY::GRVY_Timer_Class gt;       // performance timer

  // Global MPI info (for GLOB_COMM)

  int  num_tasks;		   // total # of MPI tasks available to the class
  int  num_local;		   // global MPI rank for clas GLOB_COMM
  MPI_Comm GLOB_COMM;		   // global MPI communicator provided as input to the class

  // Dedicated I/O tasks 

  bool     is_io_task;             // MPI rank is an IO task?
  int      nio_tasks;		   // number of dedicated raw I/O tasks
  int      io_rank;		   // MPI rank of local I/O task
  MPI_Comm IO_COMM;		   // MPI communicator for raw I/O tasks

  // Data transfer 

  bool     is_xfer_task;           // MPI rank is a data transfer task?
  int      nxfer_tasks;		   // number of dedicated data transfer tasks
  int      xfer_rank;		   // MPI rank of local data transfer task
  MPI_Comm XFER_COMM;		   // MPI communicator for data transfer tasks

  // Data sort 

  bool     is_sort_task;           // MPI rank is a sort task?
  int      nsort_tasks;		   // number of dedicated sort tasks
  int      sort_rank;		   // MPI rank of local sort task
  MPI_Comm SORT_COMM;		   // MPI communicator for data sort tasks

  unsigned char rec_buf[REC_SIZE];
  
};
