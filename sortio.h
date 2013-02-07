// -*-c++-*-
//
// I/O class to aid in a large, distributed sort
//

#include <string>
#include <cassert>
#include "mpi.h"
#include "grvy.h"

class sortio_Class {

 public:
  sortio_Class();
 ~sortio_Class() {};

  void Initialize  (std::string inputfile, MPI_Comm IO_COMM);

 private:
  bool master;			// master task?
  bool initialized;		// class initialized?
  int  nio_tasks;		// number of dedicated MPI I/O tasks
  int  io_rank;			// MPI rank of local I/O task
  int  num_files_total;		// total # of input files to sort
  std::string indir;		// input directory
  
};
