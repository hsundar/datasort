// -*-c++-*-
//
// ----------------------------------------------------------------
// I/O class to aid in a large, distributed sort
// 
// Originally: February 2013 (karl@tacc.utexas.edu)
// ----------------------------------------------------------------

#ifndef SORTIO_H_
#define SORTIO_H_

#include "mpi.h"
#include <string>
#include <map>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <queue>
#include <list>

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include "grvy.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define REC_SIZE 100
#define INFO     GRVY_INFO
#define DEBUG    GRVY_DEBUG
#define ERROR    GRVY_INFO

// simple MPI message record - used to track active messages in flight

class MsgRecord {
  
  int bufNum_;			// buffer num in use by message
  MPI_Request handle_;		// MPI message request handle
  
public:
  MsgRecord() : bufNum_(0), handle_(0) { }
  MsgRecord(const int bufNum,const int handle) {bufNum_=bufNum; handle_ = handle;}

  // we call it a match if bufNum equals
  bool operator == (const MsgRecord &rhs) { return(rhs.bufNum_ == bufNum_ ); }

  // access
  int getBufNum() { return(bufNum_); }
  int getHandle() { return(handle_); }
};

class sortio_Class {

 public:
  sortio_Class();
 ~sortio_Class();

  void Initialize(std::string inputfile, MPI_Comm IN_COMM);
  void overrideNumFiles  (int nfiles);
  void overrideNumIOHosts(int hosts);
  void ReadFiles(); 
  void SplitComm();
  void Summarize();
  void Init_Read();
  void manageSortProcess();
  void IO_Tasks_Work();
  void RecvDataFromIOTasks();
  void Transfer_Tasks_Work();
  void beginRecvTransferProcess();
  int  CycleDestRank();
  void checkForSendCompletion(bool waitFlag, int waterMark, int iter);
  void addBuffertoEmptyQueue (int bufNum);

 private:
  bool master;			         // master MPI rank?
  bool initialized_;		         // class initialized?
  bool overrideNumFiles_;                // Override num_files setting?
  bool overrideNumIOHosts_;              // Override num_io_hosts setting?
  bool random_read_offset_;              // Randomly change rank ordering for read to minimize cache effects?
  bool mpi_initialized_by_sortio;        // did we have to call MPI_Init()?

  int  numFilesTotal_;		         // total # of input files to sort
  int  numIoHosts_;		         // total # of desired unique IO hosts
  int  numSortHosts_;			 // total # of detected sort hosts;
  int  verifyMode_;			 // verification mode (1=input data)
  int  sortMode_;                        // sort mode (0=disable)

  unsigned long numRecordsRead_;         // total # of records read locally
  unsigned long recordsPerFile_;         // # of records read locally per file (assumed constant)

  std::string fileBaseName_;	         // input file basename
  std::string inputDir_;	         // input directory

  GRVY::GRVY_Timer_Class gt;             // performance timer

  // Global MPI info (for GLOB_COMM)

  int  numTasks_;		         // total # of MPI tasks available to the class
  int  numLocal_;		         // global MPI rank for clas GLOB_COMM
  MPI_Comm GLOB_COMM;		         // global MPI communicator provided as input to the class

  // Dedicated I/O tasks 

  bool     isIOTask_;                    // MPI rank is an IO task?
  bool     isMasterIO_;			 // master IO task?
  bool     isFirstRead_;		 // first file read?
  int      numIoTasks_;		         // number of dedicated raw I/O tasks
  int      ioRank_;		         // MPI rank of local I/O task
  MPI_Comm IO_COMM;		         // MPI communicator for raw I/O tasks
				        
  int      MAX_READ_BUFFERS;	         // number of read buffers
  int      MAX_FILE_SIZE_IN_MBS;         // maximum individual file size to be read in
  int      MAX_MESSAGES_WATERMARK;       // max num of allowed messages in flight per host
  std::vector<unsigned char *> buffers_; // read buffers
  std::list <size_t> emptyQueue_;        // queue to flag empty read buffers
  std::list <size_t> fullQueue_;         // queue to flag full read buffers

  // Data transfer tasks

  bool     isXFERTask_;                  // MPI rank is a data transfer task?
  bool     isMasterXFER_;		 // master XFER task?
  bool     isReadFinished_;		 // flag for signaling raw read completion
  int      numXferTasks_;	         // number of dedicated data transfer tasks
  int      xferRank_;		         // MPI rank of local data transfer task
  int      masterXFER_GlobalRank;	 // global rank of master XFER process
  int      nextDestRank_;		 // cyclic counter for next xfer rank to send data to
  int      localSortRank_;		 // MPI rank in GLOB_COMM for the first SORT task on same host
  MPI_Comm XFER_COMM;		         // MPI communicator for data transfer tasks

  size_t   dataTransferred_;		 // amount of data transferred to receiving tasks
  std::list <MsgRecord> messageQueue_;   // in-flight message queue
  
  // Data sort tasks

  bool     isSortTask_;                 // MPI rank is a sort task?
  bool     isMasterSort_;		// master SORT task?
  bool     isLocalSortMaster_;		// master sort task on this host? (used for IPC)
  int      numSortTasks_;	        // number of dedicated sort tasks
  int      numSortTasksPerHost_;	// numer of sort tasks per host
  int      sortRank_;		        // MPI rank of local sort task
  int      localXferRank_;		// MPI rank in GLOB_COMM for the XFER task on same host
  MPI_Comm SORT_COMM;		        // MPI communicator for data sort tasks
  
};

#endif
