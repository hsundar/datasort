#include "sortio.h"
#include "binOps/binUtils.h"
#include "omp_par/ompUtils.h"
#include "oct/octUtils.h"
#include "par/parUtils.h"
#include "gensort/sortRecord.h"

// --------------------------------------------------------------------
// manageSortTasksWork(): 
// 
// Receives input sort data from receiving XFER tasks via IPC
// and manages the overall sort process.
// 
// Operates on SORT_COMM.
// --------------------------------------------------------------------

void sortio_Class::manageSortProcess()
{
  assert(initialized_);

  int messageSize;
  std::vector<sortRecord > sortBuffer;

  if(!isSortTask_)
    return;

  // init shared-memory segments for transfer of data from first
  // SORT_COMM rank on this same host. Before we access, wait for a
  // handshake from the SHM creation tasks to guarantee existence
  // prior to accessing locally.

  if(isLocalSortMaster_)
    {

      assert(localXferRank_ == (numLocal_-1));

      int handshake = -1;
      MPI_Status status;
      grvy_printf(DEBUG,"[sortio][SORT/IPC][%.4i] posting IPC recv handshake (%i from %i)\n",sortRank_,
		  numLocal_,localXferRank_);

      MPI_Recv(&handshake,1,MPI_INTEGER,localXferRank_,1,GLOB_COMM,&status);
      assert(handshake == 1);

      grvy_printf(DEBUG,"[sortio][SORT/IPC][%.4i] IPC handshake completed\n",sortRank_);
    }

  MPI_Barrier(SORT_COMM);

  int *syncFlags;		// read/write notification flags
  unsigned char *buffer;	// buffer space to retrieve from XFER ranks

  using namespace boost::interprocess;

  shared_memory_object sharedMem1(open_only,"syncFlags",read_write);
  shared_memory_object sharedMem2(open_only,"rawData",  read_write);

  mapped_region region1(sharedMem1,read_write);
  mapped_region region2(sharedMem2,read_write);

  syncFlags = static_cast<int *          >(region1.get_address());
  buffer    = static_cast<unsigned char *>(region2.get_address());

  assert(syncFlags[0] == 0);
  messageSize = syncFlags[1];	// raw buffersize passed 

  par::Mpi_datatype <sortRecord> MPIRecordType; 
  const int numRecordsPerXfer = messageSize/sizeof(sortRecord);

  if(isMasterSort_)
    {
      grvy_printf(INFO,"[sortio][SORT] Message transfer size = %i\n",messageSize);
      grvy_printf(INFO,"[sortio][SORT] Number of records     = %i\n",numRecordsPerXfer);
    }

  // Start main processing loop; check for data from XFER tasks via
  // IPC and manage sort process. Note that XFER tasks receive data
  // cyclicly from lowest to highest receiving xfer rank; to allow for
  // maximum overlap, we do the same here on sort tasks

  int count = 0;
  int sortRankReceiving = 0;

  grvy_printf(DEBUG,"[sortio][SORT/IPC][%.4i] Beginning main processing loop \n",sortRank_);

  for(int ifile=0;ifile<numFilesTotal_;ifile++)
    {
      // process any new data via IPC, syncFlags is used for synchronization:
      // 
      //   syncFlags[0] = 0 -> empty
      //   syncFlags[0] = 1 -> full 

      int numFilesAvailTotal = 0;

      if(sortRank_ == sortRankReceiving)
	{
	  grvy_printf(DEBUG,"[sortio][SORT][%.4i] starting IPC xfer process\n",sortRank_);
	  assert(isLocalSortMaster_);

	  // Wait till data is available on this host 

	  if(syncFlags[0] != 1)
	    {
	      const int usleepInterval = 100;

	      if(syncFlags[0] != 1)
		for(int i=1;i<=2000000;i++)
		  {
		    usleep(usleepInterval);
		    if(syncFlags[0] == 1)
		      {
			grvy_printf(INFO,"[sortio][SORT/IPC][%.4i] buffer not available, waited for"
				    " %9.4e secs (iter=%i)\n",sortRank_,1.0e-6*i*usleepInterval,count);
			break;
		      }
		  }

	      assert(syncFlags[0] == 1);
	    }

	  // copy the data for sorting locally

#if 1
	  size_t memLimit = 1U*1000U*1000U*1000U;
	  
	  if(sortBuffer.size() > memLimit )
	    sortBuffer.clear();	// hack for testing 

	  for(int i=0;i<numRecordsPerXfer;i++)
	    sortBuffer.push_back(sortRecord::fromBuffer(&buffer[i*sizeof(sortRecord)]));
	  //	    sortBuffer.push_back(sortRecord::fromBuffer(&buffer[0]));
	  

#endif
	  grvy_printf(DEBUG,"[sortio][SORT/IPC][%.4i] %i re-enabling buffer (iter =%i)\n",
		      sortRank_,numFilesAvailTotal,count);

	  // verifyMode = 3 -> dump receiving data to verify data
	  // integrity throughout transfer process

	  if(verifyMode_ == 3)
	    {
	      char filename[1024];
	      sprintf(filename,"./partverify%i",ifile);
	      FILE *fp = fopen(filename,"wb");
	      assert(fp != NULL);

	      for(size_t i=0;i<sortBuffer.size();i++)
		fwrite(&sortBuffer[i],sizeof(sortRecord),1,fp);

	      fclose(fp);
	      sortBuffer.clear();
	    }

	  syncFlags[0] = 0;
	} 

      // identify next rank to receive...

      sortRankReceiving += numSortTasksPerHost_;
      if(sortRankReceiving >= numSortTasks_)
	sortRankReceiving = 0;
      
      if(isMasterSort_)
	grvy_printf(DEBUG,"[sortio][SORT][%.4i]: new sortrankReceiving = %i (iter=%i)\n",
		  sortRank_,sortRankReceiving,count);

      count++;
    }

  MPI_Barrier(SORT_COMM);

  if(isLocalSortMaster_)
    {
      int handshake = 2;

      MPI_Send(&handshake,1,MPI_INTEGER,localXferRank_,1,GLOB_COMM);
      assert(handshake == 2);
    }

  if(isMasterSort_)
    grvy_printf(INFO,"[sortio][SORT][%.4i]: ALL DONE\n",sortRank_);

  fflush(NULL);

  // Now we have the data, let's test some sorting

  //#define DO_SORT

#ifdef DO_SORT

  const unsigned int numBins = 10;  

  gt.BeginTimer("Local Sort");
  omp_par::merge_sort(&sortBuffer[0],&sortBuffer[sortBuffer.size()]);
  gt.EndTimer("Local Sort");


  gt.BeginTimer("Global Binning");
  std::vector<sortRecord> sortBins = par::Sorted_approx_Select(sortBuffer,numBins,SORT_COMM);
  gt.EndTimer("Global Binning");

  assert(sortBins.size() == numBins);

  gt.BeginTimer("Bucket and Write");
  par::bucketDataAndWrite(sortBuffer,sortBins,"/tmp/foo",SORT_COMM);
  gt.EndTimer("Bucket and Write");

  MPI_Barrier(SORT_COMM);

#endif

  return;
}
