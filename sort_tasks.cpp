#include "sortio.h"
#include "binOps/binUtils.h"
#include "omp_par/ompUtils.h"
#include "oct/octUtils.h"
#include "par/sort_profiler.h"
#include "par/parUtils.h"
#include "gensort/sortRecord.h"

#define _PROFILE_SORT

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
      grvy_printf(INFO,"[sortio][SORT/IPC][%.4i] posting IPC recv handshake (%i from %i)\n",sortRank_,
		  numLocal_,localXferRank_);

      MPI_Recv(&handshake,1,MPI_INT,localXferRank_,1,GLOB_COMM,&status);
      assert(handshake == 1);
      
      grvy_printf(INFO,"[sortio][SORT/IPC][%.4i] IPC handshake completed\n",sortRank_);
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

  bool needBinning              = true;
  const int numBins             = 10;  
  const int numRecordsPerXfer   = messageSize/sizeof(sortRecord);
  //  const size_t binningWaterMark = 2*numRecordsPerXfer;
  const size_t binningWaterMark = numRecordsPerXfer*numSortHosts_;

  char tmpFilename[1024];	     // location for tmp file
  std::vector<sortRecord> sortBins;  // binning buckets
  std::vector< std::vector<int> > tmpWriteSizes;
  
  if(isMasterSort_)
    {
      grvy_printf(INFO,"[sortio][SORT] Message transfer size = %i\n",messageSize);
      grvy_printf(INFO,"[sortio][SORT] Number of records     = %i\n",numRecordsPerXfer);
    }

  // Start main processing loop; check for data from XFER tasks via
  // IPC and manage sort process. Note that XFER tasks receive data
  // cyclicly from lowest to highest receiving xfer rank; to allow for
  // maximum overlap, we do the same here on sort tasks

  int count            = 0;
  int outputCount      = 0;
  int numFilesReceived = 0;

  grvy_printf(DEBUG,"[sortio][SORT/IPC][%.4i] Beginning main processing loop \n",sortRank_);

  MPI_Barrier(SORT_COMM);

  while(numFilesReceived < numFilesTotal_)
    {
      // process any new data via IPC, syncFlags is used for synchronization (0=empty,1=full)

      int localDataAvail  = 0;
      int globalDataAvail = 0;

      // check for any data available

      if(isLocalSortMaster_)
	if(syncFlags[0] == 1)
	  {
	    
	    if(sortMode_ > 0)
	      {
		gt.BeginTimer("Sort/Copy");
		for(int i=0;i<numRecordsPerXfer;i++)
		  sortBuffer.push_back(sortRecord::fromBuffer(&buffer[i*sizeof(sortRecord)]));
		gt.EndTimer("Sort/Copy");
	      }

	    grvy_printf(DEBUG,"[sortio][SORT/IPC][%.4i] re-enabling buffer (iter =%i)\n",sortRank_,count);

	    localDataAvail = 1;
	    syncFlags[0]   = 0;
	  }

      assert (MPI_Allreduce(&localDataAvail,&globalDataAvail,1,MPI_INT,MPI_SUM,SORT_COMM) == MPI_SUCCESS);

      numFilesReceived += globalDataAvail;

      if(isMasterSort_)
	grvy_printf(DEBUG,"[sortio][SORT][%.4i]: numFilesReceived = %i\n",sortRank_,numFilesReceived);

      // Check if we have enough data to do initial binning

      if(sortMode_ > 0 && needBinning)
	{
	  int numRecordsLocal  = sortBuffer.size();
	  int numRecordsGlobal = 0;

	  //assert (MPI_Allreduce(&numRecordsLocal,&numRecordsGlobal,1,MPI_INT,MPI_MIN,SORT_COMM) == MPI_SUCCESS);
	  assert (MPI_Allreduce(&numRecordsLocal,&numRecordsGlobal,1,MPI_INT,MPI_SUM,SORT_COMM) == MPI_SUCCESS);

	  if(numRecordsGlobal >= binningWaterMark)
	    {
	      if(isMasterSort_)
		grvy_printf(INFO,"[sortio][SORT][%.4i]: Doing initial sort binning\n",sortRank_);

	      gt.BeginTimer("Local Sort");
	      omp_par::merge_sort(&sortBuffer[0],&sortBuffer[sortBuffer.size()]);
	      gt.EndTimer("Local Sort");
		
	      gt.BeginTimer("Global Binning");
	      sortBins = par::Sorted_approx_Select(sortBuffer,numBins-1,SORT_COMM);
	      gt.EndTimer("Global Binning");

	      needBinning = false;
	    }
	}

      // Save any new data

      if( (sortMode_ > 0) && !needBinning && (globalDataAvail > 0) )
	{
	  // Extra heuristics to buffer up a bit of data

	  //if( (numFilesReceived < (numFilesTotal_ - 2*numSortHosts_)) && (globalDataAvail < numSortHosts_))
	  //	    break;

	  sprintf(tmpFilename,"/tmp/utsort/%i/proc%.4i",outputCount,sortRank_);

	  if(isLocalSortMaster_)
	    grvy_check_file_path(tmpFilename);

	  MPI_Barrier(SORT_COMM);

	  grvy_printf(INFO,"[sortio][SORT][%.4i]: Size of sortBuffer for bucket = %zi\n",sortRank_,sortBuffer.size());

	  gt.BeginTimer("Bucket and Write");
	  std::vector<int> writeCounts = par::bucketDataAndWrite(sortBuffer,sortBins,tmpFilename,SORT_COMM);
	  gt.EndTimer("Bucket and Write");	    

	  assert(writeCounts.size() == numBins );
	  tmpWriteSizes.push_back(writeCounts);

	  sortBuffer.clear();

	  outputCount++;
	}

      count++;
    }

  MPI_Barrier(SORT_COMM);

  // Tally up all the binned records written

  if(sortMode_ > 0)
    {
      assert(tmpWriteSizes.size() == outputCount);
      int numWrittenLocal  = 0;
      int numWrittenGlobal = 0;
      
      for(size_t i=0;i<tmpWriteSizes.size();i++)
	for(int j=0;j<numBins;j++)
	  numWrittenLocal += tmpWriteSizes[i][j];
      
      assert (MPI_Reduce(&numWrittenLocal,&numWrittenGlobal,1,MPI_INT,MPI_SUM,0,SORT_COMM) == MPI_SUCCESS);
      
      if(isMasterSort_)
	grvy_printf(INFO,"[sortio][FINALSORT] Total # of records written = %i\n",numWrittenGlobal);
      
      assert(numWrittenGlobal = (numFilesReceived*numRecordsPerXfer));

      // Re-read binned data to complete final sort

      for(int ibin=0;ibin<numBins;ibin++)
	{
	  if(isMasterSort_)
	    grvy_printf(INFO,"[sortio][FINALSORT] Working on bin %i of %i...\n",ibin,numBins);
	  
	  int    numTotal   = 0;
	  size_t startIndex = 0;
	  
	  for(int iter=0;iter<outputCount;iter++)
	    numTotal += tmpWriteSizes[iter][ibin];
	  
	  std::vector<sortRecord> binnedData(numTotal);
	  
	  gt.BeginTimer("Read Temp Data");
	  
	  for(int iter=0;iter<outputCount;iter++)
	    {
	      int numLocal = tmpWriteSizes[iter][ibin];
	      
	      sprintf(tmpFilename,"/tmp/utsort/%i/proc%.4i_%.3i.dat",iter,sortRank_,ibin);
	      FILE *fp = fopen(tmpFilename,"rb");
	      if(fp == NULL)
		grvy_printf(ERROR,"[sortio][FINALSORT][%.4i] Unable to access file %s\n",sortRank_,tmpFilename);

	      assert(fp != NULL);
	      
	      fread(&binnedData[startIndex],sizeof(sortRecord),numLocal,fp);
	      fclose(fp);
	      startIndex += numLocal;
	    }
	  
	  gt.EndTimer("Read Temp Data");
	  assert(startIndex == numTotal);

	  std::vector<sortRecord> out;

	  gt.BeginTimer("Final Sort");

#if 1
	  //par::HyperQuickSort(binnedData, out, SORT_COMM);
          par::HyperQuickSort_kway(binnedData, out, SORT_COMM);
	  gt.EndTimer("Final Sort");
	  
	  assert(binnedData.size() == out.size());
	  
	  sprintf(tmpFilename,"./final_sort/part_bin%.3i_p%.5i",ibin,sortRank_);
	  grvy_check_file_path(tmpFilename);
	  
	  FILE *fp = fopen(tmpFilename,"wb");
	  assert(fp != NULL);
	  
	  //fwrite(&binnedData[0],sizeof(sortRecord),binnedData.size(),fp);
	  fwrite(&out[0],sizeof(sortRecord),out.size(),fp);
	  fclose(fp);
#endif
	}

    }

  // wasn't that easy?

  MPI_Barrier(SORT_COMM);

  if(isLocalSortMaster_)
    {
      int handshake = 2;

      MPI_Send(&handshake,1,MPI_INTEGER,localXferRank_,1,GLOB_COMM);
      assert(handshake == 2);
    }

  if(isMasterSort_)
    {
      grvy_printf(INFO,"[sortio][SORT][%.4i]: ALL DONE\n",sortRank_);
      gt.Summarize();
    }

  fflush(NULL);

  return;
}
