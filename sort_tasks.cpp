#include "sortio.h"

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

  if(sortMode_ <= 0)		// no overlap in naive/read-only mode
    return;

  int messageSize;
  std::vector<sortRecord > sortBuffer;

  if(!isSortTask_)
    return;

  // init shared-memory segments for transfer of data from first
  // SORT_COMM rank on this same host. Before we access, wait for a
  // handshake from the SHM creation tasks to guarantee existence
  // prior to accessing locally.

  MPI_Barrier(SORT_COMM);

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

#if 0
  if(syncFlags[0] != 0)
    printf("invalid sync flag on sort rank %i (value = %i)\n",sortRank_,syncFlags[0]);
  
  assert(syncFlags[0] == 0);
#endif

  messageSize = syncFlags[1];	// raw buffersize passed 

  bool needBinning              = true;
  const int numRecordsPerXfer   = messageSize/sizeof(sortRecord);
  const size_t binningWaterMark = 1*numSortHosts_;
  //  const size_t binningWaterMark = numSortHosts_/10;

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
  int maxDirNum        = 0;     // flag to keep track of max directory number created for tmp files
  int activeBin        = 0;	// currently active binning communicator
  int maxPerBin        = 0;     // max size of tmp data per host from any bin

  if(isMasterSort_)
    grvy_printf(INFO,"[sortio][SORT/IPC][%.4i] Beginning main processing loop \n",sortRank_);

  MPI_Barrier(SORT_COMM);

  gt.BeginTimer("Sort/Recv");

  // Perform initial binning to determine bucket thresholds (this is only done
  // once and occurs only on first BIN group)

  int fileOnHandFirst = 0;

  if(isBinTask_[0])
    while(true)
      {
	int localData  = 0;
	int globalData = 0;

	if(syncFlags[0] == 1)	// indicates data available
	  {
	    
	    if(sortMode_ > 1)
	      {
		grvy_printf(INFO,"[sortio][SORT/IPC][%.4i] found data to copy\n",sortRank_);
		gt.BeginTimer("Sort/Copy");
		for(int i=0;i<numRecordsPerXfer;i++)
		  sortBuffer.push_back(sortRecord::fromBuffer(&buffer[i*sizeof(sortRecord)]));
		gt.EndTimer("Sort/Copy");
	      }

	    grvy_printf(DEBUG,"[sortio][SORT/IPC][%.4i] re-enabling buffer (iter =%i)\n",sortRank_,count);

	    localData    = 1;
	    syncFlags[0] = 0;

	    assert (MPI_Allreduce(&localData,&globalData, 1,MPI_INT,MPI_SUM,BIN_COMMS_[0]) == MPI_SUCCESS);
	
	    numFilesReceived += globalData;

	    int numRecordsLocal  = sortBuffer.size();
	    int numRecordsGlobal = 0;

	    if( (globalData >= binningWaterMark) )
	      {
		if(sortMode_ > 1)
		  {
		    if(isMasterSort_)
		      grvy_printf(INFO,"[sortio][SORT/BIN][%.4i] %i files gathered, starting local binning...\n",
				  sortRank_,numFilesReceived);

		      grvy_printf(INFO,"[sortio][SORT][%.4i]: # of files available = %zi\n",
				  sortRank_,sortBuffer.size());

		    gt.BeginTimer("Local Sort");
		    omp_par::merge_sort(&sortBuffer[0],&sortBuffer[sortBuffer.size()]);
		    gt.EndTimer("Local Sort");

		    gt.BeginTimer("Global Binning");
		    sortBins = par::Sorted_approx_Select(sortBuffer,numSortBins_-1,BIN_COMMS_[0]);
		    //sortBins = par::Sorted_approx_Select_old(sortBuffer,numSortBins_-1,BIN_COMMS_[0]);
		    gt.EndTimer("Global Binning");
		    
		    assert(sortBins.size() == numSortBins_ - 1 );
		    
		    outputCount = 0; // first write
		    
		    sprintf(tmpFilename,"/tmp/utsort/%i/proc%.4i",outputCount,binRanks_[0]);
		    grvy_printf(DEBUG,"[%.4i] saving to file %s\n",sortRank_,tmpFilename);
		    
		    grvy_check_file_path(tmpFilename);
		    grvy_printf(DEBUG,"[sortio][SORT][%.4i]: Size of sortBuffer for bucket = %zi\n",sortRank_,
				sortBuffer.size());
		    
		    gt.BeginTimer("Bucket and Write");
		    std::vector<int> writeCounts = par::bucketDataAndWrite(sortBuffer,sortBins,
									   tmpFilename,BIN_COMMS_[0]);
		    gt.EndTimer("Bucket and Write");	    
		    
		    assert(writeCounts.size() == numSortBins_ );
		    tmpWriteSizes.push_back(writeCounts);
		    
		    sortBuffer.clear();
		  }

		break;
	      }
	  }
      }

  MPI_Barrier(SORT_COMM);
  gt.EndTimer("Sort/Recv");

  if(isMasterSort_)
    printf("[sortio][SORT][%.4i]: First bucket binning complete\n",sortRank_);

  // distribute the sorting bins to all of SORT_COMM for use by all
  // BIN groups (even though BIN1 already has this data, we resend
  // using a bcast to SORT_COMM for convenience

  if(sortMode_ > 1)
    {
      if(!isBinTask_[0])
	sortBins.resize(numSortBins_-1);

      std::vector<sortRecord> binOrig;
      if(binNum_ == 0 && binRanks_[0] == 1)
	binOrig = sortBins;
      
      MPI_Datatype MPISORT_TYPE = par::Mpi_datatype<sortRecord>::value();

      assert( MPI_Bcast( sortBins.data(),sortBins.size(), MPISORT_TYPE,0,SORT_COMM) == MPI_SUCCESS);

      // double check 

      if(binNum_ == 0 && binRanks_[0] == 1)
	for(size_t i=0;i<binOrig.size();i++)
	  assert(binOrig[i] == sortBins[i]);
    }

  // Transfer ownership to next BIN group

  if(isBinTask_[0] && numSortGroups_ > 1)
    cycleBinGroup(numFilesReceived,0);
  
  int iterCount = 0;
  if(isBinTask_[0])
    iterCount++;	// <-- Group 0 already did first write above

  // Commence with binning process as new data comes in; cycle through
  // BIN communicators to provide an asychnronous mechanism for saving
  // the temporary data to disk (only tasks associated with a BIN
  // group participate)

  gt.BeginTimer("Sort/Recv");

  while(numFilesReceived < numFilesTotal_)
    {
      if(binNum_ < 0)	
	break;

      if(numSortGroups_ > 1)
	numFilesReceived = waitForActivation();

      // tear-down procedure, notify remaining bin groups that we have
      // processed all files, we send a negative count here and count
      // down till the final group is terminated.

      if( (numFilesReceived == numFilesTotal_) )
	{
	  if(numSortGroups_ >= 3)
	    {
	      int terminateCounter = numSortGroups_ - 2;  // we are all done when this counter = -1
	      cycleBinGroup(-terminateCounter,binNum_);
	    }	  
	  break;
	}
      else if(numFilesReceived < -1) // we are still tearing down
	{
	  cycleBinGroup(numFilesReceived++,binNum_);
	  break;
	}
      else if(numFilesReceived == -1) // final group reached
	break;

      int count = 0;
      bool isActiveMaster = false;
      int filesOnHand = 0;

      if(binRanks_[binNum_] == 0)
	isActiveMaster = true;

      // loop until this BIN group has sufficient data available

      while(true)
	{
	  if(isActiveMaster)
	    grvy_printf(DEBUG,"[sortio][SORT/BIN][%.4i] Group %i looking for new data (iter = %i)\n",
			sortRank_,binNum_,count);

	  int localData  = 0;
	  int localSize  = 0;

	  int globalData = 0;
	  int globalSize = 0;

	  int dataLocal [2];
	  int dataGlobal[2];

	  if(syncFlags[0] == 1)
	    {
	      if(sortMode_ > 1)
		{
		  gt.BeginTimer("Sort/Copy");
		  for(int i=0;i<numRecordsPerXfer;i++)
		    sortBuffer.push_back(sortRecord::fromBuffer(&buffer[i*sizeof(sortRecord)]));
		  gt.EndTimer("Sort/Copy");
		}
	      
	      grvy_printf(DEBUG,"[sortio][SORT/IPC][%.4i] re-enabling buffer (iter =%i)\n",sortRank_,count);
	      
	      localData    = 1;
	      localSize    = sortBuffer.size()/numRecordsPerXfer;
	      syncFlags[0] = 0;

	    }

	  dataLocal[0] = localData;
	  dataLocal[1] = localSize;

	  assert (MPI_Allreduce(dataLocal,dataGlobal,2,MPI_INT,MPI_SUM,BIN_COMMS_[binNum_]) == MPI_SUCCESS);

	  globalData = dataGlobal[0];

	  numFilesReceived += globalData;
	  filesOnHand      += globalData;

	  // Continue with local binning and temporary file writes (1 per host)

	  int threshold = 0;

	  //if(numFilesReceived < (numFilesTotal_ - 2*numSortHosts_) )
	  if(numFilesReceived < (numFilesTotal_ - 1*numSortHosts_) )
	    {
	      //threshold = numSortHosts_ / 10;
	      //threshold = numSortHosts_ / 4;
	      //threshold = numSortHosts_ / 2;
	      //threshold = numSortHosts_ / 1.5;
	      //threshold = numSortHosts_;
	      threshold = numSortHosts_-1;
	      //threshold = numSortBins_;
	      //threshold = 1.5*numSortHosts_;
	    }
	  else 
	    {
	      //threshold = numFilesTotal_ - numFilesReceived - 1;
	      threshold = numFilesTotal_ - numFilesReceived;
	      //threshold = 0;
	    }

	  //#define OLD
#ifdef OLD
	  if( globalData > threshold )
#else
	  if( filesOnHand > threshold )
#endif
	    {

	      // Transfer ownership to next BIN comm

	      if(numSortGroups_ > 1)
		cycleBinGroup(numFilesReceived,binNum_);

	      if(sortMode_ > 1)
		{

		  if(binRanks_[binNum_] == 0)
		    grvy_printf(INFO,"[sortio][SORT/BIN][%.4i] %i files gathered, starting local binning...\n",
				sortRank_,filesOnHand);

		  outputCount = iterCount*numSortGroups_ + binNum_;
		  
		  sprintf(tmpFilename,"/tmp/utsort/%i/proc%.4i",outputCount,binRanks_[binNum_]);
		  grvy_printf(DEBUG,"[%.4i] saving to file %s\n",sortRank_,tmpFilename);
		  
		  grvy_check_file_path(tmpFilename);
		  grvy_printf(DEBUG,"[sortio][SORT][%.4i]: Size of sortBuffer for bucket = %zi\n",sortRank_,
			      sortBuffer.size());
		  
		  gt.BeginTimer("Bucket and Write");
		  std::vector<int> writeCounts = par::bucketDataAndWrite(sortBuffer,sortBins,
									 tmpFilename,BIN_COMMS_[binNum_]);
		  gt.EndTimer("Bucket and Write");	    
		  
		  assert(writeCounts.size() == numSortBins_ );
		  tmpWriteSizes.push_back(writeCounts);
	      
		  sortBuffer.clear();
		}
	    }

#ifdef OLD
	  if(globalData > threshold)
	    break;
#else
	  if(filesOnHand > threshold)
	    {
	      filesOnHand = 0;
	      break;
	    }
#endif


	  count++;
	} // end while(true);

      iterCount++;

    } // end main loop

  gt.EndTimer("Sort/Recv");

  sortBuffer.clear();
  sortBins.clear();

  if(binNum_ >= 0)
    grvy_printf(DEBUG,"[sortio][BIN][%.4i] Local binning complete\n",sortRank_);

  //MPI_Barrier(SORT_COMM);

  if(isMasterSort_)
    grvy_printf(INFO,"[sortio][SORT][%.4i]: numFilesReceived = %i\n",sortRank_,numFilesTotal_);

  // Tally up all the binned records written

  if(sortMode_ > 2)
    {

      int maxDirNumLocal = outputCount;
      maxDirNum          = 0;

      assert (MPI_Allreduce(&maxDirNumLocal,&maxDirNum,1,MPI_INT,MPI_MAX,SORT_COMM) == MPI_SUCCESS);

      int numWrittenLocal  = 0;
      int numWrittenGlobal = 0;

      for(size_t i=0;i<tmpWriteSizes.size();i++)
	for(int j=0;j<tmpWriteSizes[i].size();j++)
	  numWrittenLocal += tmpWriteSizes[i][j];
      
      assert (MPI_Reduce(&numWrittenLocal,&numWrittenGlobal,1,MPI_INT,MPI_SUM,0,SORT_COMM) == MPI_SUCCESS);

      int maxPerBinLocal = 0;

      for(int ibin=0;ibin<numSortBins_;ibin++)
	{
	  int count = 0;
	  for(size_t i=0;i<tmpWriteSizes.size();i++)
	    count += tmpWriteSizes[i][ibin];

	  if(count > maxPerBinLocal)
	    maxPerBinLocal = count;
	}

      assert (MPI_Allreduce(&maxPerBinLocal,&maxPerBin,1,MPI_INT,MPI_MAX,SORT_COMM) == MPI_SUCCESS);

      //      assert(numWrittenGlobal = (numFilesTotal_*numRecordsPerXfer));
      
      if(isMasterSort_)
	grvy_printf(INFO,"[sortio][FINALSORT] Max records for single bin = %i\n",maxPerBinLocal);
    }

  // We are almost there: re-read binned data to complete final sort

  if(sortMode_ > 2)
    {

      MPI_Barrier(SORT_COMM);

      if(isMasterSort_)
	{
	  grvy_printf(INFO,"[sortio][FINALSORT][%.4i]: Starting final sort\n",sortRank_);
	  grvy_printf(INFO,"[sortio][SORT] Elapsed Total time = %e\n",gt.ElapsedGlobal());
	  grvy_printf(INFO,"[sortio][SORT] Total Time for data receive process = %e\n",
		      gt.ElapsedSeconds("Sort/Recv") + 
		      gt.ElapsedSeconds("Bucket and Write") + 
		      gt.ElapsedSeconds("Global Binning") + 
		      gt.ElapsedSeconds("Local Sort") + 
		      gt.ElapsedSeconds("Sort/Copy"));
	}
      
      int outputLocal  = tmpWriteSizes.size();
      int outputCount  = 0;
      const int tag1   = 4200;
      const int tag2   = 42000;

      if(isMasterSort_)
	{
	  grvy_printf(INFO,"[sortio][FINALSORT] Total # distinct outputs (per bin) = %i\n",maxDirNum);
	  fflush(NULL);
	}

      int numRecordsReadFromTmp = 0;

      numSortGroups_ = 2;	// hack for testing to avoid mem overflow

      //      if(isBinTask_[0])
	{

	  bool first_entry  = true;
	  bool first_entry2 = true;

	  for(int ibin=0;ibin<numSortBins_;ibin++)
	    {

	      // define current master group

	      int sortGroup = ibin % numSortGroups_;

	      if(!isBinTask_[sortGroup])
		continue;
	      
	      // non-master tasks wait for notification to proceed

	      if(numSortGroups_ > 1)
		if(!first_entry || (first_entry && sortGroup != 0) )
		  {
		    MPI_Status status;
		    int activate;
		    int recvRank  = sortRank_ -1;
		    
		    if(sortGroup == 0)
		      recvRank = sortRank_ + (numSortGroups_ - 1);
		    
		    if(binRanks_[sortGroup] == 0)
		      grvy_printf(INFO,"[sortio][FINALSORT] Group %i waiting to begin local read...\n",sortGroup);
		    
		    assert( MPI_Recv(&activate,1,MPI_INT,recvRank,tag1+ibin,SORT_COMM,&status) == MPI_SUCCESS);
		  }

	      first_entry = false;
		   
	      if(isBinTask_[sortGroup])
		{
		  if(binRanks_[sortGroup] == 0)
		    grvy_printf(INFO,"[sortio][FINALSORT] Group %i starting read for bin %i of %i...\n",sortGroup,
				ibin,numSortBins_);

		  // allocate buffer space for reading in tmp data (max size computed above).

		  std::vector<sortRecord> binnedData(maxPerBin*numSortBins_);
      
		  int recordsPerBinLocal = 0;
		  int recordsPerBinMax   = 0;
		  size_t startIndex      = 0;

		  gt.BeginTimer("Read Temp Data");
	      
		  int myCount = 0;
	      
		  for(int iter=0;iter<=maxDirNum;iter++)
		    {
		      sprintf(tmpFilename,"/tmp/utsort/%i/proc%.4i_%.3i.dat",iter,binRanks_[sortGroup],ibin);
		      
		      FILE *fp = fopen(tmpFilename,"rb");
		      if(fp == NULL)
			grvy_printf(ERROR,"[sortio][FINALSORT][%.4i] Unable to access file %s\n",
				    binRanks_[sortGroup],tmpFilename);
		      
		      assert(fp != NULL);
		      grvy_printf(DEBUG,"[sortio][FINALSORT][%.4i] Group %i Read in file %s\n",
				  binRanks_[sortGroup],sortGroup,tmpFilename);
		  
		      myCount = 0;
		      while(fread(&binnedData[startIndex],sizeof(sortRecord),1,fp) == 1)
			{
			  startIndex++;
			  assert(startIndex < binnedData.size());
			  myCount++;
			}
		      
		      if(!feof(fp))
			grvy_printf(ERROR,"[sortio][FINALSORT][%.4i] Warning: missed eof for %s\n",
				    binRanks_[sortGroup],tmpFilename);

		      //assert(feof(fp));

		      fclose(fp);

		      numRecordsReadFromTmp += myCount;

		    } // end loop over maxDirNum

		  gt.EndTimer("Read Temp Data");

		  // notifiy next sort group to commence read...

		  if(numSortGroups_ > 1)
		    {
		      int destRank = sortRank_ + 1;
		      int notify = 0;
		      if(sortGroup == numSortGroups_ - 1)
			destRank = sortRank_ - (numSortGroups_ - 1);
		      
		      if(binRanks_[sortGroup] == 0)
			grvy_printf(INFO,"[sortio][FINALSORT] Group %i notifying next group to begin read...\n",
				    sortGroup);
		      fflush(NULL);

		      if(ibin < (numSortBins_-1))
			assert(MPI_Send(&notify,1,MPI_INT,destRank,tag1+ibin+1,SORT_COMM) == MPI_SUCCESS);
		    }

		  // do final sort

		  int globalRead;
		  binnedData.resize(startIndex);

		  fflush(NULL);

		  omp_set_num_threads(numSortThreads_);

		  MPI_Barrier(BIN_COMMS_[sortGroup]);
		  gt.BeginTimer("Final Sort");

		  if(binRanks_[sortGroup] == 0)
		    grvy_printf(INFO,"[sortio][FINALSORT] Group %i calling final sort with input size = %zi\n",
				sortGroup,binnedData.size());

		  fflush(NULL);
	      
		  //par::HyperQuickSort_kway(binnedData, out, BIN_COMMS_[sortGroup]);
		  //par::HyperQuickSort_kway(binnedData, BIN_COMMS_[sortGroup]);  // working for SC13
		  par::sampleSort(binnedData,BIN_COMMS_[sortGroup]);
		  gt.EndTimer("Final Sort");
	      
		  if(binRanks_[sortGroup] == 0)
		    grvy_printf(INFO,"[sortio][FINALSORT] Group %i finished sort\n",sortGroup);

		  //binnedData.clear();

		  if(isBinTask_[sortGroup])
		    printResults(BIN_COMMS_[sortGroup]);

		  // do final write

		  if(binRanks_[sortGroup] == 0)
		    grvy_printf(INFO,"[sortio][FINALSORT] Group %i starting final write\n",sortGroup);
	      
		  gt.BeginTimer("Final Write");	  
		  sprintf(tmpFilename,"%s/part_bin%.3i_p%.5i",outputDir_.c_str(),ibin,sortRank_);
		  grvy_check_file_path(tmpFilename);
		  
		  FILE *fp = fopen(tmpFilename,"wb");
		  assert(fp != NULL);
		  
		  fwrite(&binnedData[0],sizeof(sortRecord),binnedData.size(),fp);
		  fclose(fp);
		  gt.EndTimer("Final Write");	  
		  
		  fflush(NULL);

		}

	      if(binRanks_[sortGroup] == 0)
		grvy_printf(INFO,"[sortio][SORT] Elapsed total time (bin = %i) = %e\n",ibin,gt.ElapsedGlobal());

	    } // end loop over numSortBins_

	  // verify we re-read in all the data

	  int globalRead = 0;

	  assert (MPI_Allreduce(&numRecordsReadFromTmp,&globalRead,1,
				MPI_INT,MPI_SUM,SORT_COMM) == MPI_SUCCESS);
	  

	  MPI_Barrier(SORT_COMM);
	  assert(globalRead == numFilesTotal_*numRecordsPerXfer);

	} 
    } 

  // now, wasn't that easy? big data shoplifters of the world
  // unite. send notification to companion IPC tasks that we are all
  // done

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

// --------------------------------------------------------------------
// Send notification message to proceses in next Bin COMM indicating it
// is their turn to do some work
// --------------------------------------------------------------------

void sortio_Class::cycleBinGroup(int numFilesTotal,int currentGroup)
{
  int destRank = sortRank_ + 1;
  static int localIter = 0;

  if(currentGroup == numSortGroups_ - 1)
    destRank = sortRank_ - (numSortGroups_ - 1);

  // circular ring for group participation

  int nextGroup = currentGroup++;

  if(nextGroup >= numSortGroups_)
    nextGroup = 0;

  const int tag=20;

  grvy_printf(DEBUG,"[sortio][Bin/Cycle] Rank %i (group %i) is activating rank %i (%i)\n",
	      sortRank_,binNum_,destRank,localIter );

  assert (MPI_Send(&numFilesTotal,1,MPI_INT,destRank,tag+activeBin_,SORT_COMM) == MPI_SUCCESS);

  localIter++;
		   
  return;
}

// --------------------------------------------------------------------
// Receive notification message from previously active Bin COMM
// indicating it is our turn to do some work
// --------------------------------------------------------------------

int sortio_Class::waitForActivation()
{
  int numFilesTotal;
  const int tag=20;
  MPI_Status status;

  // circular ring

  int recvRank = sortRank_ - 1;

  if(binNum_ == 0)
    recvRank = sortRank_ + (numSortGroups_ - 1);

  grvy_printf(DEBUG,"[sortio][Bin/Wait] Rank %i (group %i) is waiting to go active from %i\n",
	      sortRank_,binNum_,recvRank);

  assert (MPI_Recv(&numFilesTotal,1,MPI_INT,recvRank,tag+activeBin_,SORT_COMM,&status) == MPI_SUCCESS);

  grvy_printf(DEBUG,"[sortio][Bin/Wait] Rank %i (group %i) is active (numFiles = %i)\n",
	      sortRank_,binNum_,numFilesTotal);
  

  return(numFilesTotal);
}
