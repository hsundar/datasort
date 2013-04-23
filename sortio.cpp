//
// I/O class to aid in a large, distributed sort
//

#include "sortio.h"

sortio_Class::sortio_Class() 
{
  initialized_              = false;
  master                    = false;
  isMasterIO_               = false;
  isMasterXFER_             = false;
  isLocalSortMaster_        = false;
  isMasterSort_             = false;
  overrideNumFiles_         = false;
  overrideNumIOHosts_       = false;
  overrideNumSortThreads_   = false;
  random_read_offset_       = false;
  mpi_initialized_by_sortio = false;
  isIOTask_                 = false;
  isXFERTask_               = false;
  isSortTask_               = false;
  isReadFinished_           = false;
  isFirstRead_              = true;
  recordsPerFile_           = 0;
  numIoTasks_               = 0;
  numXferTasks_             = 0;
  numSortTasks_             = 0;
  numRecordsRead_           = 0;
  verifyMode_               = 0;
  sortMode_                 = 0;
  activeBin_                = 0;
  binNum_                   = -1;
  localSortRank_            = -1;
  localXferRank_            = -1;
  fileBaseName_             = "part";

  setvbuf( stdout, NULL, _IONBF, 0 );
}

sortio_Class::~sortio_Class() 
{
  if(mpi_initialized_by_sortio)
    MPI_Finalize();
}

void sortio_Class::overrideNumFiles(int nfiles)
{
  overrideNumFiles_ = true;
  numFilesTotal_   = nfiles;
  return;
}

void sortio_Class::overrideNumIOHosts(int hosts)
{
  overrideNumIOHosts_ = true;
  numIoHosts_        = hosts;
  return;
}

void sortio_Class::overrideNumSortThreads(int numThreads)
{
  overrideNumSortThreads_ = true;
  numSortThreads_         = numThreads;
  return;
}

void sortio_Class::Summarize()
{

  MPI_Barrier(GLOB_COMM);
  fflush(NULL);

  if(!isSortTask_)
    gt.Finalize();

  if(master)
    printf("\n[sortio] --- Local Read Performance----------- \n");

  if(master)
    gt.Summarize();

  fflush(NULL);
  MPI_Barrier(GLOB_COMM);

  // local performance

  double time_local, read_rate;

  if(isIOTask_)
    {
      time_local = gt.ElapsedSeconds("Raw Read");
      read_rate  = 1.0*numRecordsRead_*REC_SIZE/(1000*1000*1000*time_local);

      assert(time_local > 0.0);
      assert(numRecordsRead_ > 0);
    }

  fflush(NULL);
  MPI_Barrier(GLOB_COMM);

  // global performance

  unsigned long num_records_global;
  double time_best;
  double time_worst;
  double time_avg;
  double aggregate_rate;

  if(isIOTask_)
    {
      MPI_Allreduce(&numRecordsRead_,&num_records_global,1,MPI_UNSIGNED_LONG,MPI_SUM,IO_COMM);
      MPI_Allreduce(&time_local,&time_worst,    1,MPI_DOUBLE,MPI_MAX,IO_COMM);
      MPI_Allreduce(&time_local,&time_best,     1,MPI_DOUBLE,MPI_MIN,IO_COMM);
      MPI_Allreduce(&time_local,&time_avg,      1,MPI_DOUBLE,MPI_SUM,IO_COMM);
      MPI_Allreduce(&read_rate, &aggregate_rate,1,MPI_DOUBLE,MPI_SUM,IO_COMM);
    }

  double time_to_recv_data;

  if(isXFERTask_ && (xferRank_ == numIoTasks_))	// <-- defined as first recv task 
    {
      //printf("querying XFER/Recv on global rank %i\n",numLocal_);
      time_to_recv_data = gt.ElapsedSeconds("XFER/Recv");
    }

  fflush(NULL);

  if(master)
    {
      time_avg /= numIoTasks_;
      double total_gbs = 1.0*num_records_global*REC_SIZE/(1.0*1000*1000*1000);

      printf("\n");
      printf("[sortio] --- Global Read Performance ----------- \n");
      printf("[sortio] --> Total records read = %li\n",num_records_global);
      if(total_gbs < 1000)
	printf("[sortio] --> Total amount of data read  = %7.3f (GBs)\n",total_gbs);
      else
	printf("[sortio] --> Total amount of data read  = %7.3f (TBs)\n",total_gbs/1000.0);

      printf("\n");
      printf("[sortio] --> Global    read performance = %7.3f (GB/sec)\n",total_gbs/time_worst);
      printf("[sortio] --> Average   read performance = %7.3f (GB/sec)\n",total_gbs/time_avg);
      printf("[sortio] --> Aggregate read performance = %7.3f (GB/sec)\n",aggregate_rate);

      printf("\n[sortio] --- Receiving XFER Performance----------- \n");
    } 

  MPI_Barrier(GLOB_COMM);
  if(isXFERTask_ && (xferRank_ == numIoTasks_))	// <-- defined as first recv task 
    {
      double total_recv_gbs = 1.0*dataTransferred_/(1000*1000*1000);

      if(total_recv_gbs < 1000)
	printf("[sortio] --> Total data received        = %7.3f (GBs)\n",total_recv_gbs);
      else
	printf("[sortio] --> Total data received        = %7.3f (TBs)\n",total_recv_gbs/1000.0);

      printf("[sortio] --> Transfer performance       = %7.3f (GB/sec)\n",total_recv_gbs/time_to_recv_data);
    }
  MPI_Barrier(GLOB_COMM);

  return;
}

void sortio_Class::Initialize(std::string ifile, MPI_Comm COMM)
{
  assert(!initialized_);

  gt.Init("Sort IO Subsystem");
  gt.BeginTimer("Initialize");

  // Init MPI if necessary

  int is_mpi_initialized;
  MPI_Initialized(&is_mpi_initialized);

  if(!is_mpi_initialized)
    {
      int provided;

      MPI_Init_thread(NULL,NULL,MPI_THREAD_FUNNELED,&provided);
      //MPI_Init_thread(NULL,NULL,MPI_THREAD_MULTIPLE,&provided);
      mpi_initialized_by_sortio = true;
    }

  // Query global MPI environment

  MPI_Comm_size(COMM,&numTasks_);
  MPI_Comm_rank(COMM,&numLocal_);

  GLOB_COMM = COMM;

  // Read I/O runtime controls

  if(numLocal_ == 0)
    {
      master = true;
      GRVY::GRVY_Input_Class iparse;

      assert( iparse.Open    ("input.dat")                         != 0);
      if(!overrideNumFiles_)
	assert( iparse.Read_Var("sortio/num_files",&numFilesTotal_) != 0);

      // Register defaults

      iparse.Register_Var("sortio/max_read_buffers",       10);
      iparse.Register_Var("sortio/max_file_size_in_mbs",  100);
      iparse.Register_Var("sortio/max_messages_watermark", 10);
      iparse.Register_Var("sortio/verify_mode",             0);
      iparse.Register_Var("sortio/sort_mode",               1);
      iparse.Register_Var("sortio/num_sort_groups",         1);
      if(!overrideNumSortThreads_)
	iparse.Register_Var("sortio/num_sort_threads",      1);

      if(!overrideNumIOHosts_)
	assert( iparse.Read_Var("sortio/num_io_hosts",        &numIoHosts_)           != 0 );

      assert( iparse.Read_Var("sortio/input_dir",             &inputDir_)              != 0 );
      assert( iparse.Read_Var("sortio/output_dir",            &outputDir_)             != 0 );
      assert( iparse.Read_Var("sortio/verify_mode",           &verifyMode_)            != 0 );
      assert( iparse.Read_Var("sortio/sort_mode",             &sortMode_)              != 0 );
      assert( iparse.Read_Var("sortio/num_sort_groups",       &numSortGroups_        ) != 0 );
      assert( iparse.Read_Var("sortio/num_sort_threads",      &numSortThreads_       ) != 0 );
      assert( iparse.Read_Var("sortio/max_read_buffers",      &MAX_READ_BUFFERS)       != 0 );
      assert( iparse.Read_Var("sortio/max_file_size_in_mbs"  ,&MAX_FILE_SIZE_IN_MBS)   != 0 );
      assert( iparse.Read_Var("sortio/max_messages_watermark",&MAX_MESSAGES_WATERMARK) != 0 );

      // Simple sanity checks

      assert( numIoHosts_          > 0);
      assert( MAX_READ_BUFFERS     > 0);
      assert( MAX_FILE_SIZE_IN_MBS > 0);
      assert( (numSortGroups_  >= 2) && (numSortGroups_  < 16) );   // Assume 16-way hosts or less (need 2 minimum)
      assert( (numSortThreads_ >  0) && (numSortThreads_ < 16) ); 
      assert( MAX_FILE_SIZE_IN_MBS*MAX_READ_BUFFERS <= 20*1024 ); // Assume less than 20 GB/host
      assert( MAX_MESSAGES_WATERMARK < 100);

      grvy_printf(INFO,"[sortio]\n");
      grvy_printf(INFO,"[sortio] Runtime input parsing:\n");
      grvy_printf(INFO,"[sortio] --> Total number of files to read = %i\n",numFilesTotal_);
      grvy_printf(INFO,"[sortio] --> Input directory               = %s\n",inputDir_.c_str());
      grvy_printf(INFO,"[sortio] --> Output directory              = %s\n",outputDir_.c_str());
      grvy_printf(INFO,"[sortio] --> Number of read buffers        = %i\n",MAX_READ_BUFFERS);
      grvy_printf(INFO,"[sortio] --> Size of each read buffer      = %i MBs\n",MAX_FILE_SIZE_IN_MBS);
      grvy_printf(INFO,"[sortio] --> Number of sort groups         = %i\n",numSortGroups_);
      grvy_printf(INFO,"[sortio] --> Number of sort threads        = %i\n",numSortThreads_);

    }

  // Broadcast necessary runtime controls to all tasks

  int tmp_string_size  = inputDir_.size() + 1;
  int tmp_string_size2 = outputDir_.size() + 1;
  char *tmp_string     = NULL;
  char *tmp_string2    = NULL;

  //  random_read_offset_  = true;

  assert( MPI_Bcast(&numFilesTotal_,        1,MPI_INT,0,COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&numIoHosts_,           1,MPI_INT,0,COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&numSortGroups_,        1,MPI_INT,0,COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&numSortThreads_,       1,MPI_INT,0,COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&verifyMode_,           1,MPI_INT,0,COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&sortMode_,             1,MPI_INT,0,COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&MAX_READ_BUFFERS,      1,MPI_INT,0,COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&MAX_FILE_SIZE_IN_MBS,  1,MPI_INT,0,COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&MAX_MESSAGES_WATERMARK,1,MPI_INT,0,COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&tmp_string_size,       1,MPI_INT,0,COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&tmp_string_size2,      1,MPI_INT,0,COMM) == MPI_SUCCESS );
  
  tmp_string = (char *)calloc(tmp_string_size,sizeof(char));
  strcpy(tmp_string,inputDir_.c_str());

  tmp_string2 = (char *)calloc(tmp_string_size2,sizeof(char));
  strcpy(tmp_string2,outputDir_.c_str());

  assert (MPI_Bcast(tmp_string, tmp_string_size, MPI_CHAR,0,COMM) == MPI_SUCCESS);
  assert (MPI_Bcast(tmp_string2,tmp_string_size2,MPI_CHAR,0,COMM) == MPI_SUCCESS);

  if(!master)
    {
      inputDir_  = tmp_string;
      outputDir_ = tmp_string2;
    }

  free(tmp_string);
  free(tmp_string2);

  // initialize RNG

  srand(numLocal_);

  MPI_Barrier(COMM);

  initialized_ = true;
  gt.EndTimer("Initialize");

  return; 
}

// --------------------------------------------------------------------
// SplitComm(): take input communicator and split into three types of
// groups based on runtime settings:
// 
// (1) Read Data Group     (  IO_COMM  )
// (2) Data Transfer Group ( XFER_COMM )
// (3) Data Sort Group     ( SORT_COMM )
// --------------------------------------------------------------------

void sortio_Class::SplitComm()
{
  assert(initialized_ == true );

  char hostname[MPI_MAX_PROCESSOR_NAME];
  int len;

  MPI_Get_processor_name(hostname, &len);

  grvy_printf(DEBUG,"[sortio] Detected global Rank %i -> %s\n",numLocal_,hostname);

  char *hostnames_ALL;

  if(master)
    {
      hostnames_ALL = (char *)malloc(numTasks_*MPI_MAX_PROCESSOR_NAME*sizeof(char));
      assert(hostnames_ALL != NULL);
    }

  assert (MPI_Gather(&hostname[0],     MPI_MAX_PROCESSOR_NAME,MPI_CHAR,
		     &hostnames_ALL[0],MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,GLOB_COMM) == MPI_SUCCESS);

  std::vector<int>      io_comm_ranks; 
  std::vector<int>    xfer_comm_ranks;

  std::vector<int>    sort_comm_ranks;
  std::vector<int>    first_sort_rank;

  std::vector< std::vector<int> > binCommRanks;

  std::map<std::string,std::vector<int> > uniq_hosts; // hostname -> global MPI rank mapping



  if(master)
    {
      // Determine unique hostnames -> rank mapping

      for(int i=0;i<numTasks_;i++)
	{
	  std::string host = &hostnames_ALL[i*MPI_MAX_PROCESSOR_NAME];
 	  grvy_printf(DEBUG,"[sortio] parsed host = %s\n",host.c_str());
	  uniq_hosts[host].push_back(i);
	}

      std::map<std::string,std::vector<int> >::iterator it;

      int count = 0;

      // Flag tasks for different work groups

      assert (numIoHosts_ > 0);
      assert (numIoHosts_ < numTasks_);

      grvy_printf(INFO,"[sortio]\n");
      grvy_printf(INFO,"[sortio] Rank per host detection:\n");

      for(int i=0;i<numSortGroups_;i++)
	{
	  std::vector <int> iVec;
	  binCommRanks.push_back(iVec);
	}


      for(it = uniq_hosts.begin(); it != uniq_hosts.end(); it++ ) 
	{
	  // Logic: we use the first num_io_hosts for IO (and we
	  // currently assume 1 MPI task per IO host). We use all
	  // remaining hosts for SORT with 1 MPI task per sort host
	  // dedicated for data XFER and all remaining tasks used for
	  // sorting.

	  if(count < numIoHosts_)
	    {
	      io_comm_ranks.push_back  ((*it).second[0]);
	      xfer_comm_ranks.push_back((*it).second[0]);
	    }
	  else
	    {
	      xfer_comm_ranks.push_back((*it).second[0]);

	      for(size_t proc=1;proc<(*it).second.size();proc++)
		{
		  if(proc == 1)
		    first_sort_rank.push_back((*it).second[proc]);
		  sort_comm_ranks.push_back((*it).second[proc]);
		}

	      // Also build up multiple binning communicators


	      //binCommRanks.push_back(std::vector<int> iVec);

	      assert( (*it).second.size() >= (numSortGroups_ + 1) );

	      for(int i=0;i<numSortGroups_;i++)
		binCommRanks[i].push_back( (*it).second[i+1] );

	    }
	    
	  grvy_printf(INFO,"[sortio]    %s -> %3i MPI task(s)/host\n",(*it).first.c_str(),(*it).second.size());
	  count++;
	}


      numIoTasks_     = io_comm_ranks.size();
      numXferTasks_   = xfer_comm_ranks.size();
      numSortTasks_   = sort_comm_ranks.size();
      numSortHosts_   = first_sort_rank.size();

      // Final sort does stalls with non-powers of 2, verify now.

      if(sortMode_ > 1)
	assert(isPowerOfTwo(numSortHosts_));

      // quick sanity checks and assumptions

      assert(numIoTasks_   > 0);
      assert(numXferTasks_ > 0);
      assert(numXferTasks_ > numIoTasks_);
      assert(numSortTasks_ > 0);

      assert(numIoTasks_   == numIoHosts_);
      assert(numSortHosts_ == ( (int)uniq_hosts.size() - numIoHosts_));

      assert(numXferTasks_ + numSortTasks_ <= numTasks_);

      // we assume all hosts have the same number of MPI tasks specified

      int numTasksPerHost = (*uniq_hosts.begin()).second.size();
      for(it = uniq_hosts.begin(); it != uniq_hosts.end(); it++ ) 
	assert( (int)(*it).second.size() == numTasksPerHost);

      //      numSortTasksPerHost_ = numTasksPerHost - 1; // 1 task belongs to XFER_COMM

      // Create desired MPI sub communicators based on runtime settings

      grvy_printf(INFO,"[sortio]\n");
      grvy_printf(INFO,"[sortio] Total number of hosts available    = %4i\n",uniq_hosts.size());
      grvy_printf(INFO,"[sortio] --> Number of   IO hosts           = %4i\n",numIoHosts_);
      grvy_printf(INFO,"[sortio] --> Number of SORT hosts           = %4i\n",numSortHosts_);
      grvy_printf(INFO,"[sortio]\n");
      grvy_printf(INFO,"[sortio] Work Task Division:\n");
      grvy_printf(INFO,"[sortio] --> Number of IO   MPI tasks       = %4i\n",numIoTasks_);
      grvy_printf(INFO,"[sortio] --> Number of XFER MPI tasks       = %4i\n",numXferTasks_);
      grvy_printf(INFO,"[sortio] --> Number of SORT MPI tasks       = %4i\n",numSortTasks_);
      grvy_printf(INFO,"[sortio] --> Number of BIN  groups          = %4i\n",numSortGroups_);

    }

  // Build up new MPI task groups

  assert( MPI_Bcast(&numIoTasks_,          1,MPI_INT,0,GLOB_COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&numXferTasks_,        1,MPI_INT,0,GLOB_COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&numSortTasks_,        1,MPI_INT,0,GLOB_COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&numSortHosts_,        1,MPI_INT,0,GLOB_COMM) == MPI_SUCCESS );

  assert(numSortHosts_ > 0);

  if(!master)
    {
      io_comm_ranks.reserve  (numIoTasks_   );
      xfer_comm_ranks.reserve(numXferTasks_ );
      sort_comm_ranks.reserve(numSortTasks_ );
      first_sort_rank.reserve(numSortHosts_ );

      for(int i=0;i<numSortGroups_;i++)
	{
	  std::vector<int> iVec(numSortHosts_);
	  binCommRanks.push_back(iVec);
	}

    }

  assert( MPI_Bcast(  io_comm_ranks.data(),numIoTasks_,  MPI_INT,0,GLOB_COMM) == MPI_SUCCESS);
  assert( MPI_Bcast(xfer_comm_ranks.data(),numXferTasks_,MPI_INT,0,GLOB_COMM) == MPI_SUCCESS);
  assert( MPI_Bcast(sort_comm_ranks.data(),numSortTasks_,MPI_INT,0,GLOB_COMM) == MPI_SUCCESS);
  assert( MPI_Bcast(first_sort_rank.data(),numSortHosts_,MPI_INT,0,GLOB_COMM) == MPI_SUCCESS);

  assert(binCommRanks.size() == numSortGroups_);

  for(int i=0;i<numSortGroups_;i++)
    assert(binCommRanks[i].size() == numSortHosts_);

  for(int i=0;i<numSortGroups_;i++)
    assert( MPI_Bcast(binCommRanks[i].data(),numSortHosts_,MPI_INT,0,GLOB_COMM) == MPI_SUCCESS);

  MPI_Group group_global;
  MPI_Group group_io;
  MPI_Group group_xfer;
  MPI_Group group_sort;
  std::vector<MPI_Group> groups_binning(numSortGroups_);

  assert( MPI_Comm_group(GLOB_COMM,&group_global)  == MPI_SUCCESS );

  // New groups/communicators for IO, XFER, SORT, and BIN(s)

  assert( MPI_Group_incl(group_global,   numIoTasks_,   io_comm_ranks.data(),   &group_io) == MPI_SUCCESS);
  assert( MPI_Group_incl(group_global, numXferTasks_, xfer_comm_ranks.data(), &group_xfer) == MPI_SUCCESS);
  assert( MPI_Group_incl(group_global, numSortTasks_, sort_comm_ranks.data(), &group_sort) == MPI_SUCCESS);

  for(int i=0;i<numSortGroups_;i++)
    assert( MPI_Group_incl(group_global, numSortHosts_, binCommRanks[i].data(), &groups_binning[i]) == MPI_SUCCESS);


  assert( MPI_Comm_create(GLOB_COMM,   group_io,   &IO_COMM) == MPI_SUCCESS);
  assert( MPI_Comm_create(GLOB_COMM, group_xfer, &XFER_COMM) == MPI_SUCCESS);
  assert( MPI_Comm_create(GLOB_COMM, group_sort, &SORT_COMM) == MPI_SUCCESS);

  BIN_COMMS_.reserve(numSortGroups_);

  for(int i=0;i<numSortGroups_;i++)
    assert( MPI_Comm_create(GLOB_COMM, groups_binning[i], &BIN_COMMS_[i]) == MPI_SUCCESS);

  // is the local rank part of the new groups?

  int rank_tmp;

  isBinTask_.resize(numSortGroups_,false);

  assert( MPI_Group_rank( group_io, &rank_tmp) == MPI_SUCCESS);
  if(rank_tmp != MPI_UNDEFINED)
    isIOTask_ = true;

  assert( MPI_Group_rank( group_xfer, &rank_tmp) == MPI_SUCCESS);
  if(rank_tmp != MPI_UNDEFINED)
    isXFERTask_ = true;

  assert( MPI_Group_rank( group_sort, &rank_tmp) == MPI_SUCCESS);
  if(rank_tmp != MPI_UNDEFINED)
      isSortTask_ = true;

  for(int i=0;i<numSortGroups_;i++)
    {
      assert( MPI_Group_rank( groups_binning[i], &rank_tmp) == MPI_SUCCESS);
      if(rank_tmp != MPI_UNDEFINED)
	{
	  isBinTask_[i] = true;
	  binNum_       = i;
	}
    }

  MPI_Barrier(GLOB_COMM);

  //  cache the new communicator ranks...

  if(isIOTask_)
    {
      assert( MPI_Comm_rank(IO_COMM,&ioRank_) == MPI_SUCCESS);
      if(ioRank_ == 0)
	isMasterIO_ = true;
    }
  if(isXFERTask_)
    {
      assert( MPI_Comm_rank(XFER_COMM,&xferRank_) == MPI_SUCCESS);
      if(xferRank_ == 0)
	{
	  isMasterXFER_ = true;
	  masterXFER_GlobalRank = numLocal_;
	}

      // On receiving XFER tasks, identify global rank of first sort
      // task that also resides on this host (for IPC communication)

      if(xferRank_ >= numIoHosts_)
	{
	  int index = xferRank_-numIoHosts_;   
	  localSortRank_ = first_sort_rank[index];
	  grvy_printf(DEBUG,"[sortio] --> IPC setup: XFER rank %i -> global master sort rank %i\n",
		      xferRank_,localSortRank_);
	}
      
    }

  if(isSortTask_)
    {
      assert( MPI_Comm_rank(SORT_COMM,&sortRank_) == MPI_SUCCESS);
      if(sortRank_ == 0)
	isMasterSort_ = true;

      // For the master Sort task on each host, identify global rank
      // of XFER task that also resides on this host (for IPC
      // communication)

      assert( (numIoHosts_ + numSortHosts_) <= numXferTasks_);

      for(int i=0;i<numSortHosts_;i++)
	{
	  if(numLocal_ == first_sort_rank[i])
	    {
	      isLocalSortMaster_ = true;
	      localXferRank_     = xfer_comm_ranks[i+numIoHosts_];
	      grvy_printf(DEBUG,"[sortio] --> IPC setup: SORT rank %i -> global xfer rank %i\n",
			  sortRank_,localXferRank_);
	    }
	}

      // cache local binning ranks

      binRanks_.resize(numSortGroups_,-1);

      for(int i=0;i<numSortGroups_;i++)
	if(isBinTask_[i])
	  assert( MPI_Comm_rank(BIN_COMMS_[i],&binRanks_[i]) == MPI_SUCCESS);
    }

  MPI_Barrier(GLOB_COMM);

  // summarize the config (data printed from master rank to make the output easy on 
  // the eyes for the time being)

  if(master)
    {
      grvy_printf(INFO,"[sortio]\n");
      grvy_printf(INFO,"[sortio] MPI WorkGroup Summary (%i hosts, %i MPI tasks)\n",uniq_hosts.size(),numTasks_);

      grvy_printf(INFO,"[sortio] --> Communicator Ranking Demarcation\n");
      grvy_printf(INFO,"[sortio]\n");
      grvy_printf(INFO,"[sortio] ----------------------------------------------------");
      for(int i=2;i<=numSortGroups_;i++)
	grvy_printf(INFO,"--------");
      grvy_printf(INFO,"\n");

      grvy_printf(INFO,"[sortio] [Hostname]  [Global]  [ IO ]  [XFER]  [SORT]  [BIN1]");
      for(int i=2;i<=numSortGroups_;i++)
	grvy_printf(INFO,"  [BIN%i]",i);
      grvy_printf(INFO,"\n");

      grvy_printf(INFO,"[sortio] ----------------------------------------------------");
      for(int i=2;i<=numSortGroups_;i++)
	grvy_printf(INFO,"--------");
      grvy_printf(INFO,"\n");
    }


  const int numColumns = 3 + numSortGroups_;
  int *ranks_tmp;

  ranks_tmp = new int[numColumns];
  MPI_Status status;

  for(int proc=0;proc<numTasks_;proc++)
    {
      if(master)
	{
	  grvy_printf(INFO,"[sortio]  %.8s    %.6i ",&hostnames_ALL[proc*MPI_MAX_PROCESSOR_NAME],proc);
	}

      if(master && (proc == 0) )
	{
	  if(isIOTask_)
	    printf("  %.6i",ioRank_);
	  else
	    printf("  ------");

	  if(isXFERTask_)
	    printf("  %.6i",xferRank_);
	  else
	    printf("  ------");

	  if(isSortTask_)
	    printf("  %.6i\n",sortRank_);
	  else
	    printf("  ------\n");

	  continue;
	}

      if(numLocal_ == proc)
	{
	  ranks_tmp[0] = (  isIOTask_ ) ?   ioRank_ : -1 ;
	  ranks_tmp[1] = (isXFERTask_ ) ? xferRank_ : -1 ;
	  ranks_tmp[2] = (isSortTask_ ) ? sortRank_ : -1 ;

	  for(int i=0;i<numSortGroups_;i++)
	    ranks_tmp[i+3] = (isBinTask_[i] ) ? binRanks_[i] : -1 ;

	  assert (MPI_Send(ranks_tmp,numColumns,MPI_INT,0,100,GLOB_COMM) == MPI_SUCCESS) ;
	}
      
      if(master)
	{
	  assert (MPI_Recv(ranks_tmp,numColumns,MPI_INT,proc,100,GLOB_COMM,&status) == MPI_SUCCESS) ;

	  for(int i=0;i<3;i++)
	    {
	      if(ranks_tmp[i] >= 0)
		printf("  %.6i",ranks_tmp[i]);
	      else
		printf("  ------");
	    }

	  // now show binning groups

	  for(int i=0;i<numSortGroups_;i++)
	    {
	      if(ranks_tmp[i+3] >= 0)
		printf("  %.6i",ranks_tmp[i+3]);
	      else
		printf("  ------");
	    }
	  grvy_printf(INFO,"\n");
	  fflush(NULL);
	}
    }

  delete [] ranks_tmp;


  if(master)
    {
      for(int i=0;i<2;i++)
	{
	  grvy_printf(INFO,"[sortio] ----------------------------------------------------");
	  for(int i=2;i<=numSortGroups_;i++)
	    grvy_printf(INFO,"--------");
	  grvy_printf(INFO,"\n");
	}
    }

  MPI_Barrier(GLOB_COMM);

  // clean-up

  assert (MPI_Group_free(&group_io  ) == MPI_SUCCESS );
  assert (MPI_Group_free(&group_xfer) == MPI_SUCCESS );
  assert (MPI_Group_free(&group_sort) == MPI_SUCCESS );
  for(int i=0;i<numSortGroups_;i++)
    assert (MPI_Group_free(&groups_binning[i]) == MPI_SUCCESS );

  if(master)
    free(hostnames_ALL);  
  return;
}

// --------------------------------------------------------------------
// Reenable input buffer for use by reader tasks by adding to the
// Empty queue 
//
// Thread-safety: all Empty/Full queue updates are locked
// --------------------------------------------------------------------

void sortio_Class::addBuffertoEmptyQueue(int bufNum)
{
#pragma omp critical (IO_XFER_UPDATES_lock) // Thread-safety: all queue updates are locked
  {
    emptyQueue_.push_back(bufNum);
    grvy_printf(DEBUG,"[sortio][IO/XFER][%.4i] added %i buff back to emptyQueue\n",ioRank_,bufNum);
  }
  return;
}

int sortio_Class::isPowerOfTwo(unsigned int x)
{
  return ((x != 0) && !(x & (x - 1)));
}
