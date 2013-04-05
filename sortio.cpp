//
// I/O class to aid in a large, distributed sort
//

#include "sortio.h"

sortio_Class::sortio_Class() 
{
  initialized               = false;
  master                    = false;
  master_io                 = false;
  isMasterXFER_             = false;
  isLocalSortMaster_        = false;
  master_sort               = false;
  overrideNumFiles_         = false;
  overrideNumIOHosts_       = false;
  random_read_offset        = false;
  mpi_initialized_by_sortio = false;
  is_io_task                = false;
  is_xfer_task              = false;
  is_sort_task              = false;
  isReadFinished_           = false;
  isFirstRead_              = true;
  records_per_file_         = 0;
  nio_tasks                 = 0;
  nxfer_tasks               = 0;
  nsort_tasks               = 0;
  num_records_read          = 0;
  localSortRank_            = -1;
  localXferRank_            = -1;
  basename                  = "part";

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
  num_files_total   = nfiles;
  return;
}

void sortio_Class::overrideNumIOHosts(int hosts)
{
  overrideNumIOHosts_ = true;
  num_io_hosts        = hosts;
  return;
}

void sortio_Class::Summarize()
{

  MPI_Barrier(GLOB_COMM);
  fflush(NULL);

  gt.Finalize();

  if(master)
    printf("\n[sortio] --- Local Read Performance----------- \n");

  if(master)
    gt.Summarize();

  fflush(NULL);
  MPI_Barrier(GLOB_COMM);

  // local performance

  double time_local, read_rate;

  if(is_io_task)
    {
      time_local = gt.ElapsedSeconds("Raw Read");
      read_rate  = 1.0*num_records_read*REC_SIZE/(1000*1000*1000*time_local);

      assert(time_local > 0.0);
      assert(num_records_read > 0);
    }

  fflush(NULL);
  MPI_Barrier(GLOB_COMM);

  // global performance

  unsigned long num_records_global;
  double time_best;
  double time_worst;
  double time_avg;
  double aggregate_rate;

  if(is_io_task)
    {
      MPI_Allreduce(&num_records_read,&num_records_global,1,MPI_UNSIGNED_LONG,MPI_SUM,IO_COMM);
      MPI_Allreduce(&time_local,&time_worst,    1,MPI_DOUBLE,MPI_MAX,IO_COMM);
      MPI_Allreduce(&time_local,&time_best,     1,MPI_DOUBLE,MPI_MIN,IO_COMM);
      MPI_Allreduce(&time_local,&time_avg,      1,MPI_DOUBLE,MPI_SUM,IO_COMM);
      MPI_Allreduce(&read_rate, &aggregate_rate,1,MPI_DOUBLE,MPI_SUM,IO_COMM);
    }

  double time_to_recv_data;

  if(is_xfer_task && (xfer_rank == nio_tasks))	// <-- defined as first recv task 
    {
      //printf("querying XFER/Recv on global rank %i\n",num_local);
      time_to_recv_data = gt.ElapsedSeconds("XFER/Recv");
    }

  fflush(NULL);

  if(master)
    {
      time_avg /= nio_tasks;
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
  if(is_xfer_task && (xfer_rank == nio_tasks))	// <-- defined as first recv task 
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
  assert(!initialized);

  gt.Init("Sort IO Subsystem");
  gt.BeginTimer("Initialize");

  // Init MPI if necessary

  int is_mpi_initialized;
  MPI_Initialized(&is_mpi_initialized);

  if(!is_mpi_initialized)
    {
      int provided;

      MPI_Init_thread(NULL,NULL,MPI_THREAD_FUNNELED,&provided);
      mpi_initialized_by_sortio = true;
    }

  // Query global MPI environment

  MPI_Comm_size(COMM,&num_tasks);
  MPI_Comm_rank(COMM,&num_local);

  GLOB_COMM = COMM;

  // Read I/O runtime controls

  if(num_local == 0)
    {
      master = true;
      GRVY::GRVY_Input_Class iparse;

      assert( iparse.Open    ("input.dat")                         != 0);
      if(!overrideNumFiles_)
	assert( iparse.Read_Var("sortio/num_files",&num_files_total) != 0);

      // Register defaults

      iparse.Register_Var("sortio/max_read_buffers",     10);
      iparse.Register_Var("sortio/max_file_size_in_mbs",100);

      if(!overrideNumIOHosts_)
	assert( iparse.Read_Var("sortio/num_io_hosts",        &num_io_hosts)         != 0 );
      assert( iparse.Read_Var("sortio/input_dir",           &indir)                != 0 );
      assert( iparse.Read_Var("sortio/max_read_buffers",    &MAX_READ_BUFFERS)     != 0 );
      assert( iparse.Read_Var("sortio/max_file_size_in_mbs",&MAX_FILE_SIZE_IN_MBS) != 0 );

      // Simple sanity checks

      assert( num_io_hosts         > 0);
      assert( MAX_READ_BUFFERS     > 0);
      assert( MAX_FILE_SIZE_IN_MBS > 0);
      assert( MAX_FILE_SIZE_IN_MBS*MAX_READ_BUFFERS <= 16*1024 ); // Assume less than 16 GB/host

      grvy_printf(INFO,"[sortio]\n");
      grvy_printf(INFO,"[sortio] Runtime input parsing:\n");
      grvy_printf(INFO,"[sortio] --> Total number of files to read = %i\n",num_files_total);
      grvy_printf(INFO,"[sortio] --> Input directory               = %s\n",indir.c_str());
      grvy_printf(INFO,"[sortio] --> Number of read buffers        = %i\n",MAX_READ_BUFFERS);
      grvy_printf(INFO,"[sortio] --> Size of each read buffer      = %i MBs\n",MAX_FILE_SIZE_IN_MBS);
    }

  // Broadcast necessary runtime controls to all tasks

  int tmp_string_size = indir.size() + 1;
  char *tmp_string    = NULL;

  //  random_read_offset  = true;

  assert( MPI_Bcast(&num_files_total,     1,MPI_INTEGER,0,COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&num_io_hosts,        1,MPI_INTEGER,0,COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&MAX_READ_BUFFERS,    1,MPI_INTEGER,0,COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&MAX_FILE_SIZE_IN_MBS,1,MPI_INTEGER,0,COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&tmp_string_size,     1,MPI_INTEGER,0,COMM) == MPI_SUCCESS );
  
  tmp_string = (char *)calloc(tmp_string_size,sizeof(char));
  strcpy(tmp_string,indir.c_str());

  assert (MPI_Bcast(tmp_string,tmp_string_size,MPI_CHAR,0,COMM) == MPI_SUCCESS);

  if(!master)
    indir = tmp_string;

  free(tmp_string);

  MPI_Barrier(COMM);

  initialized = true;
  gt.EndTimer("Initialize");

  return; 
}

// --------------------------------------------------------------------
// SplitComm(): take input communicator and split into three groups
// based on runtime settings:
// 
// (1) Read Data Group     (  IO_COMM  )
// (2) Data Transfer Group ( XFER_COMM )
// (3) Data Sort Group     ( SORT_COMM )
// --------------------------------------------------------------------

void sortio_Class::SplitComm()
{
  assert(initialized == true );

  char hostname[MPI_MAX_PROCESSOR_NAME];
  int len;

  MPI_Get_processor_name(hostname, &len);

  grvy_printf(DEBUG,"[sortio] Detected global Rank %i -> %s\n",num_local,hostname);

  char *hostnames_ALL;

  if(master)
    {
      hostnames_ALL = (char *)malloc(num_tasks*MPI_MAX_PROCESSOR_NAME*sizeof(char));
      assert(hostnames_ALL != NULL);
    }

  assert (MPI_Gather(&hostname[0],     MPI_MAX_PROCESSOR_NAME,MPI_CHAR,
		     &hostnames_ALL[0],MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,GLOB_COMM) == MPI_SUCCESS);

  std::vector<int>      io_comm_ranks; 
  std::vector<int>    xfer_comm_ranks;
  std::vector<int>    sort_comm_ranks;
  std::vector<int>    first_sort_rank;

  std::map<std::string,std::vector<int> > uniq_hosts; // hostname -> global MPI rank mapping

  if(master)
    {
      // Determine unique hostnames -> rank mapping

      for(int i=0;i<num_tasks;i++)
	{
	  std::string host = &hostnames_ALL[i*MPI_MAX_PROCESSOR_NAME];
 	  grvy_printf(DEBUG,"[sortio] parsed host = %s\n",host.c_str());
	  uniq_hosts[host].push_back(i);
	}

      std::map<std::string,std::vector<int> >::iterator it;

      int count = 0;

      // Flag tasks for different work groups

      assert (num_io_hosts > 0);
      assert (num_io_hosts < num_tasks);

      grvy_printf(INFO,"[sortio]\n");
      grvy_printf(INFO,"[sortio] Rank per host detection:\n");

      for(it = uniq_hosts.begin(); it != uniq_hosts.end(); it++ ) 
	{
	  // Logic: we use the first num_io_hosts for IO (and we
	  // currently assume 1 MPI task per IO host). We use all
	  // remaining hosts for SORT with 1 MPI task per sort host
	  // dedicated for data XFER and all remaining tasks used for
	  // sorting.

	  if(count < num_io_hosts)
	    {
	      io_comm_ranks.push_back  ((*it).second[0]);
	      xfer_comm_ranks.push_back((*it).second[0]);
	    }
	  else
	    {
	      xfer_comm_ranks.push_back((*it).second[0]);

	      for(int proc=1;proc<(*it).second.size();proc++)
		{
		  if(proc == 1)
		    first_sort_rank.push_back((*it).second[proc]);
		  sort_comm_ranks.push_back((*it).second[proc]);
		}
	    }
	    
	  grvy_printf(INFO,"[sortio]    %s -> %3i MPI task(s)/host\n",(*it).first.c_str(),(*it).second.size());
	  count++;
	}

      nio_tasks     = io_comm_ranks.size();
      nxfer_tasks   = xfer_comm_ranks.size();
      nsort_tasks   = sort_comm_ranks.size();
      numSortHosts_ = first_sort_rank.size();

      // quick sanity checks

      assert(nio_tasks   > 0);
      assert(nxfer_tasks > 0);
      assert(nxfer_tasks > nio_tasks);
      assert(nsort_tasks > 0);
      assert(nio_tasks == num_io_hosts);
      assert(nxfer_tasks + nsort_tasks <= num_tasks);
      assert(numSortHosts_ == (uniq_hosts.size()-num_io_hosts));

      // Create desired MPI sub communicators based on runtime settings

      grvy_printf(INFO,"[sortio]\n");
      grvy_printf(INFO,"[sortio] Total number of hosts available = %4i\n",uniq_hosts.size());
      grvy_printf(INFO,"[sortio] --> Number of   IO hosts        = %4i\n",num_io_hosts);
      grvy_printf(INFO,"[sortio] --> Number of SORT hosts        = %4i\n",uniq_hosts.size()-num_io_hosts);
      grvy_printf(INFO,"[sortio]\n");
      grvy_printf(INFO,"[sortio] Work Task Division:\n");
      grvy_printf(INFO,"[sortio] --> Number of IO   MPI tasks    = %4i\n",nio_tasks);
      grvy_printf(INFO,"[sortio] --> Number of XFER MPI tasks    = %4i\n",nxfer_tasks);
      grvy_printf(INFO,"[sortio] --> Number of SORT MPI tasks    = %4i\n",nsort_tasks);

    }

  // Build up new MPI task groups

  assert( MPI_Bcast(&nio_tasks,     1,MPI_INTEGER,0,GLOB_COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&nxfer_tasks,   1,MPI_INTEGER,0,GLOB_COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&nsort_tasks,   1,MPI_INTEGER,0,GLOB_COMM) == MPI_SUCCESS );
  assert( MPI_Bcast(&numSortHosts_, 1,MPI_INTEGER,0,GLOB_COMM) == MPI_SUCCESS );

  if(!master)
    {
      io_comm_ranks.reserve  (nio_tasks);
      xfer_comm_ranks.reserve(nxfer_tasks);
      sort_comm_ranks.reserve(nsort_tasks);
      first_sort_rank.reserve(numSortHosts_);
    }

  assert( MPI_Bcast(  io_comm_ranks.data(),    nio_tasks,MPI_INTEGER,0,GLOB_COMM) == MPI_SUCCESS);
  assert( MPI_Bcast(xfer_comm_ranks.data()  ,nxfer_tasks,MPI_INTEGER,0,GLOB_COMM) == MPI_SUCCESS);
  assert( MPI_Bcast(sort_comm_ranks.data(),  nsort_tasks,MPI_INTEGER,0,GLOB_COMM) == MPI_SUCCESS);
  assert( MPI_Bcast(first_sort_rank.data(),numSortHosts_,MPI_INTEGER,0,GLOB_COMM) == MPI_SUCCESS);

  MPI_Group group_global;
  MPI_Group group_io;
  MPI_Group group_xfer;
  MPI_Group group_sort;

  assert( MPI_Comm_group(GLOB_COMM,&group_global)  == MPI_SUCCESS );

  // New groups for IO, XFER, SORT, and special Scatter group (1 per IO host)

  assert( MPI_Group_incl(group_global,   nio_tasks,   io_comm_ranks.data(),   &group_io) == MPI_SUCCESS);
  assert( MPI_Group_incl(group_global, nxfer_tasks, xfer_comm_ranks.data(), &group_xfer) == MPI_SUCCESS);
  assert( MPI_Group_incl(group_global, nsort_tasks, sort_comm_ranks.data(), &group_sort) == MPI_SUCCESS);

  assert( MPI_Comm_create(GLOB_COMM,   group_io,   &IO_COMM) == MPI_SUCCESS);
  assert( MPI_Comm_create(GLOB_COMM, group_xfer, &XFER_COMM) == MPI_SUCCESS);
  assert( MPI_Comm_create(GLOB_COMM, group_sort, &SORT_COMM) == MPI_SUCCESS);

  // is the local rank part of the new groups?

  int rank_tmp;

  assert( MPI_Group_rank( group_io, &rank_tmp) == MPI_SUCCESS);
  if(rank_tmp != MPI_UNDEFINED)
    is_io_task = true;

  assert( MPI_Group_rank( group_xfer, &rank_tmp) == MPI_SUCCESS);
  if(rank_tmp != MPI_UNDEFINED)
    is_xfer_task = true;

  assert( MPI_Group_rank( group_sort, &rank_tmp) == MPI_SUCCESS);
  if(rank_tmp != MPI_UNDEFINED)
      is_sort_task = true;

  MPI_Barrier(GLOB_COMM);

  //  cache the new communicator ranks...

  if(is_io_task)
    {
      assert( MPI_Comm_rank(IO_COMM,&io_rank) == MPI_SUCCESS);
      if(io_rank == 0)
	master_io = true;
    }
  if(is_xfer_task)
    {
      assert( MPI_Comm_rank(XFER_COMM,&xfer_rank) == MPI_SUCCESS);
      if(xfer_rank == 0)
	{
	  isMasterXFER_ = true;
	  masterXFER_GlobalRank = num_local;
	}

      // On receiving XFER tasks, identify global rank of first sort
      // task that also resides on this host (for IPC communication)

      //grvy_printf(INFO,"[sortio]\n");

      if(xfer_rank >= num_io_hosts)
	{
	  int index = xfer_rank-num_io_hosts;   
	  localSortRank_ = first_sort_rank[index];
	  grvy_printf(DEBUG,"[sortio] --> IPC setup: XFER rank %i -> global master sort rank %i\n",
		      xfer_rank,localSortRank_);
	}
      
    }
  if(is_sort_task)
    {
      assert( MPI_Comm_rank(SORT_COMM,&sort_rank) == MPI_SUCCESS);
      if(sort_rank == 0)
	master_sort = true;

      // For the master Sort task on each host, identify global rank
      // of XFER task that also resides on this host (for IPC
      // communication)

      for(int i=0;i<nxfer_tasks;i++)
	{
	  if(num_local == first_sort_rank[i])
	    {
	      isLocalSortMaster_ = true;
	      localXferRank_     = xfer_comm_ranks[i+num_io_hosts];
	      grvy_printf(DEBUG,"[sortio] --> IPC setup: SORT rank %i -> global xfer rank %i\n",sort_rank,localXferRank_);
	    }
	}
    }

  MPI_Barrier(GLOB_COMM);

  // summarize the config (data printed from master rank to make the output easy on 
  // the eyes for the time being)

  if(master)
    {
      grvy_printf(INFO,"[sortio]\n");
      grvy_printf(INFO,"[sortio] MPI WorkGroup Summary (%i hosts, %i MPI tasks)\n",uniq_hosts.size(),num_tasks);
      grvy_printf(INFO,"[sortio]\n");
      grvy_printf(INFO,"[sortio] --------------------------------------------------------------\n");
      grvy_printf(INFO,"[sortio] [Hostname]  [Global Rank]  [IO Rank]  [XFER Rank]  [SORT Rank] \n");
      grvy_printf(INFO,"[sortio] --------------------------------------------------------------\n");
    }

  int ranks_tmp[3];
  MPI_Status status;

  for(int proc=0;proc<num_tasks;proc++)
    {
      if(master)
	{
	  grvy_printf(INFO,"[sortio]  %.8s       %.6i ",&hostnames_ALL[proc*MPI_MAX_PROCESSOR_NAME],proc);
	}

      if(master && (proc == 0) )
	{
	  if(is_io_task)
	    printf("      %.6i",io_rank);
	  else
	    printf("      ------");

	  if(is_xfer_task)
	    printf("      %.6i",xfer_rank);
	  else
	    printf("      ------");

	  if(is_sort_task)
	    printf("      %.6i\n",sort_rank);
	  else
	    printf("      ------\n");

	  continue;
	}

      if(num_local == proc)
	{
	  ranks_tmp[0] = (  is_io_task ) ?   io_rank : -1 ;
	  ranks_tmp[1] = (is_xfer_task ) ? xfer_rank : -1 ;
	  ranks_tmp[2] = (is_sort_task ) ? sort_rank : -1 ;
	  assert (MPI_Send(ranks_tmp,3,MPI_INTEGER,0,1000,GLOB_COMM) == MPI_SUCCESS) ;
	}
      
      if(master)
	{
	  assert (MPI_Recv(ranks_tmp,3,MPI_INTEGER,proc,1000,GLOB_COMM,&status) == MPI_SUCCESS) ;

	  for(int i=0;i<3;i++)
	    {
	      if(ranks_tmp[i] >= 0)
		printf("      %.6i",ranks_tmp[i]);
	      else
		printf("      ------");
	    }
	  grvy_printf(INFO,"\n");
	}
    }

  if(master)
    {
      grvy_printf(INFO,"[sortio] --------------------------------------------------------------\n");
      grvy_printf(INFO,"[sortio] --------------------------------------------------------------\n");
    }
  
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
    grvy_printf(INFO,"[sortio][IO/XFER][%.4i] added %i buff back to emptyQueue\n",io_rank,bufNum);
  }
  return;
}
