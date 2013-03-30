//
// I/O class to aid in a large, distributed sort
//

#include "sortio.h"

sortio_Class::sortio_Class() 
{
  initialized               = false;
  master                    = false;
  master_io                 = false;
  master_xfer               = false;
  master_sort               = false;
  override_numfiles         = false;
  random_read_offset        = false;
  mpi_initialized_by_sortio = false;
  is_io_task                = false;
  is_xfer_task              = false;
  is_sort_task              = false;
  nio_tasks                 = 0;
  nxfer_tasks               = 0;
  nsort_tasks               = 0;
  num_records_read          = 0;
  basename                  = "part";

  setvbuf( stdout, NULL, _IONBF, 0 );
}

sortio_Class::~sortio_Class() 
{
  if(mpi_initialized_by_sortio)
    MPI_Finalize();
}

void sortio_Class::Override_nFiles(int nfiles)
{
  override_numfiles = true;
  num_files_total   = nfiles;
  return;
}

void sortio_Class::Summarize()
{

  fflush(NULL);
  MPI_Barrier(IO_COMM);

  //  setvbuf( stdout, NULL, _IONBF, 0 );

  gt.Finalize();

  if(master)
    printf("\n[sortio] --- Local Read Performance----------- \n");

  if(master)
    gt.Summarize();

  fflush(NULL);
  MPI_Barrier(IO_COMM);

  // local performance

  double time_local = gt.ElapsedSeconds("Raw Read");
  double read_rate  = 1.0*num_records_read*REC_SIZE/(1000*1000*1000*time_local);


  assert(time_local > 0.0);
  assert(num_records_read > 0);

#if 0
  //printf("REC_SIZE = %i\n",REC_SIZE);
  printf("[sortio][%4i]: Total (local) read speed = %7.3f (GB/sec)\n",io_rank,read_rate);
#endif

  fflush(NULL);
  MPI_Barrier(IO_COMM);

  // global performance

  unsigned long num_records_global;
  double time_best;
  double time_worst;
  double time_avg;
  double aggregate_rate;

  MPI_Allreduce(&num_records_read,&num_records_global,1,MPI_UNSIGNED_LONG,MPI_SUM,IO_COMM);
  MPI_Allreduce(&time_local,&time_worst,    1,MPI_DOUBLE,MPI_MAX,IO_COMM);
  MPI_Allreduce(&time_local,&time_best,     1,MPI_DOUBLE,MPI_MIN,IO_COMM);
  MPI_Allreduce(&time_local,&time_avg,      1,MPI_DOUBLE,MPI_SUM,IO_COMM);
  MPI_Allreduce(&read_rate, &aggregate_rate,1,MPI_DOUBLE,MPI_SUM,IO_COMM);

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
      printf("worst time from read task = %f\n",time_worst);
      printf("best  time from read task = %f\n",time_best);
      printf("avg.  time from read task = %f\n",time_avg);
      printf("[sortio] --> Global    read performance = %7.3f (GB/sec)\n",total_gbs/time_worst);
      printf("[sortio] --> Average   read performance = %7.3f (GB/sec)\n",total_gbs/time_avg);
      printf("[sortio] --> Aggregate read performance = %7.3f (GB/sec)\n",aggregate_rate);
    } 
  

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
      if(!override_numfiles)
	assert( iparse.Read_Var("sortio/num_files",&num_files_total) != 0);

      // Register defaults

      iparse.Register_Var("sortio/max_read_buffers",     10);
      iparse.Register_Var("sortio/max_file_size_in_mbs",100);

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

  std::map<std::string,std::vector<int> > uniq_hosts; // hostname -> global MPI rank mapping

  std::vector<int>    tmp_scatter;
  std::vector< std::vector<int> > scatter_comm_ranks(num_io_hosts); // dedicated Scatter comms from each IO rank

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

      //scatter_comm_ranks.reserve(num_io_hosts);

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
	      scatter_comm_ranks[count].push_back((*it).second[0]);
	    }
	  else
	    {
	      xfer_comm_ranks.push_back((*it).second[0]);
	      tmp_scatter.push_back((*it).second[0]);

	      for(int proc=1;proc<(*it).second.size();proc++)
		sort_comm_ranks.push_back((*it).second[proc]);
	    }
	    
	  grvy_printf(INFO,"[sortio]    %s -> %3i MPI task(s)/host\n",(*it).first.c_str(),(*it).second.size());
	  count++;
	}

      nio_tasks   = io_comm_ranks.size();
      nxfer_tasks = xfer_comm_ranks.size();
      nsort_tasks = sort_comm_ranks.size();

      // quick sanity checks

      assert(nio_tasks   > 0);
      assert(nxfer_tasks > 0);
      assert(nxfer_tasks > nio_tasks);
      assert(nsort_tasks > 0);
      assert(nio_tasks == num_io_hosts);
      assert(nxfer_tasks + nsort_tasks <= num_tasks);
      assert(tmp_scatter.size() == (nxfer_tasks - nio_tasks) );

      // Complete build out of scatter_comm ranks - the receiving XFER
      // ranks were cached above in tmp_scatter; now, just need to append them
      // to each scatter_comm_ranks entry;

      grvy_printf(INFO,"[sortio]\n");
      grvy_printf(INFO,"[sortio] Defining %i data transfer Scatter groups...\n",nio_tasks);
      
      for(int i=0;i<nio_tasks;i++)
	{
	  assert(scatter_comm_ranks[i].size() == 1);
	  scatter_comm_ranks[i].insert(scatter_comm_ranks[i].end(),tmp_scatter.begin(),tmp_scatter.end());

	  assert(scatter_comm_ranks[i].size() == (nxfer_tasks - nio_tasks + 1) );
	  printf("[sortio] --> Set %i global rank as leader (%zi ranks total)\n",scatter_comm_ranks[i][0],scatter_comm_ranks[i].size());
	  if(i == 0)
	    nscatter_tasks = scatter_comm_ranks[i].size();
	  else
	    assert (scatter_comm_ranks[i].size() == nscatter_tasks);
	}


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
  assert( MPI_Bcast(&nscatter_tasks,1,MPI_INTEGER,0,GLOB_COMM) == MPI_SUCCESS );

  if(!master)
    {
      io_comm_ranks.reserve  (nio_tasks);
      xfer_comm_ranks.reserve(nxfer_tasks);
      sort_comm_ranks.reserve(nsort_tasks);

      //      printf("[rank %i]:nscatter_tasks = %i (space = %zi)\n",num_local,nscatter_tasks,
      //	     scatter_comm_ranks.size());
      for(int i=0;i<nio_tasks;i++)
	scatter_comm_ranks[i].reserve(nscatter_tasks);
    }

  assert( MPI_Bcast(  io_comm_ranks.data(),  nio_tasks,MPI_INTEGER,0,GLOB_COMM) == MPI_SUCCESS);
  assert( MPI_Bcast(xfer_comm_ranks.data(),nxfer_tasks,MPI_INTEGER,0,GLOB_COMM) == MPI_SUCCESS);
  assert( MPI_Bcast(sort_comm_ranks.data(),nsort_tasks,MPI_INTEGER,0,GLOB_COMM) == MPI_SUCCESS);

  for(int i=0;i<nio_tasks;i++)
    assert( MPI_Bcast(scatter_comm_ranks[i].data(),nscatter_tasks,MPI_INTEGER,0,GLOB_COMM) == MPI_SUCCESS);

  MPI_Group group_global;
  MPI_Group group_io;
  MPI_Group group_xfer;
  MPI_Group group_sort;
  MPI_Group group_scatter;

  assert( MPI_Comm_group(GLOB_COMM,&group_global)  == MPI_SUCCESS );

  // New groups for IO, XFER, SORT, and special Scatter group 

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
	master_xfer = true;
    }
  if(is_sort_task)
    {
      assert( MPI_Comm_rank(SORT_COMM,&sort_rank) == MPI_SUCCESS);
      if(sort_rank == 0)
	master_sort = true;
    }

  // To facilitate data transfer from IO tasks to SORT tasks, we build
  // up a number of convenience MPI_Scatter() communicators. There are
  // num_io_hosts communicators which each contain all of the SORT MPI
  // ranks, plus 1 IO ranks as the leader.

  if(is_xfer_task)
    {
      Scatter_comms.resize(num_io_hosts);
      
    }

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
