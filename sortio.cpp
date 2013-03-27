//
// I/O class to aid in a large, distributed sort
//

#include "sortio.h"

sortio_Class::sortio_Class() 
{
  initialized               = false;
  master                    = false;
  override_numfiles         = false;
  random_read_offset        = false;
  mpi_initialized_by_sortio = false;
  nio_tasks                 = 0;
  num_records_read          = 0;
  basename                  = "part";

  setvbuf( stdout, NULL, _IONBF, 0 );
}

sortio_Class::~sortio_Class() 
{
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
      MPI_Init(NULL,NULL);
      mpi_initialized_by_sortio = true;
    }

  MPI_Comm_size(COMM,&num_tasks);
  MPI_Comm_rank(COMM,&num_local);

  GLOB_COMM = COMM;

  IO_COMM = COMM;

  MPI_Comm_size (IO_COMM, &nio_tasks);
  MPI_Comm_rank (IO_COMM, &io_rank  );

  // Read I/O runtime controls

  if(num_local == 0)
    {
      master = true;
      GRVY::GRVY_Input_Class iparse;

      printf("[sortio]: # of MPI reader tasks = %4i\n",nio_tasks);

      assert(iparse.Open    ("input.dat")                         != 0);
      if(!override_numfiles)
	assert(iparse.Read_Var("sortio/num_files",&num_files_total) != 0);
      assert(iparse.Read_Var("sortio/input_dir",&indir)           != 0);

      printf("[sortio]: --> total number of files to read = %i\n",num_files_total);
      printf("[sortio]: --> input directory               = %s\n",indir.c_str());
    }

  // Broadcast necessary runtime controls to I/O children

  int tmp_string_size = indir.size() + 1;
  char *tmp_string    = NULL;

  //  random_read_offset  = true;

  MPI_Bcast(&num_files_total,1,MPI_INTEGER,0,IO_COMM);
  MPI_Bcast(&tmp_string_size,1,MPI_INTEGER,0,IO_COMM);
  
  tmp_string = (char *)calloc(tmp_string_size,sizeof(char));
  strcpy(tmp_string,indir.c_str());

  MPI_Bcast(tmp_string,tmp_string_size,MPI_CHAR,0,IO_COMM);

  if(!master)
    indir = tmp_string;

  free(tmp_string);

  MPI_Barrier(IO_COMM);

  initialized = true;
  gt.EndTimer("Initialize");
  return; 
}

void sortio_Class::ReadFiles()
{

  assert(initialized);
  gt.BeginTimer("Raw Read");
  
  int num_iters = (num_files_total+nio_tasks-1)/nio_tasks;
  int read_size;

  std::string filebase(indir);
  filebase += "/";
  filebase += basename;

#if 0
  printf("%i: num_iters = %i\n",io_rank,num_iters);
#endif

  for(int iter=0;iter<num_iters;iter++)
    {

#if 0
      if(master)
	printf("[sortio][%3i]: starting read iteration %4i of %4i total\n",io_rank,iter,num_iters-1);
#endif

      std::ostringstream s_id;
      int file_suffix = iter*nio_tasks + io_rank;

      // Optionally randomize so we can minimize local host
      // file-caching for more legitimate reads; the idea here is to
      // help enable repeat testing on the same hosts for smaller
      // dataset sizes. We just keep a local file here an iterate an
      // offset on each run

      if(random_read_offset)
	{

	  int min_index = iter*nio_tasks;
	  int max_index = iter*nio_tasks + (nio_tasks-1);
	  int offset;

	  if(master)
	    {
	      FILE *fp_offset = fopen(".offset.tmp","r");
	      if(fp_offset != NULL)
		{
		  fscanf(fp_offset,"%i",&offset);
		  offset += 17;
		  fclose(fp_offset);
		}
	      else
		offset = 1;

	      if(offset > nio_tasks)
		offset = 0;

	      fp_offset = fopen(".offset.tmp","w");
	      assert(fp_offset != NULL);
	      fprintf(fp_offset,"%i\n",offset);
	      fclose(fp_offset);

#if 0
	      printf("min_index = %4i\n",min_index);
	      printf("max_index = %4i\n",max_index);
	      printf("offset    = %4i\n",offset);
#endif
	    }

	  MPI_Bcast(&offset,1,MPI_INTEGER,0,IO_COMM);

#if 0
	  printf("[sortio][%3i]: original file_suffix = %3i\n",io_rank,file_suffix);
#endif
	  file_suffix += offset;

	  if(file_suffix > (iter*nio_tasks + (nio_tasks-1)) )
	    file_suffix -= nio_tasks;

#if 0
	  printf("[sortio][%3i]: new file_suffix = %3i\n",io_rank,file_suffix);
#endif
	}

      if(file_suffix >= num_files_total)
	{
	  gt.EndTimer("Raw Read");
	  return;
	}

      s_id << file_suffix;
      std::string infile = filebase + s_id.str();

#if 1
      printf("[sortio][%3i]: filename = %s\n",io_rank,infile.c_str());
#endif

      FILE *fp = fopen(infile.c_str(),"r");
      
      if(fp == NULL)
	MPI_Abort(MPI_COMM_WORLD,42);

      read_size = REC_SIZE;

      while(read_size == REC_SIZE)
	{
	  read_size = fread(rec_buf,1,REC_SIZE,fp);

	  if(read_size == 0)
	    break;

	  if(read_size != REC_SIZE)
	    MPI_Abort(MPI_COMM_WORLD,43);

	  num_records_read++;
	}

      fclose(fp);

#if 0
      printf("[sortio][%3i]: records read = %i\n",io_rank,num_records_read);
#endif

    }

  gt.EndTimer("Raw Read");

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

  grvy_printf(DEBUG,"[sortio]: Detected global Rank %i -> %s\n",num_local,hostname);

  char *hostnames_ALL;

  if(num_local == 0) 
    {
      hostnames_ALL = (char *)malloc(num_tasks*MPI_MAX_PROCESSOR_NAME*sizeof(char));
      assert(hostnames_ALL != NULL);
    }

  MPI_Gather(&hostname[0],      MPI_MAX_PROCESSOR_NAME,MPI_CHAR,
	     &hostnames_ALL[0],MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,GLOB_COMM);

  // Determine unique hostnames -> rank mapping

  if(num_local == 0)
    {
      std::map<std::string,int> uniq_hosts;

      for(int i=0;i<num_tasks;i++)
	{
	  std::string host = &hostnames_ALL[i*MPI_MAX_PROCESSOR_NAME];
	  grvy_printf(DEBUG,"[sortio]: parsed host = %s\n",host.c_str());
	  uniq_hosts[host]++;
	}

      free(hostnames_ALL);

      std::map<std::string,int>::iterator it;

      for(it = uniq_hosts.begin(); it != uniq_hosts.end(); it++ ) 
	{
	  grvy_printf(INFO,"[sortio]: %s -> %3i MPI task(s)/host\n",(*it).first.c_str(),(*it).second);
	}
    }

  return;
}
