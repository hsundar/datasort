//
// I/O class to aid in a large, distributed sort
//

#include "sortio.h"

sortio_Class::sortio_Class() 
{
  initialized      = 0;
  master           = false;
  nio_tasks        = 0;
  num_records_read = 0;
  basename         = "part";
}

sortio_Class::~sortio_Class() 
{
}

void sortio_Class::Summarize()
{
  gt.Finalize();

  if(master)
    {
      gt.Summarize();
      printf("\n[sortio] --- Local Read Performance----------- \n");
    }

  MPI_Barrier(IO_COMM);

  // local performance

  double time_local = gt.ElapsedSeconds("Raw Read");
  double read_rate  = num_records_read*REC_SIZE/(1000*1000*1000*time_local);

  printf("[sortio][%i]: Total (local) read speed = %7.3f (GB/sec)\n",io_rank,read_rate);

  // global performance

  int num_records_global;
  double time_worst;
  double aggregate_rate;

  MPI_Allreduce(&num_records_read,&num_records_global,1,MPI_INTEGER,MPI_SUM,IO_COMM);
  MPI_Allreduce(&time_local,&time_worst,    1,MPI_DOUBLE,MPI_MAX,IO_COMM);
  MPI_Allreduce(&read_rate, &aggregate_rate,1,MPI_DOUBLE,MPI_SUM,IO_COMM);

  if(master)
    {
      double total_gbs = 1.0*num_records_global*REC_SIZE/(1000*1000*1000);

      printf("\n");
      printf("[sortio] --- Global Read Performance ----------- \n");
      printf("[sortio] --> Total records read = %i\n",num_records_global);
      if(total_gbs < 1000)
	printf("[sortio] --> Total amount of data read  = %7.3f (GBs)\n",total_gbs);
      else
	printf("[sortio] --> Total amount of data read  = %7.3f (TBs)\n",total_gbs/1000.0);

      printf("\n");
      printf("[sortio] --> Global    read performance = %7.3f (GB/sec)\n",total_gbs/time_worst);
      printf("[sortio] --> Aggregate read performance = %7.3f (GB/sec)\n",aggregate_rate);
    } 
  

  return;
}

void sortio_Class::Initialize(std::string ifile, MPI_Comm COMM)
{
  assert(!initialized);

  gt.Init("Sort IO Subsystem");

  IO_COMM = COMM;

  MPI_Comm_size (IO_COMM, &nio_tasks);
  MPI_Comm_rank (IO_COMM, &io_rank  );

  // Read I/O runtime controls

  if(io_rank == 0)
    {
      master = true;
      GRVY::GRVY_Input_Class iparse;

      printf("[sortio]: # of MPI reader tasks = %4i\n",nio_tasks);

      assert(iparse.Open    ("input.dat")                         != 0);
      assert(iparse.Read_Var("sortio/num_files",&num_files_total) != 0);
      assert(iparse.Read_Var("sortio/input_dir",&indir)           != 0);

      printf("[sortio]: --> total number of files to read = %i\n",num_files_total);
      printf("[sortio]: --> input directory               = %s\n",indir.c_str());
    }

  // Broadcast necessary runtime controls to I/O children

  int tmp_string_size = indir.size() + 1;
  char *tmp_string    = NULL;

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
  return; 
}

void sortio_Class::ReadFiles()
{

  gt.BeginTimer("Raw Read");
  
  int num_iters = (num_files_total+nio_tasks-1)/nio_tasks;
  int read_size;

  std::string filebase(indir);
  filebase += "/";
  filebase += basename;

  printf("%i: num_iters = %i\n",io_rank,num_iters);

  for(int iter=0;iter<num_iters;iter++)
    {
      if(master)
	printf("[sortio][%3i]: starting read iteration %4i of %4i total\n",io_rank,iter,num_iters-1);

      std::ostringstream s_id;
      int file_suffix = iter*nio_tasks + io_rank;

      if(file_suffix >= num_files_total)
	{
	  gt.EndTimer("Raw Read");
	  return;
	}

      s_id << file_suffix;
      std::string infile = filebase + s_id.str();

      printf("[sortio][%3i]: filename = %s\n",io_rank,infile.c_str());

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

      printf("[sortio][%3i]: records read = %i\n",io_rank,num_records_read);

    }

  gt.EndTimer("Raw Read");

}

