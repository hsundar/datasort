//
// I/O class to aid in a large, distributed sort
//

#include "sortio.h"


sortio_Class::sortio_Class() 
{
  initialized  = 0;
  master       = false;
  nio_tasks    = 0;
  basename     = "part";
}

void sortio_Class::Initialize(std::string ifile, MPI_Comm IO_COMM)
{
  assert(!initialized);

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

  // Bcast necessary runtime controls to I/O children

  int tmp_string_size = indir.size()+1;
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

  if(master)
    printf("end of %s\n",__func__);

  initialized = true;

  return; 
}

void sortio_Class::ReadFiles()
{
  
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
      s_id << iter*nio_tasks + io_rank;
      std::string infile = filebase + s_id.str();

      printf("[sortio][%3i]: filename = %s\n",io_rank,infile.c_str());

      FILE *fp = fopen(infile.c_str(),"r");
      
      if(fp == NULL)
	MPI_Abort(MPI_COMM_WORLD,42);

      read_size = REC_SIZE;

      int num_records_read = 0;

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

}

