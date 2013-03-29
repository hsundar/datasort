#include "sortio.h"

// --------------------------------------------------------------------
// ReadFiles(): primary routine for reading raw sort data
//
// * Operates on IO_COMM communicator
// * Threading - data is buffered from thread 1 to thread 0 as it is 
//               read for subsequent distribution to sort tasks.
// --------------------------------------------------------------------

void sortio_Class::ReadFiles()
{

  assert(initialized);

  // This return only operates on IO_tasks

  if(!is_io_task)
    return;

  gt.BeginTimer("Raw Read");

  // Initialize read buffers - todo: think about memory pinning here

  const int MAX_REGIONS          = 10;	// quick local hack - todo: read from input
  const int MAX_FILE_SIZE_IN_MBS = 100;

  std::vector<unsigned char *> regions;
  regions.reserve(MAX_REGIONS);

  for(int i=0;i<MAX_REGIONS;i++)
    {
      regions[i] = (unsigned char*) calloc(MAX_FILE_SIZE_IN_MBS*1024*1024,sizeof(unsigned char));
      assert(regions[i] != NULL);
    }

  // Initialize region_flag used for thread coordination between
  // reader and data xfer threads. If false, the region is empty and
  // is available to be read into. If true, the region is populated
  // with sort data and is ready to be read from by the xfer thread.

  std::vector<bool> region_flag(MAX_REGIONS,false);

  // Initialize and launch threading environment for an asychronous data
  // transfer mechanism

  const int num_io_threads_per_host = 2;
  omp_set_num_threads(num_io_threads_per_host);

#pragma omp parallel
#pragma omp sections
  {

    int thread_id;

    #pragma omp section		// XFER thread
    {
      Transfer_Tasks_Work();
      usleep(3000000);
      //thread_id = omp_get_thread_num(); 
      //      printf("[%i]: thread id for Master thread = %i\n",io_rank,thread_id);

    #pragma omp critical (io_region_update)
      {
	// update region_flag here
      }

    }

    #pragma omp section		// IO thread
    {
      IO_Tasks_Work();
      //      thread_id = omp_get_thread_num(); 
      //      printf("[%i]: thread id for child reader thread = %i\n",io_rank,thread_id);

      //      int count = 0;
      //      for(int i=0;i<MAX_REGIONS;i++)
    }

  }

  return;

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

#if 0
	  if(read_size != REC_SIZE)
	    MPI_Abort(MPI_COMM_WORLD,43);
#endif

	  num_records_read++;
	}

      fclose(fp);

#if 0
      printf("[sortio][%3i]: records read = %i\n",io_rank,num_records_read);
#endif

    }

  gt.EndTimer("Raw Read");

}
