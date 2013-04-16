#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <omp.h>
#include <sstream>


#include <binUtils.h>
#include <ompUtils.h>
#include <parUtils.h>
#include <octUtils.h>

#include <TreeNode.h>
#include <gensort.h>
#include <sortRecord.h>

int main(int argc, char *argv[])
{
  int rank, p;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  int datasize = 10000000 + rank;

  std::vector<int> qq(datasize);

  srand (rank);
  for (int i=0; i<datasize; i++) {
    qq[i] = rand();
  }

  omp_par::merge_sort(&qq[0], &qq[qq.size()]);

  std::vector<int> sortBins = par::Sorted_approx_Select(qq, 10, MPI_COMM_WORLD);

  printf("size of qq   = %zi\n",qq.size());
  printf("size of bins = %zi\n",sortBins.size());

  char filename[1024];
  sprintf(filename,"/tmp/foo_%.4i",rank);

  const int MAXITER = 50;
  for(int iter=0;iter<MAXITER;iter++)
    {
      for(int i=0;i<datasize;i++)
	qq[i] = rand();

      printf("bucketing %i of %i...\n",iter,MAXITER);
      par::bucketDataAndWrite(qq, sortBins, filename, MPI_COMM_WORLD);
    }
  
  printf("rank %i of %i is done\n",rank,p);
  
  MPI_Finalize();
  return 0;
}
