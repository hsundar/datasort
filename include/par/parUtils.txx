// -*-c++-*-

/**
  @file parUtils.txx
  @brief Definitions of the templated functions in the par module.
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  @author Hari Sundar, hsundar@gmail.com
  @author Shravan Veerapaneni, shravan@seas.upenn.edu
  @author Santi Swaroop Adavani, santis@gmail.com
 */

#include "binUtils.h"
#include "seqUtils.h"
#include "dtypes.h"
#include <cassert>
#include <iostream>
#include <algorithm>
#include <cstring>
#include "dendro.h"
#include <fcntl.h>

#ifdef _PROFILE_SORT
  #include "sort_profiler.h"
#endif

#include "ompUtils.h"
#include "binUtils.h"
#include <mpi.h>

#ifdef __DEBUG__
#ifndef __DEBUG_PAR__
#define __DEBUG_PAR__
#endif
#endif

//#define KWAY 16  // karl testing

#ifndef KWAY
		#define KWAY 8
#endif 

// #define OVERLAP_KWAY_COMM

		// #define LOAD_BALANCE_COMM

namespace par {

  template <typename T>
    inline int Mpi_Isend(T* buf, int count, int dest, int tag,
        MPI_Comm comm, MPI_Request* request) {

      MPI_Isend(buf, count, par::Mpi_datatype<T>::value(),
          dest, tag, comm, request);

      return 1;

    }

  template <typename T>
    inline int Mpi_Issend(T* buf, int count, int dest, int tag,
        MPI_Comm comm, MPI_Request* request) {

      MPI_Issend(buf, count, par::Mpi_datatype<T>::value(),
          dest, tag, comm, request);

      return 1;

    }

  template <typename T>
    inline int Mpi_Recv(T* buf, int count, int source, int tag,
        MPI_Comm comm, MPI_Status* status) {

      MPI_Recv(buf, count, par::Mpi_datatype<T>::value(),
          source, tag, comm, status);

      return 1;

    }

  template <typename T>
    inline int Mpi_Irecv(T* buf, int count, int source, int tag,
        MPI_Comm comm, MPI_Request* request) {

      MPI_Irecv(buf, count, par::Mpi_datatype<T>::value(),
          source, tag, comm, request);

      return 1;

    }

  template <typename T, typename S>
    inline int Mpi_Sendrecv( T* sendBuf, int sendCount, int dest, int sendTag,
        S* recvBuf, int recvCount, int source, int recvTag,
        MPI_Comm comm, MPI_Status* status) {
      PROF_PAR_SENDRECV_BEGIN

        MPI_Sendrecv(sendBuf, sendCount, par::Mpi_datatype<T>::value(), dest, sendTag,
            recvBuf, recvCount, par::Mpi_datatype<S>::value(), source, recvTag, comm, status);

      PROF_PAR_SENDRECV_END
    }

  template <typename T>
    inline int Mpi_Scan( T* sendbuf, T* recvbuf, int count, MPI_Op op, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_SCAN_BEGIN

        MPI_Scan(sendbuf, recvbuf, count, par::Mpi_datatype<T>::value(), op, comm);

      PROF_PAR_SCAN_END
    }

  template <typename T>
    inline int Mpi_Allreduce(T* sendbuf, T* recvbuf, int count, MPI_Op op, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_ALLREDUCE_BEGIN

        MPI_Allreduce(sendbuf, recvbuf, count, par::Mpi_datatype<T>::value(), op, comm);

      PROF_PAR_ALLREDUCE_END
    }

  template <typename T>
    inline int Mpi_Alltoall(T* sendbuf, T* recvbuf, int count, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_ALL2ALL_BEGIN

        MPI_Alltoall(sendbuf, count, par::Mpi_datatype<T>::value(),
            recvbuf, count, par::Mpi_datatype<T>::value(), comm);

      PROF_PAR_ALL2ALL_END
    }

  template <typename T>
    inline int Mpi_Alltoallv
    (T* sendbuf, int* sendcnts, int* sdispls, 
     T* recvbuf, int* recvcnts, int* rdispls, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif

        MPI_Alltoallv(
            sendbuf, sendcnts, sdispls, par::Mpi_datatype<T>::value(), 
            recvbuf, recvcnts, rdispls, par::Mpi_datatype<T>::value(), 
            comm);
        return 0;
    }

  template <typename T>
    inline int Mpi_Gather( T* sendBuffer, T* recvBuffer, int count, int root, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_GATHER_BEGIN

        MPI_Gather(sendBuffer, count, par::Mpi_datatype<T>::value(),
            recvBuffer, count, par::Mpi_datatype<T>::value(), root, comm);

      PROF_PAR_GATHER_END
    }

  template <typename T>
    inline int Mpi_Bcast(T* buffer, int count, int root, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_BCAST_BEGIN

        MPI_Bcast(buffer, count, par::Mpi_datatype<T>::value(), root, comm);

      PROF_PAR_BCAST_END
    }

  template <typename T>
    inline int Mpi_Reduce(T* sendbuf, T* recvbuf, int count, MPI_Op op, int root, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_REDUCE_BEGIN

        MPI_Reduce(sendbuf, recvbuf, count, par::Mpi_datatype<T>::value(), op, root, comm);

      PROF_PAR_REDUCE_END
    }

  template <typename T>
    int Mpi_Allgatherv(T* sendBuf, int sendCount, T* recvBuf, 
        int* recvCounts, int* displs, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_ALLGATHERV_BEGIN

#ifdef __USE_A2A_FOR_MPI_ALLGATHER__

        int maxSendCount;
      int npes, rank;

      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      par::Mpi_Allreduce<int>(&sendCount, &maxSendCount, 1, MPI_MAX, comm);

      T* dummySendBuf = new T[maxSendCount*npes];
      assert(dummySendBuf);

      #pragma omp parallel for
      for(int i = 0; i < npes; i++) {
        for(int j = 0; j < sendCount; j++) {
          dummySendBuf[(i*maxSendCount) + j] = sendBuf[j];
        }
      }

      T* dummyRecvBuf = new T[maxSendCount*npes];
      assert(dummyRecvBuf);

      par::Mpi_Alltoall<T>(dummySendBuf, dummyRecvBuf, maxSendCount, comm);

      #pragma omp parallel for
      for(int i = 0; i < npes; i++) {
        for(int j = 0; j < recvCounts[i]; j++) {
          recvBuf[displs[i] + j] = dummyRecvBuf[(i*maxSendCount) + j];
        }
      }

      delete [] dummySendBuf;
      delete [] dummyRecvBuf;

#else

      MPI_Allgatherv(sendBuf, sendCount, par::Mpi_datatype<T>::value(),
          recvBuf, recvCounts, displs, par::Mpi_datatype<T>::value(), comm);

#endif

      PROF_PAR_ALLGATHERV_END
    }

  template <typename T>
    int Mpi_Allgather(T* sendBuf, T* recvBuf, int count, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_ALLGATHER_BEGIN

#ifdef __USE_A2A_FOR_MPI_ALLGATHER__

        int npes;
      MPI_Comm_size(comm, &npes);
      T* dummySendBuf = new T[count*npes];
      assert(dummySendBuf);
      #pragma omp parallel for
      for(int i = 0; i < npes; i++) {
        for(int j = 0; j < count; j++) {
          dummySendBuf[(i*count) + j] = sendBuf[j];
        }
      }
      par::Mpi_Alltoall<T>(dummySendBuf, recvBuf, count, comm);
      delete [] dummySendBuf;

#else

      MPI_Allgather(sendBuf, count, par::Mpi_datatype<T>::value(), 
          recvBuf, count, par::Mpi_datatype<T>::value(), comm);

#endif

      PROF_PAR_ALLGATHER_END
    }

  template <typename T>
    int Mpi_Alltoallv_sparse(T* sendbuf, int* sendcnts, int* sdispls, 
        T* recvbuf, int* recvcnts, int* rdispls, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_ALL2ALLV_SPARSE_BEGIN

#ifndef ALLTOALLV_FIX
      Mpi_Alltoallv
        (sendbuf, sendcnts, sdispls, 
         recvbuf, recvcnts, rdispls, comm);
#else

        int npes, rank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      int commCnt = 0;

      #pragma omp parallel for reduction(+:commCnt)
      for(int i = 0; i < rank; i++) {
        if(sendcnts[i] > 0) {
          commCnt++;
        }
        if(recvcnts[i] > 0) {
          commCnt++;
        }
      }

      #pragma omp parallel for reduction(+:commCnt)
      for(int i = (rank+1); i < npes; i++) {
        if(sendcnts[i] > 0) {
          commCnt++;
        }
        if(recvcnts[i] > 0) {
          commCnt++;
        }
      }

      MPI_Request* requests = new MPI_Request[commCnt];
      assert(requests);

      MPI_Status* statuses = new MPI_Status[commCnt];
      assert(statuses);

      commCnt = 0;

      //First place all recv requests. Do not recv from self.
      for(int i = 0; i < rank; i++) {
        if(recvcnts[i] > 0) {
          par::Mpi_Irecv<T>( &(recvbuf[rdispls[i]]) , recvcnts[i], i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      for(int i = (rank + 1); i < npes; i++) {
        if(recvcnts[i] > 0) {
          par::Mpi_Irecv<T>( &(recvbuf[rdispls[i]]) , recvcnts[i], i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      //Next send the messages. Do not send to self.
      for(int i = 0; i < rank; i++) {
        if(sendcnts[i] > 0) {
          par::Mpi_Issend<T>( &(sendbuf[sdispls[i]]), sendcnts[i], i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      for(int i = (rank + 1); i < npes; i++) {
        if(sendcnts[i] > 0) {
          par::Mpi_Issend<T>( &(sendbuf[sdispls[i]]), sendcnts[i], 
              i, 1, comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      //Now copy local portion.
#ifdef __DEBUG_PAR__
      assert(sendcnts[rank] == recvcnts[rank]);
#endif

      #pragma omp parallel for
      for(int i = 0; i < sendcnts[rank]; i++) {
        recvbuf[rdispls[rank] + i] = sendbuf[sdispls[rank] + i];
      }

      PROF_A2AV_WAIT_BEGIN

        MPI_Waitall(commCnt, requests, statuses);

      PROF_A2AV_WAIT_END

      delete [] requests;
      delete [] statuses;
#endif

      PROF_PAR_ALL2ALLV_SPARSE_END
    }

//*
  template <typename T>
    int Mpi_Alltoallv_dense(T* sbuff_, int* s_cnt_, int* sdisp_,
        T* rbuff_, int* r_cnt_, int* rdisp_, MPI_Comm c){

      //std::vector<double> tt(4096*200,0);
      //std::vector<double> tt_wait(4096*200,0);

#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_ALL2ALLV_DENSE_BEGIN

#ifndef ALLTOALLV_FIX
      Mpi_Alltoallv
        (sbuff_, s_cnt_, sdisp_,
         rbuff_, r_cnt_, rdisp_, c);
#else
  int kway = KWAY;
  int np, pid;
  MPI_Comm_size(c, &np);
  MPI_Comm_rank(c, &pid);
  int range[2]={0,np};
  int split_id, partner;

  std::vector<int> s_cnt(np);
  #pragma omp parallel for
  for(int i=0;i<np;i++){
    s_cnt[i]=s_cnt_[i]*sizeof(T)+2*sizeof(int);
  }
  std::vector<int> sdisp(np); sdisp[0]=0;
  omp_par::scan(&s_cnt[0],&sdisp[0],np);

  char* sbuff=new char[sdisp[np-1]+s_cnt[np-1]];
  #pragma omp parallel for
  for(int i=0;i<np;i++){
    ((int*)&sbuff[sdisp[i]])[0]=s_cnt[i];
    ((int*)&sbuff[sdisp[i]])[1]=pid;
    memcpy(&sbuff[sdisp[i]]+2*sizeof(int),&sbuff_[sdisp_[i]],s_cnt[i]-2*sizeof(int));
  }

  //int t_indx=0;
  int iter_cnt=0;
  while(range[1]-range[0]>1){
    iter_cnt++;
    if(kway>range[1]-range[0]) 
      kway=range[1]-range[0];

    std::vector<int> new_range(kway+1);
    for(int i=0;i<=kway;i++)
      new_range[i]=(range[0]*(kway-i)+range[1]*i)/kway;
    int p_class=(std::upper_bound(&new_range[0],&new_range[kway],pid)-&new_range[0]-1);
    int new_np=new_range[p_class+1]-new_range[p_class];
    int new_pid=pid-new_range[p_class];

    //Communication.
    {
      std::vector<int> r_cnt    (new_np*kway, 0);
      std::vector<int> r_cnt_ext(new_np*kway, 0);
      //Exchange send sizes.
      for(int i=0;i<kway;i++){
        MPI_Status status;
        int cmp_np=new_range[i+1]-new_range[i];
        int partner=(new_pid<cmp_np?       new_range[i]+new_pid: new_range[i+1]-1) ;
        assert(     (new_pid<cmp_np? true: new_range[i]+new_pid==new_range[i+1]  )); //Remove this.
        MPI_Sendrecv(&s_cnt[new_range[i]-new_range[0]], cmp_np, MPI_INT, partner, 0,
                     &r_cnt[new_np   *i ], new_np, MPI_INT, partner, 0, c, &status);

        //Handle extra communication.
        if(new_pid==new_np-1 && cmp_np>new_np){
          int partner=new_range[i+1]-1;
          std::vector<int> s_cnt_ext(cmp_np, 0);
          MPI_Sendrecv(&s_cnt_ext[       0], cmp_np, MPI_INT, partner, 0,
                       &r_cnt_ext[new_np*i], new_np, MPI_INT, partner, 0, c, &status);
        }
      }

      //Allocate receive buffer.
      std::vector<int> rdisp    (new_np*kway, 0);
      std::vector<int> rdisp_ext(new_np*kway, 0);
      int rbuff_size, rbuff_size_ext;
      char *rbuff, *rbuff_ext;
      {
        omp_par::scan(&r_cnt    [0], &rdisp    [0],new_np*kway);
        omp_par::scan(&r_cnt_ext[0], &rdisp_ext[0],new_np*kway);
        rbuff_size     = rdisp    [new_np*kway-1] + r_cnt    [new_np*kway-1];
        rbuff_size_ext = rdisp_ext[new_np*kway-1] + r_cnt_ext[new_np*kway-1];
        rbuff     = new char[rbuff_size    ];
        rbuff_ext = new char[rbuff_size_ext];
      }

      //Sendrecv data.
      //*
      int my_block=kway;
      while(pid<new_range[my_block]) my_block--;
//      MPI_Barrier(c);
      for(int i_=0;i_<=kway/2;i_++){
        int i1=(my_block+i_)%kway;
        int i2=(my_block+kway-i_)%kway;

        for(int j=0;j<(i_==0 || i_==kway/2?1:2);j++){
          int i=(i_==0?i1:((j+my_block/i_)%2?i1:i2));
          MPI_Status status;
          int cmp_np=new_range[i+1]-new_range[i];
          int partner=(new_pid<cmp_np?       new_range[i]+new_pid: new_range[i+1]-1) ;

          int send_dsp     =sdisp[new_range[i  ]-new_range[0]  ];
          int send_dsp_last=sdisp[new_range[i+1]-new_range[0]-1];
          int send_cnt     =s_cnt[new_range[i+1]-new_range[0]-1]+send_dsp_last-send_dsp;

//          double ttt=omp_get_wtime();
//          MPI_Sendrecv(&sbuff[send_dsp], send_cnt>0?1:0, MPI_BYTE, partner, 0,
//                       &rbuff[rdisp[new_np  * i ]], (r_cnt[new_np  *(i+1)-1]+rdisp[new_np  *(i+1)-1]-rdisp[new_np  * i ])>0?1:0, MPI_BYTE, partner, 0, c, &status);
//          tt_wait[200*pid+t_indx]=omp_get_wtime()-ttt;
//
//          ttt=omp_get_wtime();
          MPI_Sendrecv(&sbuff[send_dsp], send_cnt, MPI_BYTE, partner, 0,
                       &rbuff[rdisp[new_np  * i ]], r_cnt[new_np  *(i+1)-1]+rdisp[new_np  *(i+1)-1]-rdisp[new_np  * i ], MPI_BYTE, partner, 0, c, &status);
//          tt[200*pid+t_indx]=omp_get_wtime()-ttt;
//          t_indx++;

          //Handle extra communication.
          if(pid==new_np-1 && cmp_np>new_np){
            int partner=new_range[i+1]-1;
            std::vector<int> s_cnt_ext(cmp_np, 0);
            MPI_Sendrecv(                       NULL,                                                                       0, MPI_BYTE, partner, 0,
                         &rbuff[rdisp_ext[new_np*i]], r_cnt_ext[new_np*(i+1)-1]+rdisp_ext[new_np*(i+1)-1]-rdisp_ext[new_np*i], MPI_BYTE, partner, 0, c, &status);
          }
        }
      }
      /*/
      {
        MPI_Request* requests = new MPI_Request[4*kway];
        MPI_Status * statuses = new MPI_Status[4*kway];
        int commCnt=0;
        for(int i=0;i<kway;i++){
          MPI_Status status;
          int cmp_np=new_range[i+1]-new_range[i];
          int partner=              new_range[i]+new_pid;
          if(partner<new_range[i+1]){
            MPI_Irecv(&rbuff    [rdisp    [new_np*i]], r_cnt    [new_np*(i+1)-1]+rdisp    [new_np*(i+1)-1]-rdisp    [new_np*i ], MPI_BYTE, partner, 0, c, &requests[commCnt]); commCnt++;
          }

          //Handle extra recv.
          if(new_pid==new_np-1 && cmp_np>new_np){
            int partner=new_range[i+1]-1;
            MPI_Irecv(&rbuff_ext[rdisp_ext[new_np*i]], r_cnt_ext[new_np*(i+1)-1]+rdisp_ext[new_np*(i+1)-1]-rdisp_ext[new_np*i ], MPI_BYTE, partner, 0, c, &requests[commCnt]); commCnt++;
          }
        }
        for(int i=0;i<kway;i++){
          MPI_Status status;
          int cmp_np=new_range[i+1]-new_range[i];
          int partner=(new_pid<cmp_np?  new_range[i]+new_pid: new_range[i+1]-1);
          int send_dsp     =sdisp[new_range[i  ]-new_range[0]  ];
          int send_dsp_last=sdisp[new_range[i+1]-new_range[0]-1];
          int send_cnt     =s_cnt[new_range[i+1]-new_range[0]-1]+send_dsp_last-send_dsp;
          MPI_Issend (&sbuff[send_dsp], send_cnt, MPI_BYTE, partner, 0, c, &requests[commCnt]); commCnt++;
        }
        MPI_Waitall(commCnt, requests, statuses);
        delete[] requests;
        delete[] statuses;
      }// */

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      //Rearrange received data.
      {
        if(sbuff!=NULL) delete[] sbuff;
        sbuff=new char[rbuff_size+rbuff_size_ext];

        std::vector<int>  cnt_new(2*new_np*kway, 0);
        std::vector<int> disp_new(2*new_np*kway, 0);
        for(int i=0;i<new_np;i++)
        for(int j=0;j<kway;j++){
          cnt_new[(i*2  )*kway+j]=r_cnt    [j*new_np+i];
          cnt_new[(i*2+1)*kway+j]=r_cnt_ext[j*new_np+i];
        }
        omp_par::scan(&cnt_new[0], &disp_new[0],2*new_np*kway);

        #pragma omp parallel for
        for(int i=0;i<new_np;i++)
        for(int j=0;j<kway;j++){
          memcpy(&sbuff[disp_new[(i*2  )*kway+j]], &rbuff    [rdisp    [j*new_np+i]], r_cnt    [j*new_np+i]);
          memcpy(&sbuff[disp_new[(i*2+1)*kway+j]], &rbuff_ext[rdisp_ext[j*new_np+i]], r_cnt_ext[j*new_np+i]);
        }

        //Free memory.
        if(rbuff    !=NULL) delete[] rbuff    ;
        if(rbuff_ext!=NULL) delete[] rbuff_ext;

        s_cnt.clear();
        s_cnt.resize(new_np,0);
        sdisp.resize(new_np);
        for(int i=0;i<new_np;i++){
          for(int j=0;j<2*kway;j++)
            s_cnt[i]+=cnt_new[i*2*kway+j];
          sdisp[i]=disp_new[i*2*kway];
        }
      }

/*
      //Rearrange received data.
      {
        int * s_cnt_old=&s_cnt[new_range[0]-range[0]];
        int * sdisp_old=&sdisp[new_range[0]-range[0]];
        
        std::vector<int> s_cnt_new(&s_cnt_old[0],&s_cnt_old[new_np]);
        std::vector<int> sdisp_new(new_np       ,0                 );
        #pragma omp parallel for
        for(int i=0;i<new_np;i++){
          s_cnt_new[i]+=r_cnt[i];
        }
        omp_par::scan(&s_cnt_new[0],&sdisp_new[0],new_np);

        //Copy data to sbuff_new.
        char* sbuff_new=new char[sdisp_new[new_np-1]+s_cnt_new[new_np-1]];
        #pragma omp parallel for
        for(int i=0;i<new_np;i++){
          memcpy(&sbuff_new[sdisp_new[i]                      ],&sbuff   [sdisp_old[i]],s_cnt_old[i]);
          memcpy(&sbuff_new[sdisp_new[i]+s_cnt_old[i]         ],&rbuff   [rdisp    [i]],r_cnt    [i]);
        }

        //Free memory.
        if(sbuff   !=NULL) delete[] sbuff   ;
        if(rbuff   !=NULL) delete[] rbuff   ;

        //Substitute data for next iteration.
        s_cnt=s_cnt_new;
        sdisp=sdisp_new;
        sbuff=sbuff_new;
      }*/
    }

    range[0]=new_range[p_class  ];
    range[1]=new_range[p_class+1];
    //range[0]=new_range[0];
    //range[1]=new_range[1];
  }

  //Copy data to rbuff_.
  std::vector<char*> buff_ptr(np);
  char* tmp_ptr=sbuff;
  for(int i=0;i<np;i++){
    int& blk_size=((int*)tmp_ptr)[0];
    buff_ptr[i]=tmp_ptr;
    tmp_ptr+=blk_size;
  }
  #pragma omp parallel for
  for(int i=0;i<np;i++){
    int& blk_size=((int*)buff_ptr[i])[0];
    int& src_pid=((int*)buff_ptr[i])[1];
    assert(blk_size-2*sizeof(int)<=r_cnt_[src_pid]*sizeof(T));
    memcpy(&rbuff_[rdisp_[src_pid]],buff_ptr[i]+2*sizeof(int),blk_size-2*sizeof(int));
  }
/*
  std::vector<double> tt_sum(4096*200,0);
  std::vector<double> tt_wait_sum(4096*200,0);
  MPI_Reduce(&tt[0], &tt_sum[0], 4096*200, MPI_DOUBLE, MPI_SUM, 0, c);
  MPI_Reduce(&tt_wait[0], &tt_wait_sum[0], 4096*200, MPI_DOUBLE, MPI_SUM, 0, c);

#define MAX_PROCS 4096

  if(np==MAX_PROCS){
    if(!pid) std::cout<<"Tw=[";
    size_t j=0;
    for(size_t i=0;i<200;i++){
      for(j=0;j<MAX_PROCS;j++)
        if(!pid) std::cout<<tt_wait_sum[j*200+i]<<' ';
      if(!pid) std::cout<<";\n";
      MPI_Barrier(c);
    }
    if(!pid) std::cout<<"];\n\n\n";
  }

  MPI_Barrier(c);

  if(np==MAX_PROCS){
    if(!pid) std::cout<<"Tc=[";
    size_t j=0;
    for(size_t i=0;i<200;i++){
      for(j=0;j<MAX_PROCS;j++)
        if(!pid) std::cout<<tt_sum[j*200+i]<<' ';
      if(!pid) std::cout<<";\n";
      MPI_Barrier(c);
    }
    if(!pid) std::cout<<"];\n\n\n";
  }
  // */
  //Free memory.
  if(sbuff   !=NULL) delete[] sbuff;
#endif

      PROF_PAR_ALL2ALLV_DENSE_END
    }
/*/
  template <typename T>
    int Mpi_Alltoallv_dense(T* sendbuf, int* sendcnts, int* sdispls, 
        T* recvbuf, int* recvcnts, int* rdispls, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_ALL2ALLV_DENSE_BEGIN

#ifndef ALLTOALLV_FIX
      Mpi_Alltoallv
        (sendbuf, sendcnts, sdispls, 
         recvbuf, recvcnts, rdispls, comm);
#else
      int npes, rank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      //Processors may send a lot of information to themselves and a lesser
      //amount to others. If so, we don't want to waste communication by
      //including the local copy size in the max message size. 
      int maxNumElemSend = 0;
      for(int i = 0; i < rank; i++) {
        if(sendcnts[i] > maxNumElemSend) {
          maxNumElemSend = sendcnts[i];
        }
      }

      for(int i = (rank + 1); i < npes; i++) {
        if(sendcnts[i] > maxNumElemSend) {
          maxNumElemSend = sendcnts[i];
        }
      }

      int allToAllCount;
      par::Mpi_Allreduce<int>(&maxNumElemSend, &allToAllCount, 1, MPI_MAX, comm);

      T* tmpSendBuf = new T[allToAllCount*npes];
      assert(tmpSendBuf);

      T* tmpRecvBuf = new T[allToAllCount*npes];
      assert(tmpRecvBuf);

      for(int i = 0; i < rank; i++) {
        for(int j = 0; j < sendcnts[i]; j++) {
          tmpSendBuf[(allToAllCount*i) + j] = sendbuf[sdispls[i] + j];            
        }
      }

      for(int i = (rank + 1); i < npes; i++) {
        for(int j = 0; j < sendcnts[i]; j++) {
          tmpSendBuf[(allToAllCount*i) + j] = sendbuf[sdispls[i] + j];            
        }
      }

      par::Mpi_Alltoall<T>(tmpSendBuf, tmpRecvBuf, allToAllCount, comm);

      for(int i = 0; i < rank; i++) {
        for(int j = 0; j < recvcnts[i]; j++) {
          recvbuf[rdispls[i] + j] = tmpRecvBuf[(allToAllCount*i) + j];      
        }
      }

      //Now copy local portion.
#ifdef __DEBUG_PAR__
      assert(sendcnts[rank] == recvcnts[rank]);
#endif

      for(int j = 0; j < recvcnts[rank]; j++) {
        recvbuf[rdispls[rank] + j] = sendbuf[sdispls[rank] + j];      
      }

      for(int i = (rank + 1); i < npes; i++) {
        for(int j = 0; j < recvcnts[i]; j++) {
          recvbuf[rdispls[i] + j] = tmpRecvBuf[(allToAllCount*i) + j];      
        }
      }

      delete [] tmpSendBuf;
      delete [] tmpRecvBuf;
#endif

      PROF_PAR_ALL2ALLV_DENSE_END
    }
// */

  template<typename T>
    unsigned int defaultWeight(const T *a){
      return 1;
    }

  template <typename T> 
    int scatterValues(std::vector<T> & in, std::vector<T> & out, 
        DendroIntL outSz, MPI_Comm comm ) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_SCATTER_BEGIN

        int rank, npes;

      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      MPI_Request request;
      MPI_Status status;

      DendroIntL inSz = in.size();
      out.resize(outSz);

      DendroIntL off1 = 0, off2 = 0;
      DendroIntL * scnIn = NULL;
      if(inSz) {  
        scnIn = new DendroIntL [inSz]; 
        assert(scnIn);
      }

      // perform a local scan first ...
      DendroIntL zero = 0;
      if(inSz) {
        scnIn[0] = 1;
        for (DendroIntL i = 1; i < inSz; i++) {
          scnIn[i] = scnIn[i-1] + 1;
        }//end for
        // now scan with the final members of 
        par::Mpi_Scan<DendroIntL>(scnIn+inSz-1, &off1, 1, MPI_SUM, comm ); 
      } else{
        par::Mpi_Scan<DendroIntL>(&zero, &off1, 1, MPI_SUM, comm ); 
      }

      // communicate the offsets ...
      if (rank < (npes-1)){
        par::Mpi_Issend<DendroIntL>( &off1, 1, (rank + 1), 0, comm, &request );
      }
      if (rank){
        par::Mpi_Recv<DendroIntL>( &off2, 1, (rank - 1), 0, comm, &status );
      } else{
        off2 = 0; 
      }

      // add offset to local array
      for (DendroIntL i = 0; i < inSz; i++) {
        scnIn[i] = scnIn[i] + off2;  // This has the global scan results now ...
      }//end for

      //Gather Scan of outCnts
      DendroIntL *outCnts;
      outCnts = new DendroIntL[npes];
      assert(outCnts);

      if(rank < (npes-1)) {
        MPI_Status statusWait;
        MPI_Wait(&request,&statusWait);
      }

      if( outSz ) {
        par::Mpi_Scan<DendroIntL>( &outSz, &off1, 1, MPI_SUM, comm ); 
      }else {
        par::Mpi_Scan<DendroIntL>( &zero, &off1, 1, MPI_SUM, comm ); 
      }

      par::Mpi_Allgather<DendroIntL>( &off1, outCnts, 1, comm);

      int * sendSz = new int [npes];
      assert(sendSz);

      int * recvSz = new int [npes];
      assert(recvSz);

      int * sendOff = new int [npes];
      assert(sendOff);

      int * recvOff = new int [npes];
      assert(recvOff);

      // compute the partition offsets and sizes so that All2Allv can be performed.
      // initialize ...
      for (int i = 0; i < npes; i++) {
        sendSz[i] = 0;
      }

      //The Heart of the algorithm....
      //scnIn and outCnts are both sorted 
      DendroIntL inCnt = 0;
      int pCnt = 0;
      while( (inCnt < inSz) && (pCnt < npes) ) {
        if( scnIn[inCnt] <= outCnts[pCnt]  ) {
          sendSz[pCnt]++;
          inCnt++;
        }else {
          pCnt++;
        }
      }

      // communicate with other procs how many you shall be sending and get how
      // many to recieve from whom.
      par::Mpi_Alltoall<int>(sendSz, recvSz, 1, comm);

      int nn=0; // new value of nlSize, ie the local nodes.
      for (int i=0; i<npes; i++) {
        nn += recvSz[i];
      }

      // compute offsets ...
      sendOff[0] = 0;
      recvOff[0] = 0;
      for (int i=1; i<npes; i++) {
        sendOff[i] = sendOff[i-1] + sendSz[i-1];
        recvOff[i] = recvOff[i-1] + recvSz[i-1];
      }

      assert(static_cast<unsigned int>(nn) == outSz);
      // perform All2All  ... 
      T* inPtr = NULL;
      T* outPtr = NULL;
      if(!in.empty()) {
        inPtr = &(*(in.begin()));
      }
      if(!out.empty()) {
        outPtr = &(*(out.begin()));
      }
      par::Mpi_Alltoallv_sparse<T>(inPtr, sendSz, sendOff, 
          outPtr, recvSz, recvOff, comm);

      // clean up...
      if(scnIn) {
        delete [] scnIn;
        scnIn = NULL;
      }

      delete [] outCnts;
      outCnts = NULL;

      delete [] sendSz;
      sendSz = NULL;

      delete [] sendOff;
      sendOff = NULL;

      delete [] recvSz;
      recvSz = NULL;

      delete [] recvOff;
      recvOff = NULL;

      PROF_PAR_SCATTER_END
    }


  template<typename T>
    int concatenate(std::vector<T> & listA, std::vector<T> & listB,
        MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PAR_CONCAT_BEGIN

        int rank;
      int npes;

      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &npes);

      assert(!(listA.empty()));

      //1. First perform Allreduce to get total listA size
      //and total listB size; 

      DendroIntL locAsz_locBsz[2];
      DendroIntL globAsz_globBsz[2];

      locAsz_locBsz[0] = listA.size();
      locAsz_locBsz[1] = listB.size();
      globAsz_globBsz[0] = 0;
      globAsz_globBsz[1] = 0;

      par::Mpi_Allreduce<DendroIntL>(locAsz_locBsz, globAsz_globBsz, 2, MPI_SUM, comm);

      //2. Re-distribute A and B independently so that
      //B is distributed only on the high rank processors
      //and A is distribute only on the low rank processors.

      DendroIntL avgTotalSize = ((globAsz_globBsz[0] + globAsz_globBsz[1])/npes);

      //since listA is not empty on any of the active procs,
      //globASz > npes so avgTotalSize >= 1

      DendroIntL remTotalSize = ((globAsz_globBsz[0] + globAsz_globBsz[1])%npes);

      int numSmallProcs = (npes - remTotalSize);

      //In the final merged list, there will be exactly remTotalSize number
      //of processors each having (avgTotalSize + 1) elements and there will
      //be exactly numSmallProcs number of processors each having
      //avgTotalSize elements. 
      //Also, len(A) + len(B) = (numSmallProcs*avg) + (remTotalSize*(avg+1))

      std::vector<T> tmpA;
      std::vector<T> tmpB;

      int numAhighProcs;
      int numAlowProcs;
      int numBothProcs;
      int numBhighProcs;
      int numBlowProcs;
      DendroIntL aSizeForBoth;
      DendroIntL bSizeForBoth;

      if( globAsz_globBsz[1] <= (numSmallProcs*avgTotalSize) ) {
        numBhighProcs = 0;
        numBlowProcs = ((globAsz_globBsz[1])/avgTotalSize); 
        bSizeForBoth = ((globAsz_globBsz[1])%avgTotalSize);

        assert(numBlowProcs <= numSmallProcs);

        //remBsize is < avgTotalSize. So it will fit on one proc.
        if(bSizeForBoth) {
          numBothProcs = 1;
          if(numBlowProcs < numSmallProcs) {
            //We don't know if remTotalSize is 0 or not. 
            //So, let the common proc be a low proc.
            aSizeForBoth = (avgTotalSize - bSizeForBoth);
            numAhighProcs = remTotalSize;
            numAlowProcs = (numSmallProcs - (1 + numBlowProcs));
          } else {             
            //No more room for small procs. The common has to be a high proc.
            aSizeForBoth = ((avgTotalSize + 1) - bSizeForBoth);
            numAhighProcs = (remTotalSize - 1);
            numAlowProcs = 0;
          }
        } else {
          numBothProcs = 0;
          aSizeForBoth = 0;
          numAhighProcs = remTotalSize;
          numAlowProcs = (numSmallProcs - numBlowProcs);
        }
      } else {
        //Some B procs will have (avgTotalSize+1) elements
        DendroIntL numBusingAvgPlus1 = ((globAsz_globBsz[1])/(avgTotalSize + 1));
        DendroIntL remBusingAvgPlus1 = ((globAsz_globBsz[1])%(avgTotalSize + 1));
        if (numBusingAvgPlus1 <= remTotalSize) {
          //Each block can use (avg+1) elements each, since there will be some
          //remaining for A  
          numBhighProcs = numBusingAvgPlus1;
          numBlowProcs = 0;
          bSizeForBoth = remBusingAvgPlus1;
          if(bSizeForBoth) {
            numBothProcs = 1;
            if (numBhighProcs < remTotalSize) {
              //We don't know if numSmallProcs is 0 or not.
              //So, let the common proc be a high proc 
              aSizeForBoth = ((avgTotalSize + 1) - bSizeForBoth);
              numAhighProcs = (remTotalSize - (numBhighProcs + 1));
              numAlowProcs = numSmallProcs;
            } else {
              //No more room for high procs. The common has to be a low proc. 
              aSizeForBoth = (avgTotalSize - bSizeForBoth);
              numAhighProcs = 0;
              numAlowProcs = (numSmallProcs - 1);
            }
          } else {
            numBothProcs = 0;
            aSizeForBoth = 0;
            numAhighProcs = (remTotalSize - numBhighProcs);
            numAlowProcs = numSmallProcs;
          }
        } else {
          //Since numBusingAvgPlus1 > remTotalSize*(avg+1) 
          //=> len(B) > remTotalSize*(avg+1)
          //=> len(A) < numSmallProcs*avg
          //This is identical to the first case (except for 
          //the equality), with A and B swapped.

          assert( globAsz_globBsz[0] < (numSmallProcs*avgTotalSize) );

          numAhighProcs = 0;
          numAlowProcs = ((globAsz_globBsz[0])/avgTotalSize); 
          aSizeForBoth = ((globAsz_globBsz[0])%avgTotalSize);

          assert(numAlowProcs < numSmallProcs);

          //remAsize is < avgTotalSize. So it will fit on one proc.
          if(aSizeForBoth) {
            numBothProcs = 1;
            //We don't know if remTotalSize is 0 or not. 
            //So, let the common proc be a low proc.
            bSizeForBoth = (avgTotalSize - aSizeForBoth);
            numBhighProcs = remTotalSize;
            numBlowProcs = (numSmallProcs - (1 + numAlowProcs));
          } else {
            numBothProcs = 0;
            bSizeForBoth = 0;
            numBhighProcs = remTotalSize;
            numBlowProcs = (numSmallProcs - numAlowProcs);
          }
        }
      }

      assert((numAhighProcs + numAlowProcs + numBothProcs
            + numBhighProcs + numBlowProcs) == npes);

      assert((aSizeForBoth + bSizeForBoth) <= (avgTotalSize+1));

      if(numBothProcs) {
        assert((aSizeForBoth + bSizeForBoth) >= avgTotalSize);
      } else {
        assert(aSizeForBoth == 0); 
        assert(bSizeForBoth == 0); 
      }

      if((aSizeForBoth + bSizeForBoth) == (avgTotalSize + 1)) {
        assert((numAhighProcs + numBothProcs + numBhighProcs) == remTotalSize);
        assert((numAlowProcs + numBlowProcs) == numSmallProcs);
      } else {
        assert((numAhighProcs + numBhighProcs) == remTotalSize);
        assert((numAlowProcs + numBothProcs + numBlowProcs) == numSmallProcs);
      }

      //The partition is as follow:
      //1. numAhighProcs with (avg+1) elements each exclusively from A,
      //2. numAlowProcs with avg elements each exclusively from A
      //3. numBothProcs with aSizeForBoth elements from A and
      // bSizeForBoth elements from B
      //4. numBhighProcs with (avg+1) elements each exclusively from B.
      //5. numBlowProcs with avg elements each exclusively from B.

      if(rank < numAhighProcs) {
        par::scatterValues<T>(listA, tmpA, (avgTotalSize + 1), comm);
        par::scatterValues<T>(listB, tmpB, 0, comm);
      } else if (rank < (numAhighProcs + numAlowProcs)) {
        par::scatterValues<T>(listA, tmpA, avgTotalSize, comm);
        par::scatterValues<T>(listB, tmpB, 0, comm);
      } else if (rank < (numAhighProcs + numAlowProcs + numBothProcs)) {
        par::scatterValues<T>(listA, tmpA, aSizeForBoth, comm);
        par::scatterValues<T>(listB, tmpB, bSizeForBoth, comm);
      } else if (rank <
          (numAhighProcs + numAlowProcs + numBothProcs + numBhighProcs)) {
        par::scatterValues<T>(listA, tmpA, 0, comm);
        par::scatterValues<T>(listB, tmpB, (avgTotalSize + 1), comm);
      } else {
        par::scatterValues<T>(listA, tmpA, 0, comm);
        par::scatterValues<T>(listB, tmpB, avgTotalSize, comm);
      }

      listA = tmpA;
      listB = tmpB;
      tmpA.clear();
      tmpB.clear();

      //3. Finally do a simple concatenation A = A + B. If the previous step
      //was performed correctly, there will be atmost 1 processor, which has both
      //non-empty A and non-empty B. On other processors one of the two lists
      //will be empty
      if(listA.empty()) {
        listA = listB;
      } else {
        if(!(listB.empty())) {
          listA.insert(listA.end(), listB.begin(), listB.end());
        }
      }

      listB.clear();

      PROF_PAR_CONCAT_END
    }

  template <typename T>
    int maxLowerBound(const std::vector<T> & keys, const std::vector<T> & searchList,
        std::vector<T> & results, MPI_Comm comm) {
      PROF_SEARCH_BEGIN

        int rank, npes;

      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      // allocate memory for the mins array
      std::vector<T> mins (npes);
      assert(!searchList.empty());

      T* searchListPtr = NULL;
      T* minsPtr = NULL;
      if(!searchList.empty()) {
        searchListPtr = &(*(searchList.begin()));
      }
      if(!mins.empty()) {
        minsPtr = &(*(mins.begin()));
      }
      par::Mpi_Allgather<T>(searchListPtr, minsPtr, 1, comm);

      //For each key decide which processor to send to
      unsigned int *part = NULL;

      if(keys.size()) {
        part = new unsigned int[keys.size()];
        assert(part);
      }

      for ( unsigned int i=0; i<keys.size(); i++ ) {
        //maxLB returns the smallest index in a sorted array such
        //that a[ind] <= key and  a[index +1] > key
        bool found = par::maxLowerBound<T>(mins,keys[i], part+i,NULL,NULL);
        if ( !found ) {
          //This key is smaller than the mins from every processor.
          //No point in searching.
          part[i] = rank;
        }
      }

      mins.clear();

      int *numKeysSend = new int[npes];
      assert(numKeysSend);

      int *numKeysRecv = new int[npes];
      assert(numKeysRecv);

      for ( int i=0; i<npes; i++ ) {
        numKeysSend[i] = 0;
      }

      // calculate the number of keys to send ...
      for ( unsigned int i=0; i<keys.size(); i++ ) {
        numKeysSend[part[i]]++;
      }

      // Now do an All2All to get numKeysRecv
      par::Mpi_Alltoall<int>(numKeysSend, numKeysRecv, 1, comm);

      unsigned int totalKeys=0;        // total number of local keys ...
      for ( int i=0; i<npes; i++ ) {
        totalKeys += numKeysRecv[i];
      }

      // create the send and recv buffers ...
      std::vector<T> sendK (keys.size());
      std::vector<T> recvK (totalKeys);

      // the mapping ..
      unsigned int * comm_map = NULL;

      if(keys.size()) {
        comm_map = new unsigned int [keys.size()];
        assert(comm_map);
      }

      // Now create sendK
      int *sendOffsets = new int[npes]; 
      assert(sendOffsets);
      sendOffsets[0] = 0;

      int *recvOffsets = new int[npes]; 
      assert(recvOffsets);
      recvOffsets[0] = 0;

      int *numKeysTmp = new int[npes]; 
      assert(numKeysTmp);
      numKeysTmp[0] = 0; 

      // compute offsets ...
      for ( int i=1; i<npes; i++ ) {
        sendOffsets[i] = sendOffsets[i-1] + numKeysSend[i-1];
        recvOffsets[i] = recvOffsets[i-1] + numKeysRecv[i-1];
        numKeysTmp[i] = 0; 
      }

      for ( unsigned int i=0; i< keys.size(); i++ ) {
        unsigned int ni = numKeysTmp[part[i]];
        numKeysTmp[part[i]]++;
        // set entry ...
        sendK[sendOffsets[part[i]] + ni] = keys[i];
        // save mapping .. will need it later ...
        comm_map[i] = sendOffsets[part[i]] + ni;
      }

      if(part) {
        delete [] part;
      }

      assert(numKeysTmp);
      delete [] numKeysTmp;
      numKeysTmp = NULL;

      T* sendKptr = NULL;
      T* recvKptr = NULL;
      if(!sendK.empty()) {
        sendKptr = &(*(sendK.begin()));
      }
      if(!recvK.empty()) {
        recvKptr = &(*(recvK.begin()));
      }

      par::Mpi_Alltoallv_sparse<T>(sendKptr, numKeysSend, sendOffsets, 
          recvKptr, numKeysRecv, recvOffsets, comm);


      std::vector<T>  resSend (totalKeys);
      std::vector<T>  resRecv (keys.size());

      //Final local search.
      for ( unsigned int i = 0; i < totalKeys; i++) {
        unsigned int idx;
        bool found = par::maxLowerBound<T>( searchList, recvK[i], &idx,NULL,NULL );
        if(found) {
          resSend[i] = searchList[idx];
        }
      }//end for i

      //Exchange Results
      //Return what you received in the earlier communication.
      T* resSendPtr = NULL;
      T* resRecvPtr = NULL;
      if(!resSend.empty()) {
        resSendPtr = &(*(resSend.begin()));
      }
      if(!resRecv.empty()) {
        resRecvPtr = &(*(resRecv.begin()));
      }
      par::Mpi_Alltoallv_sparse<T>(resSendPtr, numKeysRecv, recvOffsets, 
          resRecvPtr, numKeysSend, sendOffsets, comm);

      assert(sendOffsets);
      delete [] sendOffsets;
      sendOffsets = NULL;

      assert(recvOffsets);
      delete [] recvOffsets;
      recvOffsets = NULL;

      assert(numKeysSend);
      delete [] numKeysSend;
      numKeysSend = NULL;

      assert(numKeysRecv);
      delete [] numKeysRecv;
      numKeysRecv = NULL;

      for ( unsigned int i=0; i < keys.size(); i++ ) {
        results[i] = resRecv[comm_map[i]];  
      }//end for

      // Clean up ...
      if(comm_map) {
        delete [] comm_map;
      }

      PROF_SEARCH_END
    }

  template<typename T>
    int partitionW(std::vector<T>& nodeList, unsigned int (*getWeight)(const T *), MPI_Comm comm){
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_PARTW_BEGIN

      int npes;

      MPI_Comm_size(comm, &npes);

      if(npes == 1) {
        PROF_PARTW_END
      }

      if(getWeight == NULL) {
        getWeight = par::defaultWeight<T>;
      }

      int rank;

      MPI_Comm_rank(comm, &rank);

      MPI_Request request;
      MPI_Status status;
      const bool nEmpty = nodeList.empty();

      DendroIntL  off1= 0, off2= 0, localWt= 0, totalWt = 0;

      DendroIntL* wts = NULL;
      DendroIntL* lscn = NULL;
      DendroIntL nlSize = nodeList.size();
      if(nlSize) {
        wts = new DendroIntL[nlSize];
        assert(wts);

        lscn= new DendroIntL[nlSize]; 
        assert(lscn);
      }

      // First construct arrays of id and wts.
      #pragma omp parallel for reduction(+:localWt)
      for (DendroIntL i = 0; i < nlSize; i++){
        wts[i] = (*getWeight)( &(nodeList[i]) );
        localWt+=wts[i];
      }

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-1 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      // compute the total weight of the problem ...
      par::Mpi_Allreduce<DendroIntL>(&localWt, &totalWt, 1, MPI_SUM, comm);

      // perform a local scan on the weights first ...
      DendroIntL zero = 0;
      if(!nEmpty) {
        lscn[0]=wts[0];
//        for (DendroIntL i = 1; i < nlSize; i++) {
//          lscn[i] = wts[i] + lscn[i-1];
//        }//end for
        omp_par::scan(&wts[1],lscn,nlSize);
        // now scan with the final members of 
        par::Mpi_Scan<DendroIntL>(lscn+nlSize-1, &off1, 1, MPI_SUM, comm ); 
      } else{
        par::Mpi_Scan<DendroIntL>(&zero, &off1, 1, MPI_SUM, comm ); 
      }

      // communicate the offsets ...
      if (rank < (npes-1)){
        par::Mpi_Issend<DendroIntL>( &off1, 1, rank+1, 0, comm, &request );
      }
      if (rank){
        par::Mpi_Recv<DendroIntL>( &off2, 1, rank-1, 0, comm, &status );
      }
      else{
        off2 = 0; 
      }

      // add offset to local array
      #pragma omp parallel for
      for (DendroIntL i = 0; i < nlSize; i++) {
        lscn[i] = lscn[i] + off2;       // This has the global scan results now ...
      }//end for

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-2 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      int * sendSz = new int [npes];
      assert(sendSz);

      int * recvSz = new int [npes];
      assert(recvSz);

      int * sendOff = new int [npes]; 
      assert(sendOff);
      sendOff[0] = 0;

      int * recvOff = new int [npes]; 
      assert(recvOff);
      recvOff[0] = 0;

      // compute the partition offsets and sizes so that All2Allv can be performed.
      // initialize ...

      #pragma omp parallel for
      for (int i = 0; i < npes; i++) {
        sendSz[i] = 0;
      }

      // Now determine the average load ...
      DendroIntL npesLong = npes;
      DendroIntL avgLoad = (totalWt/npesLong);

      DendroIntL extra = (totalWt%npesLong);

      //The Heart of the algorithm....
      if(avgLoad > 0) {
        for (DendroIntL i = 0; i < nlSize; i++) {
          if(lscn[i] == 0) {
            sendSz[0]++;
          }else {
            int ind=0;
            if ( lscn[i] <= (extra*(avgLoad + 1)) ) {
              ind = ((lscn[i] - 1)/(avgLoad + 1));
            }else {
              ind = ((lscn[i] - (1 + extra))/avgLoad);
            }
            assert(ind < npes);
            sendSz[ind]++;
          }//end if-else
        }//end for */ 
/*
        //This is more effecient and parallelizable than the above.
        //This has a bug trying a simpler approach below.
        int ind_min,ind_max;
        ind_min=(lscn[0]*npesLong)/totalWt-1;
        ind_max=(lscn[nlSize-1]*npesLong)/totalWt+2;
        if(ind_min< 0       )ind_min=0;
        if(ind_max>=npesLong)ind_max=npesLong;
        #pragma omp parallel for
        for(int i=ind_min;i<ind_max;i++){
          DendroIntL wt1=(totalWt*i)/npesLong;
          DendroIntL wt2=(totalWt*(i+1))/npesLong;
          int end = std::upper_bound(&lscn[0], &lscn[nlSize], wt2, std::less<DendroIntL>())-&lscn[0];
          int start = std::upper_bound(&lscn[0], &lscn[nlSize], wt1, std::less<DendroIntL>())-&lscn[0];
          if(i==npesLong-1)end  =nlSize;
          if(i==         0)start=0     ;
          sendSz[i]=end-start;
        }// */

#ifdef __DEBUG_PAR__
        int tmp_sum=0;
        for(int i=0;i<npes;i++) tmp_sum+=sendSz[i];
        assert(tmp_sum==nlSize);
#endif

      }else {
        sendSz[0]+= nlSize;
      }//end if-else

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-3 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      if(rank < (npes-1)) {
        MPI_Status statusWait;
        MPI_Wait(&request, &statusWait);
      }

      // communicate with other procs how many you shall be sending and get how
      // many to recieve from whom.
      par::Mpi_Alltoall<int>(sendSz, recvSz, 1, comm);

#ifdef __DEBUG_PAR__
      DendroIntL totSendToOthers = 0;
      DendroIntL totRecvFromOthers = 0;
      for (int i = 0; i < npes; i++) {
        if(rank != i) {
          totSendToOthers += sendSz[i];
          totRecvFromOthers += recvSz[i];
        }
      }
#endif

      DendroIntL nn=0; // new value of nlSize, ie the local nodes.
      #pragma omp parallel for reduction(+:nn)
      for (int i = 0; i < npes; i++) {
        nn += recvSz[i];
      }

      // compute offsets ...
//      for (int i = 1; i < npes; i++) {
//        sendOff[i] = sendOff[i-1] + sendSz[i-1];
//        recvOff[i] = recvOff[i-1] + recvSz[i-1];
//      }
      omp_par::scan(sendSz,sendOff,npes);
      omp_par::scan(recvSz,recvOff,npes);

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-4 passed."<<std::endl;
      }
      MPI_Barrier(comm);
      /*
         std::cout<<rank<<": newSize: "<<nn<<" oldSize: "<<(nodeList.size())
         <<" send: "<<totSendToOthers<<" recv: "<<totRecvFromOthers<<std::endl;
       */
      MPI_Barrier(comm);
#endif

      // allocate memory for the new arrays ...
      std::vector<T > newNodes(nn);

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Final alloc successful."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      // perform All2All  ... 
      T* nodeListPtr = NULL;
      T* newNodesPtr = NULL;
      if(!nodeList.empty()) {
        nodeListPtr = &(*(nodeList.begin()));
      }
      if(!newNodes.empty()) {
        newNodesPtr = &(*(newNodes.begin()));
      }
      par::Mpi_Alltoallv_sparse<T>(nodeListPtr, sendSz, sendOff, 
          newNodesPtr, recvSz, recvOff, comm);

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-5 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      // reset the pointer ...
      swap(nodeList, newNodes);
      newNodes.clear();

      // clean up...
      if(!nEmpty) {
        delete [] lscn;
        delete [] wts;
      }
      delete [] sendSz;
      sendSz = NULL;

      delete [] sendOff;
      sendOff = NULL;

      delete [] recvSz;
      recvSz = NULL;

      delete [] recvOff;
      recvOff = NULL;

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"Partition: Stage-6 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      PROF_PARTW_END
    }//end function

  template<typename T>
    int removeDuplicates(std::vector<T>& vecT, bool isSorted, MPI_Comm comm){
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_REMDUP_BEGIN
        int size, rank;
      MPI_Comm_size(comm,&size);
      MPI_Comm_rank(comm,&rank);

      std::vector<T> tmpVec;
      if(!isSorted) {                  
        //Sort partitions vecT and tmpVec internally.
        par::HyperQuickSort<T>(vecT, tmpVec, comm);                                    
      }else {
        swap(tmpVec, vecT);
      }

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"RemDup: Stage-1 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      vecT.clear();
      par::partitionW<T>(tmpVec, NULL, comm);

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"RemDup: Stage-2 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      //Remove duplicates locally
      seq::makeVectorUnique<T>(tmpVec,true); 

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"RemDup: Stage-3 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      //Creating groups

      int new_rank, new_size; 
      MPI_Comm   new_comm;
      // very quick and dirty solution -- assert that tmpVec is non-emply at every processor (repetetive calls to splitComm2way exhaust MPI resources)
      // par::splitComm2way(tmpVec.empty(), &new_comm, comm);
      new_comm=comm;
      assert(!tmpVec.empty());
      MPI_Comm_rank (new_comm, &new_rank);
      MPI_Comm_size (new_comm, &new_size);

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"RemDup: Stage-4 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      //Checking boundaries... 
      if(!tmpVec.empty()) {
        T end = tmpVec[tmpVec.size()-1];          
        T endRecv;

        //communicate end to the next processor.
        MPI_Status status;

        par::Mpi_Sendrecv<T, T>(&end, 1, ((new_rank <(new_size-1))?(new_rank+1):0), 1, &endRecv,
            1, ((new_rank > 0)?(new_rank-1):(new_size-1)), 1, new_comm, &status);

        //Remove endRecv if it exists (There can be no more than one copy of this)
        if(new_rank) {
          typename std::vector<T>::iterator Iter = find(tmpVec.begin(),tmpVec.end(),endRecv);
          if(Iter != tmpVec.end()) {
            tmpVec.erase(Iter);
          }//end if found    
        }//end if p not 0          
      }//end if not empty

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"RemDup: Stage-5 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      swap(vecT, tmpVec);
      tmpVec.clear();
      par::partitionW<T>(vecT, NULL, comm);

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!rank) {
        std::cout<<"RemDup: Stage-6 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      PROF_REMDUP_END
    }//end function
/*
  template<typename T>
    int HyperQuickSort(std::vector<T>& arr, std::vector<T> & SortedElem, MPI_Comm comm_){ // O( ((N/p)+log(p))*(log(N/p)+log(p)) ) 
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_SORT_BEGIN

      std::vector<double> tt(100,0);
      MPI_Barrier(comm_); tt[0]-=omp_get_wtime(); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TIC
      double ttt=-omp_get_wtime(); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TIC

      // Copy communicator.
      MPI_Comm comm=comm_;

      // Get comm size and rank.
      int npes, npes_, myrank, myrank_;
      MPI_Comm_size(comm, &npes); npes_=npes;
      MPI_Comm_rank(comm, &myrank); myrank_=myrank;
      int omp_p=omp_get_max_threads();
      srand(myrank);

      // Local and global sizes. O(log p)
      DendroIntL totSize, nelem = arr.size(); assert(nelem);
      par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
      DendroIntL nelem_ = nelem;

      tt[0]+=omp_get_wtime(); //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TOC

      // Local sort.
      MPI_Barrier(comm_); tt[1]-=omp_get_wtime(); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TIC
      T* arr_=new T[nelem]; memcpy (&arr_[0], &arr[0], nelem*sizeof(T));
      omp_par::merge_sort(&arr_[0], &arr_[arr.size()]);
      tt[1]+=omp_get_wtime(); //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TOC

      // Binary split and merge in each iteration.
      while(npes>1 && totSize>0){ // O(log p) iterations.

        //Determine splitters. O( log(N/p) + log(p) )
        MPI_Barrier(comm); tt[2]-=omp_get_wtime(); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TIC
        T split_key;
        DendroIntL totSize_new;
        //while(true)
        { 
          // Take random splitters. O( 1 ) -- Let p * splt_count = glb_splt_count = const = 100~1000
          int splt_count=(1000*nelem)/totSize; 
          if(npes>1000) splt_count=(((float)rand()/(float)RAND_MAX)*totSize<(1000*nelem)?1:0);
          if(splt_count>nelem) splt_count=nelem;
          std::vector<T> splitters(splt_count);
          for(size_t i=0;i<splt_count;i++) 
            splitters[i]=arr_[rand()%nelem];

          // Gather all splitters. O( log(p) )
          int glb_splt_count;
          std::vector<int> glb_splt_cnts(npes);
          std::vector<int> glb_splt_disp(npes,0);
          par::Mpi_Allgather<int>(&splt_count, &glb_splt_cnts[0], 1, comm);
          omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
          glb_splt_count=glb_splt_cnts[npes-1]+glb_splt_disp[npes-1];
          std::vector<T> glb_splitters(glb_splt_count);
          MPI_Allgatherv(&    splitters[0], splt_count, par::Mpi_datatype<T>::value(), 
                         &glb_splitters[0], &glb_splt_cnts[0], &glb_splt_disp[0], 
                         par::Mpi_datatype<T>::value(), comm);

          // Determine split key. O( log(N/p) + log(p) )
          std::vector<DendroIntL> disp(glb_splt_count,0);
          if(nelem>0){
            #pragma omp parallel for
            for(size_t i=0;i<glb_splt_count;i++){
              disp[i]=std::lower_bound(&arr_[0], &arr_[nelem], glb_splitters[i])-&arr_[0];
            }
          }
          std::vector<DendroIntL> glb_disp(glb_splt_count,0);
          MPI_Allreduce(&disp[0], &glb_disp[0], glb_splt_count, par::Mpi_datatype<DendroIntL>::value(), MPI_SUM, comm);

          DendroIntL* split_disp=&glb_disp[0];
          for(size_t i=0;i<glb_splt_count;i++)
            if(abs(glb_disp[i]-totSize/2)<abs(*split_disp-totSize/2)) split_disp=&glb_disp[i];
          split_key=glb_splitters[split_disp-&glb_disp[0]];

          totSize_new=(myrank<=(npes-1)/2?*split_disp:totSize-*split_disp);
          //double err=(((double)*split_disp)/(totSize/2))-1.0;
          //if(fabs(err)<0.01 || npes<=16) break;
          //else if(!myrank) std::cout<<err<<'\n';
        }
        tt[2]+=omp_get_wtime(); //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TOC

        // Split problem into two. O( N/p )
        int split_id=(npes-1)/2;
        {
          MPI_Barrier(comm); tt[3]-=omp_get_wtime(); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TIC
          int new_p0=(myrank<=split_id?0:split_id+1);
          int cmp_p0=(myrank> split_id?0:split_id+1);
          int new_np=(myrank<=split_id? split_id+1: npes-split_id-1);
          int cmp_np=(myrank> split_id? split_id+1: npes-split_id-1);

          int partner = myrank+cmp_p0-new_p0;
          if(partner>=npes) partner=npes-1;
          assert(partner>=0);

          bool extra_partner=( npes%2==1  && npes-1==myrank );

          // Exchange send sizes.
          char *sbuff, *lbuff;
          int     rsize=0,     ssize=0, lsize=0;
          int ext_rsize=0, ext_ssize=0;
          size_t split_indx=(nelem>0?std::lower_bound(&arr_[0], &arr_[nelem], split_key)-&arr_[0]:0);
          ssize=       (myrank> split_id? split_indx: nelem-split_indx )*sizeof(T);
          sbuff=(char*)(myrank> split_id? &arr_[0]   :  &arr_[split_indx]);
          lsize=       (myrank<=split_id? split_indx: nelem-split_indx )*sizeof(T);
          lbuff=(char*)(myrank<=split_id? &arr_[0]   :  &arr_[split_indx]);

          MPI_Status status;
          MPI_Sendrecv                  (&    ssize,1,MPI_INT, partner,0,   &    rsize,1,MPI_INT, partner,   0,comm,&status);
          if(extra_partner) MPI_Sendrecv(&ext_ssize,1,MPI_INT,split_id,0,   &ext_rsize,1,MPI_INT,split_id,   0,comm,&status);

          // Exchange data.
          char*     rbuff=              new char[    rsize]       ;
          char* ext_rbuff=(ext_rsize>0? new char[ext_rsize]: NULL);
          MPI_Sendrecv                  (sbuff,ssize,MPI_BYTE, partner,0,       rbuff,    rsize,MPI_BYTE, partner,   0,comm,&status);
          if(extra_partner) MPI_Sendrecv( NULL,    0,MPI_BYTE,split_id,0,   ext_rbuff,ext_rsize,MPI_BYTE,split_id,   0,comm,&status);
          tt[3]+=omp_get_wtime(); //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TOC

          MPI_Barrier(comm); tt[4]-=omp_get_wtime(); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TIC
          int nbuff_size=lsize+rsize+ext_rsize;
          char* nbuff= new char[nbuff_size];
          omp_par::merge<T*>((T*)lbuff, (T*)&lbuff[lsize], (T*)rbuff, (T*)&rbuff[rsize], (T*)nbuff, omp_p, std::less<T>());
          if(ext_rsize>0 && nbuff!=NULL){
            char* nbuff1= new char[nbuff_size];
            omp_par::merge<T*>((T*)nbuff, (T*)&nbuff[lsize+rsize], (T*)ext_rbuff, (T*)&ext_rbuff[ext_rsize], (T*)nbuff1, omp_p, std::less<T>());
            if(nbuff!=NULL) delete[] nbuff; nbuff=nbuff1;
          }

          // Copy new data.
          totSize=totSize_new;
          nelem = nbuff_size/sizeof(T);
          if(arr_!=NULL) delete[] arr_; 
          arr_=(T*) nbuff; nbuff=NULL;

          //Free memory.
          if(    rbuff!=NULL) delete[]     rbuff;
          if(ext_rbuff!=NULL) delete[] ext_rbuff;
          tt[4]+=omp_get_wtime(); //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TOC
        }

        {// Split comm.  O( log(p) ) ??
          MPI_Barrier(comm); tt[5]-=omp_get_wtime(); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TIC
          MPI_Comm scomm;
          MPI_Comm_split(comm, myrank<=split_id, myrank, &scomm );
          comm=scomm;
          npes  =(myrank<=split_id? split_id+1: npes  -split_id-1);
          myrank=(myrank<=split_id? myrank    : myrank-split_id-1);
          tt[5]+=omp_get_wtime(); //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TOC
        }
      }

      MPI_Barrier(comm); tt[6]-=omp_get_wtime(); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< TIC
      SortedElem.assign(arr_, &arr_[nelem]);
      SortedElem.resize(nelem);

      par::partitionW<T>(SortedElem, NULL , comm_);
      tt[6]+=omp_get_wtime(); //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TOC
      ttt+=omp_get_wtime(); //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TOC

      // Print timing.
      std::vector<double> tt_sum(100,0);
      std::vector<double> tt_max(100,0);
      MPI_Reduce(&tt[0], &tt_sum[0], 100, MPI_DOUBLE, MPI_SUM, 0, comm_);
      MPI_Reduce(&tt[0], &tt_max[0], 100, MPI_DOUBLE, MPI_MAX, 0, comm_);
      if(!myrank_) for(int i=0;i<7;i++) std::cout<<tt_sum[i]/npes_<<' '<<tt_max[i]<<'\n';
      if(!myrank_) std::cout<<ttt<<'\n';

      PROF_SORT_END
    }//end function
*/

  /* Hari Notes ....
   *
   * Avoid unwanted allocations within Hypersort ...
   * 
   * 1. try to sort in place ... no output buffer, user can create a copy if
   *    needed.
   * 2. have a std::vector<T> container for rbuff. the space required can be 
   *    reserved before doing MPI_SendRecv
   * 3. alternatively, keep a send buffer and recv into original buffer. 
   *
   */ 
  template<typename T>
    int HyperQuickSort(std::vector<T>& arr, MPI_Comm comm_){ // O( ((N/p)+log(p))*(log(N/p)+log(p)) ) 
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_SORT_BEGIN

      // Copy communicator. 
      MPI_Comm comm=comm_;

      // Get comm size and rank.
      int npes, myrank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &myrank);
      
      if(npes==1){
        omp_par::merge_sort(&arr[0],&arr[arr.size()]);
        // SortedElem  = arr;
        PROF_SORT_END
      }
      // buffers ... keeping all allocations together 
      std::vector<T>  commBuff;
      std::vector<T>  mergeBuff;
      std::vector<int> glb_splt_cnts(npes);
      std::vector<int> glb_splt_disp(npes,0);


      int omp_p=omp_get_max_threads();
      srand(myrank);

      // Local and global sizes. O(log p)
      DendroIntL totSize, nelem = arr.size(); assert(nelem);
      par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
      DendroIntL nelem_ = nelem;

      // Local sort.
      omp_par::merge_sort(&arr[0], &arr[arr.size()]);

      // Binary split and merge in each iteration.
      while(npes>1 && totSize>0){ // O(log p) iterations.

        //Determine splitters. O( log(N/p) + log(p) )
        T split_key;
        DendroIntL totSize_new;
        //while(true)
        { 
          // Take random splitters. O( 1 ) -- Let p * splt_count = glb_splt_count = const = 100~1000
          int splt_count = (1000*nelem)/totSize; 
          if (npes>1000) 
            splt_count = ( ((float)rand()/(float)RAND_MAX)*totSize < (1000*nelem) ? 1 : 0 );
          
          if ( splt_count > nelem ) 
						splt_count = nelem;
          
          std::vector<T> splitters(splt_count);
          for(size_t i=0;i<splt_count;i++) 
            splitters[i]=arr_[rand()%nelem];
					/* Fisher-Yates shuffle
          unsigned int j;
          for (size_t i=0; i<splt_count; i++) 
            splitters[i]=arr[i];
          for (size_t i=splt_count; i<arr.size(); i++) {
            j = binOp::reversibleHash(i)%(i+1);
            if ( j < splt_count )
              splitters[j]=arr[i];
          } */

          // Gather all splitters. O( log(p) )
          int glb_splt_count;

          par::Mpi_Allgather<int>(&splt_count, &glb_splt_cnts[0], 1, comm);
          omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
         
          glb_splt_count = glb_splt_cnts[npes-1] + glb_splt_disp[npes-1];

          std::vector<T> glb_splitters(glb_splt_count);
          
          MPI_Allgatherv(&splitters[0], splt_count, par::Mpi_datatype<T>::value(), 
                         &glb_splitters[0], &glb_splt_cnts[0], &glb_splt_disp[0], 
                         par::Mpi_datatype<T>::value(), comm);

          // Determine split key. O( log(N/p) + log(p) )
          std::vector<DendroIntL> disp(glb_splt_count,0);
          
          if(nelem>0){
            #pragma omp parallel for
            for(size_t i=0;i<glb_splt_count;i++){
              disp[i]=std::lower_bound(&arr[0], &arr[nelem], glb_splitters[i]) - &arr[0];
            }
          }
          std::vector<DendroIntL> glb_disp(glb_splt_count,0);
          MPI_Allreduce(&disp[0], &glb_disp[0], glb_splt_count, par::Mpi_datatype<DendroIntL>::value(), MPI_SUM, comm);

          DendroIntL* split_disp = &glb_disp[0];
          for(size_t i=0; i<glb_splt_count; i++)
            if ( abs(glb_disp[i] - totSize/2) < abs(*split_disp - totSize/2) ) 
							split_disp = &glb_disp[i];
          split_key = glb_splitters[split_disp - &glb_disp[0]];

          totSize_new=(myrank<=(npes-1)/2?*split_disp:totSize-*split_disp);
          //double err=(((double)*split_disp)/(totSize/2))-1.0;
          //if(fabs(err)<0.01 || npes<=16) break;
          //else if(!myrank) std::cout<<err<<'\n';
        }

        // Split problem into two. O( N/p )
        int split_id=(npes-1)/2;
        {
          int new_p0 = (myrank<=split_id?0:split_id+1);
          int cmp_p0 = (myrank> split_id?0:split_id+1);
          int new_np = (myrank<=split_id? split_id+1: npes-split_id-1);
          int cmp_np = (myrank> split_id? split_id+1: npes-split_id-1);

          int partner = myrank+cmp_p0-new_p0;
          if(partner>=npes) partner=npes-1;
          assert(partner>=0);

          bool extra_partner=( npes%2==1  && npes-1==myrank );

          // Exchange send sizes.
          char *sbuff, *lbuff;

          int     rsize=0,     ssize=0, lsize=0;
          int ext_rsize=0, ext_ssize=0;
          size_t split_indx=(nelem>0?std::lower_bound(&arr[0], &arr[nelem], split_key)-&arr[0]:0);
          ssize=       (myrank> split_id? split_indx: nelem-split_indx )*sizeof(T);
          sbuff=(char*)(myrank> split_id? &arr[0]   :  &arr[split_indx]);
          lsize=       (myrank<=split_id? split_indx: nelem-split_indx )*sizeof(T);
          lbuff=(char*)(myrank<=split_id? &arr[0]   :  &arr[split_indx]);

          MPI_Status status;
          MPI_Sendrecv                  (&    ssize,1,MPI_INT, partner,0,   &    rsize,1,MPI_INT, partner,   0,comm,&status);
          if(extra_partner) MPI_Sendrecv(&ext_ssize,1,MPI_INT,split_id,0,   &ext_rsize,1,MPI_INT,split_id,   0,comm,&status);

          // Exchange data.
          commBuff.reserve(rsize/sizeof(T));
          char*     rbuff = (char *)(&commBuff[0]);
          char* ext_rbuff=(ext_rsize>0? new char[ext_rsize]: NULL);
          MPI_Sendrecv                  (sbuff,ssize,MPI_BYTE, partner,0,       rbuff,    rsize,MPI_BYTE, partner,   0,comm,&status);
          if(extra_partner) MPI_Sendrecv( NULL,    0,MPI_BYTE,split_id,0,   ext_rbuff,ext_rsize,MPI_BYTE,split_id,   0,comm,&status);

          int nbuff_size=lsize+rsize+ext_rsize;
          mergeBuff.reserve(nbuff_size/sizeof(T));
          char* nbuff= (char *)(&mergeBuff[0]);  // new char[nbuff_size];
          omp_par::merge<T*>((T*)lbuff, (T*)&lbuff[lsize], (T*)rbuff, (T*)&rbuff[rsize], (T*)nbuff, omp_p, std::less<T>());
          if(ext_rsize>0 && nbuff!=NULL){
            // XXX case not handled 
            char* nbuff1= new char[nbuff_size];
            omp_par::merge<T*>((T*)nbuff, (T*)&nbuff[lsize+rsize], (T*)ext_rbuff, (T*)&ext_rbuff[ext_rsize], (T*)nbuff1, omp_p, std::less<T>());
            if(nbuff!=NULL) delete[] nbuff; nbuff=nbuff1;
          }

          // Copy new data.
          totSize=totSize_new;
          nelem = nbuff_size/sizeof(T);
          /*
          if(arr_!=NULL) delete[] arr_; 
          arr_=(T*) nbuff; nbuff=NULL;
          */
          mergeBuff.swap(arr);

          //Free memory.
          // if(    rbuff!=NULL) delete[]     rbuff;
          if(ext_rbuff!=NULL) delete[] ext_rbuff;
        }

        {// Split comm.  O( log(p) ) ??
          MPI_Comm scomm;
          MPI_Comm_split(comm, myrank<=split_id, myrank, &scomm );
          comm=scomm;
          npes  =(myrank<=split_id? split_id+1: npes  -split_id-1);
          myrank=(myrank<=split_id? myrank    : myrank-split_id-1);
        }
      }

      // SortedElem.resize(nelem);
      // SortedElem.assign(arr, &arr[nelem]);
      // if(arr_!=NULL) delete[] arr_;

      // par::partitionW<T>(SortedElem, NULL , comm_);
//      par::partitionW<T>(arr, NULL , comm_);

      PROF_SORT_END
    }//end function

  //--------------------------------------------------------------------------------
  template<typename T>
    int HyperQuickSort(std::vector<T>& arr, std::vector<T> & SortedElem, MPI_Comm comm_){ // O( ((N/p)+log(p))*(log(N/p)+log(p)) ) 
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_SORT_BEGIN
	#ifdef _PROFILE_SORT
		 		total_sort.start();
	#endif

      // Copy communicator.
      MPI_Comm comm=comm_;

      // Get comm size and rank.
      int npes, myrank, myrank_;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &myrank); myrank_=myrank;
      if(npes==1){
        // @dhairya isn't this wrong for the !sort-in-place case ... 
#ifdef _PROFILE_SORT
		 		seq_sort.start();
#endif        
				omp_par::merge_sort(&arr[0],&arr[arr.size()]);
#ifdef _PROFILE_SORT
		 		seq_sort.stop();
#endif        
				SortedElem  = arr;
#ifdef _PROFILE_SORT
		 		total_sort.stop();
#endif        
				PROF_SORT_END
      }

      int omp_p=omp_get_max_threads();
      srand(myrank);

      // Local and global sizes. O(log p)
      DendroIntL totSize, nelem = arr.size(); assert(nelem);
      par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
      DendroIntL nelem_ = nelem;

      // Local sort.
#ifdef _PROFILE_SORT
		 	seq_sort.start();
#endif			
      T* arr_=new T[nelem]; memcpy (&arr_[0], &arr[0], nelem*sizeof(T));      
			omp_par::merge_sort(&arr_[0], &arr_[arr.size()]);
#ifdef _PROFILE_SORT
		 	seq_sort.stop();
#endif
      // Binary split and merge in each iteration.
      while(npes>1 && totSize>0){ // O(log p) iterations.

        //Determine splitters. O( log(N/p) + log(p) )
#ifdef _PROFILE_SORT
			 	hyper_compute_splitters.start();
#endif				
        T split_key;
        DendroIntL totSize_new;
        //while(true)
        { 
          // Take random splitters. O( 1 ) -- Let p * splt_count = glb_splt_count = const = 100~1000
          int splt_count=(1000*nelem)/totSize; 
          if(npes>1000) splt_count=(((float)rand()/(float)RAND_MAX)*totSize<(1000*nelem)?1:0);
          if(splt_count>nelem) splt_count=nelem;
          std::vector<T> splitters(splt_count);
          for(size_t i=0;i<splt_count;i++) 
            splitters[i]=arr_[rand()%nelem];

          // Gather all splitters. O( log(p) )
          int glb_splt_count;
          std::vector<int> glb_splt_cnts(npes);
          std::vector<int> glb_splt_disp(npes,0);
          par::Mpi_Allgather<int>(&splt_count, &glb_splt_cnts[0], 1, comm);
          omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
          glb_splt_count=glb_splt_cnts[npes-1]+glb_splt_disp[npes-1];
          std::vector<T> glb_splitters(glb_splt_count);
          MPI_Allgatherv(&    splitters[0], splt_count, par::Mpi_datatype<T>::value(), 
                         &glb_splitters[0], &glb_splt_cnts[0], &glb_splt_disp[0], 
                         par::Mpi_datatype<T>::value(), comm);

          // Determine split key. O( log(N/p) + log(p) )
          std::vector<DendroIntL> disp(glb_splt_count,0);
          if(nelem>0){
            #pragma omp parallel for
            for(size_t i=0;i<glb_splt_count;i++){
              disp[i]=std::lower_bound(&arr_[0], &arr_[nelem], glb_splitters[i])-&arr_[0];
            }
          }
          std::vector<DendroIntL> glb_disp(glb_splt_count,0);
          MPI_Allreduce(&disp[0], &glb_disp[0], glb_splt_count, par::Mpi_datatype<DendroIntL>::value(), MPI_SUM, comm);

          DendroIntL* split_disp=&glb_disp[0];
          for(size_t i=0;i<glb_splt_count;i++)
            if( labs(glb_disp[i]-totSize/2) < labs(*split_disp-totSize/2)) split_disp=&glb_disp[i];
          split_key=glb_splitters[split_disp-&glb_disp[0]];

          totSize_new=(myrank<=(npes-1)/2?*split_disp:totSize-*split_disp);
          //double err=(((double)*split_disp)/(totSize/2))-1.0;
          //if(fabs(err)<0.01 || npes<=16) break;
          //else if(!myrank) std::cout<<err<<'\n';
        }
#ifdef _PROFILE_SORT
			 	hyper_compute_splitters.stop();
#endif
			
        // Split problem into two. O( N/p )
        int split_id=(npes-1)/2;
        {
#ifdef _PROFILE_SORT
				 	hyper_communicate.start();
#endif				
					
          int new_p0=(myrank<=split_id?0:split_id+1);
          int cmp_p0=(myrank> split_id?0:split_id+1);
          int new_np=(myrank<=split_id? split_id+1: npes-split_id-1);
          int cmp_np=(myrank> split_id? split_id+1: npes-split_id-1);

          int partner = myrank+cmp_p0-new_p0;
          if(partner>=npes) partner=npes-1;
          assert(partner>=0);

          bool extra_partner=( npes%2==1  && npes-1==myrank );

          // Exchange send sizes.
          char *sbuff, *lbuff;
          int     rsize=0,     ssize=0, lsize=0;
          int ext_rsize=0, ext_ssize=0;
          size_t split_indx=(nelem>0?std::lower_bound(&arr_[0], &arr_[nelem], split_key)-&arr_[0]:0);
          ssize=       (myrank> split_id? split_indx: nelem-split_indx )*sizeof(T);
          sbuff=(char*)(myrank> split_id? &arr_[0]   :  &arr_[split_indx]);
          lsize=       (myrank<=split_id? split_indx: nelem-split_indx )*sizeof(T);
          lbuff=(char*)(myrank<=split_id? &arr_[0]   :  &arr_[split_indx]);

          MPI_Status status;
          MPI_Sendrecv                  (&    ssize,1,MPI_INT, partner,0,   &    rsize,1,MPI_INT, partner,   0,comm,&status);
          if(extra_partner) MPI_Sendrecv(&ext_ssize,1,MPI_INT,split_id,0,   &ext_rsize,1,MPI_INT,split_id,   0,comm,&status);

          // Exchange data.
          char*     rbuff=              new char[    rsize]       ;
          char* ext_rbuff=(ext_rsize>0? new char[ext_rsize]: NULL);
          MPI_Sendrecv                  (sbuff,ssize,MPI_BYTE, partner,0,       rbuff,    rsize,MPI_BYTE, partner,   0,comm,&status);
          if(extra_partner) MPI_Sendrecv( NULL,    0,MPI_BYTE,split_id,0,   ext_rbuff,ext_rsize,MPI_BYTE,split_id,   0,comm,&status);
#ifdef _PROFILE_SORT
				 	hyper_communicate.stop();
				 	hyper_merge.start();
#endif
          int nbuff_size=lsize+rsize+ext_rsize;
          char* nbuff= new char[nbuff_size];
          omp_par::merge<T*>((T*)lbuff, (T*)&lbuff[lsize], (T*)rbuff, (T*)&rbuff[rsize], (T*)nbuff, omp_p, std::less<T>());
          if(ext_rsize>0 && nbuff!=NULL){
            char* nbuff1= new char[nbuff_size];
            omp_par::merge<T*>((T*)nbuff, (T*)&nbuff[lsize+rsize], (T*)ext_rbuff, (T*)&ext_rbuff[ext_rsize], (T*)nbuff1, omp_p, std::less<T>());
            if(nbuff!=NULL) delete[] nbuff; nbuff=nbuff1;
          }

          // Copy new data.
          totSize=totSize_new;
          nelem = nbuff_size/sizeof(T);
          if(arr_!=NULL) delete[] arr_; 
          arr_=(T*) nbuff; nbuff=NULL;

          //Free memory.
          if(    rbuff!=NULL) delete[]     rbuff;
          if(ext_rbuff!=NULL) delete[] ext_rbuff;
#ifdef _PROFILE_SORT
				 	hyper_merge.stop();
#endif				
        }

#ifdef _PROFILE_SORT
					hyper_comm_split.start();
#endif				
        {// Split comm.  O( log(p) ) ??
          MPI_Comm scomm;
          MPI_Comm_split(comm, myrank<=split_id, myrank, &scomm );
          comm=scomm;
          npes  =(myrank<=split_id? split_id+1: npes  -split_id-1);
          myrank=(myrank<=split_id? myrank    : myrank-split_id-1);
        }
#ifdef _PROFILE_SORT
				hyper_comm_split.stop();
#endif				
      }

      SortedElem.resize(nelem);
      SortedElem.assign(arr_, &arr_[nelem]);
      if(arr_!=NULL) delete[] arr_;

#ifdef _PROFILE_SORT
		 	sort_partitionw.start();
#endif
//      par::partitionW<T>(SortedElem, NULL , comm_);
#ifdef _PROFILE_SORT
		 	sort_partitionw.stop();
#endif

#ifdef _PROFILE_SORT
		 	total_sort.stop();
#endif
      PROF_SORT_END
    }//end function
// */

  template<typename T>
    int HyperQuickSort_kway(std::vector<T>& arr, std::vector<T> & SortedElem, MPI_Comm comm_) {
      total_sort.clear();
      seq_sort.clear();
      hyper_compute_splitters.clear();
      hyper_communicate.clear();
      hyper_merge.clear();
      hyper_comm_split.clear();
      sort_partitionw.clear();
      MPI_Barrier(comm_);

      PROF_SORT_BEGIN
#ifdef _PROFILE_SORT
      total_sort.start();
#endif
      unsigned int kway = KWAY;
      int omp_p=omp_get_max_threads();

      // Copy communicator.
      MPI_Comm comm=comm_;

      // Get comm size and rank.
      int npes, myrank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &myrank);
      srand(myrank);

      // Local and global sizes. O(log p)
      size_t totSize, nelem = arr.size(); assert(nelem);
      par::Mpi_Allreduce<size_t>(&nelem, &totSize, 1, MPI_SUM, comm);
      std::vector<T> arr_(nelem*2); //Extra buffer.
      std::vector<T> arr__(nelem*2); //Extra buffer.

      // Local sort.
#ifdef _PROFILE_SORT
      seq_sort.start();
#endif
      omp_par::merge_sort(&arr[0], &arr[arr.size()]);
#ifdef _PROFILE_SORT
      seq_sort.stop();
#endif

      while(npes>1 && totSize>0){
        if(kway>npes) kway = npes;
        int blk_size=npes/kway; 
	
	if(blk_size*kway != npes)
	  {
	    printf("blk_size = %i\n",blk_size);
	    printf("kway     = %i\n",kway);
	    printf("npes     = %i\n",npes);
	    //	    std::cout << "blk_size = " << blk_size << std::endl;
	    //	    std::cout << "kway     = " << kway     << std::endl;
	    //std::cout << "npes     = " << npes     << std::endl;
	  }

	assert(blk_size*kway==npes);
        int blk_id=myrank/blk_size, new_pid=myrank%blk_size;

        // Determine splitters.
#ifdef _PROFILE_SORT
        hyper_compute_splitters.start();
#endif
        std::vector<T> split_key = par::Sorted_approx_Select(arr, kway-1, comm);
#ifdef _PROFILE_SORT
        hyper_compute_splitters.stop();
#endif

        {// Communication
#ifdef _PROFILE_SORT
          hyper_communicate.start();
#endif
          // Determine send_size.
          std::vector<int> send_size(kway), send_disp(kway+1); send_disp[0]=0; send_disp[kway]=arr.size();
          for(int i=1;i<kway;i++) send_disp[i]=std::lower_bound(&arr[0], &arr[arr.size()], split_key[i-1])-&arr[0];
          for(int i=0;i<kway;i++) send_size[i]=send_disp[i+1]-send_disp[i];

          // Get recv_size.
          int recv_iter=0;
          std::vector<T*> recv_ptr(kway);
          std::vector<size_t> recv_cnt(kway);
          std::vector<int> recv_size(kway), recv_disp(kway+1,0);
          for(int i_=0;i_<=kway/2;i_++){
            int i1=(blk_id+i_)%kway;
            int i2=(blk_id+kway-i_)%kway;
            MPI_Status status;
            for(int j=0;j<(i_==0 || i_==kway/2?1:2);j++){
              int i=(i_==0?i1:((j+blk_id/i_)%2?i1:i2));
              int partner=blk_size*i+new_pid;
              MPI_Sendrecv(&send_size[     i   ], 1, MPI_INT, partner, 0,
                           &recv_size[recv_iter], 1, MPI_INT, partner, 0, comm, &status);
              recv_disp[recv_iter+1]=recv_disp[recv_iter]+recv_size[recv_iter];
              recv_ptr[recv_iter]=&arr_[recv_disp[recv_iter]];
              recv_cnt[recv_iter]=recv_size[recv_iter];
              recv_iter++;
            }
          }

          // Communicate data.
          int asynch_count=2;
          recv_iter=0;
					int merg_indx=2;
          std::vector<MPI_Request> reqst(kway*2);
          std::vector<MPI_Status> status(kway*2);
          arr_ .resize(recv_disp[kway]);
          arr__.resize(recv_disp[kway]);
          for(int i_=0;i_<=kway/2;i_++){
            int i1=(blk_id+i_)%kway;
            int i2=(blk_id+kway-i_)%kway;
            for(int j=0;j<(i_==0 || i_==kway/2?1:2);j++){
              int i=(i_==0?i1:((j+blk_id/i_)%2?i1:i2));
              int partner=blk_size*i+new_pid;

              if(recv_iter-asynch_count-1>=0) MPI_Waitall(2, &reqst[(recv_iter-asynch_count-1)*2], &status[(recv_iter-asynch_count-1)*2]);
              par::Mpi_Irecv <T>(&arr_[recv_disp[recv_iter]], recv_size[recv_iter], partner, 1, comm, &reqst[recv_iter*2+0]);
              par::Mpi_Issend<T>(&arr [send_disp[     i   ]], send_size[     i   ], partner, 1, comm, &reqst[recv_iter*2+1]);
              recv_iter++;

              int flag[2]={0,0};
              if(recv_iter>merg_indx) MPI_Test(&reqst[(merg_indx-1)*2],&flag[0],&status[(merg_indx-1)*2]);
              if(recv_iter>merg_indx) MPI_Test(&reqst[(merg_indx-2)*2],&flag[1],&status[(merg_indx-2)*2]);
              if(flag[0] && flag[1]){
                T* A=&arr_[0]; T* B=&arr__[0];
                for(int s=2;merg_indx%s==0;s*=2){
                  //std    ::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                  //               &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]]);
                  omp_par::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                                 &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]],omp_p,std::less<T>());
                  T* C=A; A=B; B=C; // Swap
                }
                merg_indx+=2;
              }
            }
          }
#ifdef _PROFILE_SORT
				hyper_communicate.stop();
				hyper_merge.start();
#endif
					// Merge remaining parts.
          while(merg_indx<=(int)kway){
              MPI_Waitall(1, &reqst[(merg_indx-1)*2], &status[(merg_indx-1)*2]);
              MPI_Waitall(1, &reqst[(merg_indx-2)*2], &status[(merg_indx-2)*2]);
              {
                T* A=&arr_[0]; T* B=&arr__[0];
                for(int s=2;merg_indx%s==0;s*=2){
                  //std    ::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                  //               &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]]);
                  omp_par::merge(&A[recv_disp[merg_indx-s/2]],&A[recv_disp[merg_indx    ]],
                                 &A[recv_disp[merg_indx-s  ]],&A[recv_disp[merg_indx-s/2]], &B[recv_disp[merg_indx-s]],omp_p,std::less<T>());
                  T* C=A; A=B; B=C; // Swap
                }
                merg_indx+=2;
              }
          }
					{// Swap buffers.
						int swap_cond=0;
            for(int s=2;kway%s==0;s*=2) swap_cond++;
						if(swap_cond%2==0) swap(arr,arr_);
						else swap(arr,arr__);
					}
				}

#ifdef _PROFILE_SORT
				hyper_merge.stop();
				hyper_comm_split.start();
#endif
				{// Split comm. kway  O( log(p) ) ??
    	     MPI_Comm scomm;
      	   MPI_Comm_split(comm, blk_id, myrank, &scomm );
					 if(comm!=comm_) MPI_Comm_free(&comm);
        	 comm = scomm;

			     MPI_Comm_size(comm, &npes);
           MPI_Comm_rank(comm, &myrank);
    	  }
#ifdef _PROFILE_SORT
				hyper_comm_split.stop();
#endif
      }
#ifdef _PROFILE_SORT
		 	total_sort.stop();
#endif
			SortedElem=arr;

      PROF_SORT_END
    }

  template<typename T>
    int HyperQuickSort_kway_old(std::vector<T>& arr, std::vector<T> & SortedElem, MPI_Comm comm_) { // O( ((N/p)+log(p))*(log(N/p)+log(p)) ) 
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_SORT_BEGIN
#ifdef _PROFILE_SORT
		 	total_sort.start();
#endif

			unsigned int kway = KWAY;
      // Copy communicator.
      MPI_Comm comm=comm_;

      // Get comm size and rank.
      int npes, myrank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &myrank); 
      if(npes==1){
        // FIXME dhairya isn't this wrong for the !sort-in-place case ... 
#ifdef _PROFILE_SORT
		 		seq_sort.start();
#endif
        omp_par::merge_sort(&arr[0], &arr[arr.size()]);
        SortedElem  = arr;
#ifdef _PROFILE_SORT
		 		seq_sort.stop();
				total_sort.stop();
#endif
			  PROF_SORT_END
      }

			// unsigned int logk = binOp::fastLog2(kway);
      int omp_p=omp_get_max_threads();
      srand(myrank);

      // Local and global sizes. O(log p)
      DendroIntL totSize, nelem = arr.size(); assert(nelem);
      par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
      DendroIntL nelem_ = nelem;

      // Local sort.
      std::vector<T> arr_(nelem); // =new T[nelem];
			// T* arr_ = new T[nelem];
#ifdef _PROFILE_SORT
		 	seq_sort.start();
#endif			 
			memcpy (&arr_[0], &arr[0], nelem*sizeof(T));
      omp_par::merge_sort(&arr_[0], &arr_[arr.size()]);
#ifdef _PROFILE_SORT
		 	seq_sort.stop();
#endif
      // while(npes>1 && totSize>0) {
		  while(npes>1 && totSize>0){
		  	// if (!myrank) std::cout << "========================================" << std::endl;
			  if(kway>npes) 
		      kway = npes; 

#ifdef _PROFILE_SORT
			 	hyper_compute_splitters.start();
#endif				
				std::vector<T> split_key = par::Sorted_approx_Select(arr_, kway-1, comm); // select kway-1 splitters 
				std::vector<unsigned int> min_idx, max_idx;
				std::vector<DendroIntL> splitter_ranks;
				
				//std::vector<T> guess = par::Sorted_Sample_Select(arr_, kway-1, min_idx, max_idx, splitter_ranks, comm);
#ifdef __DEBUG_PAR__				
				if (!myrank) 
				{
					std::cout << "kway = " << kway << std::endl;
					std::cout << "guess: ";
          for(size_t i = 0; i < guess.size(); ++i)
					{
						std::cout << guess[i] << " " ;
					}
					std::cout << std::endl << "ranks: ";
					for(size_t i = 0; i < splitter_ranks.size(); ++i)
					{
						std::cout << splitter_ranks[i] << " " ;
					}
					std::cout << std::endl << "ranges: ";
					for(size_t i = 0; i < splitter_ranks.size(); ++i)
					{
						std::cout << "[ " << min_idx[i] << ", " << max_idx[i] << " ]   ";
					}
					std::cout << std::endl;				
				}
#endif
			  //std::vector<T> split_key = Sorted_k_Select(arr_, min_idx, max_idx, splitter_ranks, guess, comm); 	
					
#ifdef __DEBUG_PAR__			
				if (!myrank) 
				{
					std::cout << "kway = " << kway << std::endl;
					std::cout << "split keys size = " << split_key.size() << std::endl;
          std::cout << "splitters: ";
					for(size_t i = 0; i < kway-1; ++i)
					{
						std::cout << split_key[i] << " " ;
					}
					std::cout << std::endl;				
				}
#endif
			
#ifdef _PROFILE_SORT
			 	hyper_compute_splitters.stop();
				hyper_communicate.start();
#endif			
				/*
				1. exchange send sizes - kway
				2. send recv with kway-1 partners - asynchronous 
				3. merge received buffers 
				*/
				unsigned int overhang = npes%kway;
				size_t new_np[kway], new_p0[kway]; 
				for(size_t q = 0; q < kway; ++q) new_np[q] = (q < overhang)?(npes/kway + 1):(npes/kway);
				unsigned int my_chunk=kway-1;
				new_p0[0] = 0; 
				for(size_t q = 1; q < kway; ++q) {
					new_p0[q] = new_p0[q-1] + new_np[q-1];
					if ( (myrank >= new_p0[q-1]) && (myrank < new_p0[q] ) ) my_chunk = q-1;
				}
				
				int new_pid = myrank - new_p0[my_chunk];

#ifdef __DEBUG_PAR__				
				if (!myrank) {
					std::cout << "new p0,np" << std::endl;
					for(size_t i = 0; i < kway; ++i)
					{
						std::cout << "\t" << new_p0[i] << ", " << new_np[i] << std::endl;
					}
				}
#endif				
				// create lbuff from arr_ in my_chunk
				std::vector<T> lbuff, lbuff_tmp;					
				if (nelem > 0) {
					T my_low  = my_chunk?split_key[my_chunk-1]:arr_[0];
					size_t my_indx_low  = (nelem >0 ? std::lower_bound(&arr_[0], &arr_[nelem], my_low)  - &arr_[0]:0);
					unsigned int r = my_indx_low;
					if ( my_chunk == (kway-1) ) {
						while( (r<nelem) ) lbuff.push_back(arr_[r++]);
					} else {
						while( (r<nelem) && (arr_[r]<split_key[my_chunk]) ) lbuff.push_back(arr_[r++]);
					}
				}
				// std::cout << myrank << ":  local size = " << lbuff.size() <<  std::endl;	
				// if (!myrank) std::cout << "starting iSend/Recv" << std::endl;
				
				// first iSend/iRecv cnts ...
				std::vector<int> rsize(2*kway);
				int ssize[kway]; 
				T* sbuff[kway];
				
				// loadBalance instead of following loop. 
				std::vector<MPI_Request> requests; 
				MPI_Request sRequests, rq;	
				MPI_Status  statuses[2*kway]; // overallocate 
				int r_idx=0; 
				for(size_t q = 0; q < kway; ++q) 
				{	
					if (my_chunk == q) continue; // skip self
					int partner = ( new_pid < new_np[q]? new_p0[q] + new_pid: new_p0[q]+new_np[q]-1) ;
					bool have_extra = overhang && (q < overhang) && ((new_p0[my_chunk]+new_np[my_chunk]-1) == myrank );
					int extra_partner = ((kway-1) == q)?npes-1:new_p0[q+1]-1;
	
					// std::cout << myrank << " npid:"  << new_pid << " partner:" << partner << " " << have_extra << " " << extra_partner << std::endl;
					assert(myrank != partner);
					assert(myrank != extra_partner);
	
				  // Exchange send sizes.
					T key_low  = q?split_key[q-1]:arr_[0]; 
					T key_high = ((kway-1) == q)?arr_[nelem-1]:split_key[q]; 
					
					size_t split_indx_low  = (nelem >0 ? std::lower_bound(&arr_[0], &arr_[nelem], key_low)  - &arr_[0]:0);
					size_t split_indx_high;
					
					if (nelem > 0) {
						if ((kway-1) == q) 
							split_indx_high = nelem;
						else
							split_indx_high = std::lower_bound(&arr_[0], &arr_[nelem], split_key[q]) - &arr_[0];	
					} else split_indx_high = 0;
	
				  ssize[q] = ( split_indx_high - split_indx_low );
					sbuff[q] = &arr_[split_indx_low];
						
					// iRecv
					par::Mpi_Irecv<int>( &(rsize[r_idx]), 1, partner, 1, comm, &(rq) );
					requests.push_back(rq); r_idx++;
					// iSend 
					par::Mpi_Issend<int>( &(ssize[q]), 1, partner, 1, comm, &(sRequests) );
					if (have_extra) {
						int ext_ssize=0;
						par::Mpi_Irecv<int> ( &(rsize[r_idx]), 1, extra_partner, 1, comm, &(rq) );
						requests.push_back(rq); r_idx++;
						par::Mpi_Issend<int>( &(ext_ssize),    1, extra_partner, 1, comm, &(sRequests) );
					} 					
				} // q - kway for counts
				rsize.resize(r_idx);		
						
				// no point overlapping here ...
				MPI_Waitall(requests.size(), &(*(requests.begin())), statuses);
				requests.clear();
				

        // MPI_Barrier(comm);
#ifdef __DEBUG_PAR__				
        // if(!myrank) 
        std::cout << myrank << " finished sending sizes" << std::endl;	
#endif				
				//============== Load-Balance here ================/
				//
				//  loadBalance:
				//
 				//  input:   ssize[kway], s_partner[kway], comm
				//  output:  r_partners[] and rsize[]  
				//
				//=================================================/
				
				// if (!myrank) std::cout << "sending actual data" << std::endl;
				
				// now send actual data ...
				std::vector<T*> rbuff(rsize.size());
				r_idx=0; 
				for(size_t q = 0; q < kway; ++q) 
				{
					if (my_chunk == q) continue; // skip self
					int partner = ( new_pid < new_np[q]? new_p0[q] + new_pid: new_p0[q]+new_np[q]-1) ;
					bool have_extra = overhang && (q < overhang) && ((new_p0[my_chunk]+new_np[my_chunk]-1) == myrank );
					int extra_partner = ((kway-1) == q)?npes-1:new_p0[q+1]-1;
	
				  rbuff[r_idx] = (rsize[r_idx]>0? new T[rsize[r_idx]]: NULL);
					// iRecv
					par::Mpi_Irecv<T>( rbuff[r_idx], rsize[r_idx], partner, 1, comm, &(rq) );
					requests.push_back(rq); r_idx++;
					// iSend 
					par::Mpi_Issend<T>( sbuff[q], ssize[q], partner, 1, comm, &(sRequests) );
					if (have_extra) {
						rbuff[r_idx] = (rsize[r_idx]>0? new T[rsize[r_idx]]: NULL);
						// iRecv
					  par::Mpi_Irecv<T>( rbuff[r_idx], rsize[r_idx], extra_partner, 1, comm, &(rq) );
						requests.push_back(rq); r_idx++;
						// iSend 
						par::Mpi_Issend<T>( NULL, 0, extra_partner, 1, comm, &(sRequests) );		
					}
				}
				// if (!myrank) std::cout < "finished iSend/Recv data" << std::endl;
				
#ifdef OVERLAP_KWAY_COMM				
				// overlap here 
				int index[requests.size()], count, remaining;
				remaining = requests.size();
				while(remaining) {
				  MPI_Waitsome(requests.size(), &(*(requests.begin())), &count, index, statuses);
	#ifdef _PROFILE_SORT
					hyper_communicate.stop();
				 	hyper_merge.start();
	#endif
					if (count > 0)
					{
						// if (!myrank) std::cout << "waitsome - processed " << count << std::endl;
						for(size_t p = 0; p < count; ++p)
						{
							int q = index[p];
							if (!rsize[q]) continue;
							// merge recv[index[p]] with local
						  int nbuff_size = lbuff.size() + rsize[q];
						  lbuff_tmp.resize(nbuff_size);
							// if (!myrank) std::cout << "merging " << lbuff.size() << " + " << rsize[q] << std::endl;	
			
						  omp_par::merge<T*>(&(*(lbuff.begin())), &(*(lbuff.end())), rbuff[q], &(rbuff[q][rsize[q]]), &(*(lbuff_tmp.begin())), omp_p, std::less<T>());
				      delete [] rbuff[q];
							lbuff.swap(lbuff_tmp);
							// if(!myrank) std::cout << "after merging:  " << lbuff.size() << " "  << lbuff_tmp.size() << std::endl;
				 		  // TODO : Check if lbuff_tmp needs to be cleared ... 
				 		  // lbuff_tmp.clear();				 
						}
						remaining = remaining - count;
					} else {remaining = 0; }
#ifdef _PROFILE_SORT
				 	hyper_merge.stop();
					hyper_communicate.start();
#endif 
				} // remaining 
#ifdef _PROFILE_SORT
				hyper_communicate.stop();
#endif
	 
					requests.clear(); rsize.clear(); rbuff.clear();
          arr_.swap(lbuff); lbuff.clear();
#else // OVERLAP_KWAY_COMM
        MPI_Waitall(requests.size(), &(*(requests.begin())), statuses);
				requests.clear();
				MPI_Barrier(comm);	
        
        // if (!myrank) std::cout << "In non-overlap" << std::endl;
        #ifdef _PROFILE_SORT
					hyper_communicate.stop();
					hyper_merge.start();
				#endif
					// resize lbuff
          // std::cout << myrank << " lbuf " << lbuff.size() << " rsize " << rsize.size() << std::endl;
					int newTotalSize = lbuff.size();
					T** A = new T*[rsize.size()+1];
					size_t* nA = new size_t[rsize.size()+1];

					int icnt=0;
          for (int i=0; i<rsize.size(); i++) {
            if (!rsize[i]) continue;
            newTotalSize += rsize[i];
						A[icnt]  = rbuff[i];
						nA[icnt] = rsize[i];
            icnt++;
					}
          // std::cout << myrank << " total " << newTotalSize << std::endl;
					A[rsize.size()]  = &(*(lbuff.begin()));
					nA[rsize.size()] = lbuff.size();
          // std::cout << myrank << "resizing" << std::endl;
          arr_.resize(newTotalSize);
          // arr_.clear();
          // T* mArray = new T[newTotalSize];
          // std::cout << myrank << "done resizing" << std::endl;

          // MPI_Barrier(comm);	
          // if (!myrank) std::cout << "calling fan merge " << std::endl;
					par::fan_merge(rsize.size()+1, A, nA, &(*(arr_.begin()))  );
				  // par::fan_merge(rsize.size()+1, A, nA, mArray);
          // if (!myrank) std::cout << "done fan merge " << std::endl;
					
          // std::copy ( mArray, mArray + newTotalSize, arr_.begin() );

					for (int i=0; i<rsize.size(); i++) {
            if (!rsize[i]) continue;
            delete [] rbuff[i];
          }

          // delete [] mArray;
					delete [] A;
					delete [] nA;	
					rsize.clear(); rbuff.clear(); lbuff.clear();	
				#ifdef _PROFILE_SORT
					hyper_merge.stop();
				#endif					
#endif // OVERLAP_KWAY_COMM
									
				
				// if (!myrank) std::cout << myrank << " " << nelem << " " << totSize << std::endl;
				nelem = arr_.size();
				par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
				
#ifdef __DEBUG_PAR__				
        if(!myrank) std::cout << nelem << " " << totSize << std::endl;
        if(!myrank) std::cout << "========================split comm======================= " << std::endl;

#endif				
        
#ifdef _PROFILE_SORT
				hyper_comm_split.start();
#endif				
				{// Split comm. kway  O( log(p) ) ??
          MPI_Comm scomm;
          MPI_Comm_split(comm, my_chunk, myrank, &scomm );
          comm = scomm;
		      MPI_Comm_size(comm, &npes);
		      MPI_Comm_rank(comm, &myrank);
        }
#ifdef _PROFILE_SORT
				hyper_comm_split.stop();
#endif								
      }
			
			SortedElem.swap(arr_);
      // SortedElem.resize(nelem);
      // SortedElem.assign(arr_, &arr_[nelem]);
      // if(arr_!=NULL) delete[] arr_;
#ifdef _PROFILE_SORT
		 	sort_partitionw.start();
#endif
//      par::partitionW<T>(SortedElem, NULL , comm_);
#ifdef _PROFILE_SORT
		 	sort_partitionw.stop();
#endif

#ifdef _PROFILE_SORT
		 	total_sort.stop();
#endif
      PROF_SORT_END
    }//end function
// */


  template<typename T>
    int sampleSort(std::vector<T>& arr, std::vector<T> & SortedElem, MPI_Comm comm){ 
#ifdef __PROFILE_WITH_BARRIER__
      MPI_Barrier(comm);
#endif
      PROF_SORT_BEGIN

#ifdef _PROFILE_SORT
	 		total_sort.start();
#endif

     int npes;

      MPI_Comm_size(comm, &npes);

      assert(arr.size());

      if (npes == 1) {
#ifdef _PROFILE_SORT
				seq_sort.start();
#endif
        omp_par::merge_sort(&arr[0],&arr[arr.size()]);
#ifdef _PROFILE_SORT
  			seq_sort.stop();
#endif        
				SortedElem  = arr;
#ifdef _PROFILE_SORT
		 		total_sort.stop();
#endif      
			  PROF_SORT_END
      } 

      std::vector<T>  splitters;
      std::vector<T>  allsplitters;

      int myrank;
      MPI_Comm_rank(comm, &myrank);

      DendroIntL nelem = arr.size();
      DendroIntL nelemCopy = nelem;
      DendroIntL totSize;
      par::Mpi_Allreduce<DendroIntL>(&nelemCopy, &totSize, 1, MPI_SUM, comm);

      DendroIntL npesLong = npes;
      const DendroIntL FIVE = 5;

      if(totSize < (FIVE*npesLong*npesLong)) {
        if(!myrank) {
          std::cout <<" Using bitonic sort since totSize < (5*(npes^2)). totSize: "
            <<totSize<<" npes: "<<npes <<std::endl;
        }
//        par::partitionW<T>(arr, NULL, comm);

#ifdef __DEBUG_PAR__
        MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-1 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif

        SortedElem = arr; 
        MPI_Comm new_comm;
        if(totSize < npesLong) {
          if(!myrank) {
            std::cout<<" Input to sort is small. splittingComm: "
              <<npes<<" -> "<< totSize<<std::endl;
          }
          par::splitCommUsingSplittingRank(static_cast<int>(totSize), &new_comm, comm);
        } else {
          new_comm = comm;
        }

#ifdef __DEBUG_PAR__
        MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-2 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif

        if(!SortedElem.empty()) {
          par::bitonicSort<T>(SortedElem, new_comm);
        }

#ifdef __DEBUG_PAR__
        MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-3 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif

        PROF_SORT_END
      }// end if

#ifdef __DEBUG_PAR__
      if(!myrank) {
        std::cout<<"Using sample sort to sort nodes. n/p^2 is fine."<<std::endl;
      }
#endif

      //Re-part arr so that each proc. has atleast p elements.
#ifdef _PROFILE_SORT
  		sort_partitionw.start();
#endif
//			par::partitionW<T>(arr, NULL, comm);
#ifdef _PROFILE_SORT
  		sort_partitionw.stop();
#endif
      nelem = arr.size();

#ifdef _PROFILE_SORT
			seq_sort.start();
#endif
      omp_par::merge_sort(&arr[0],&arr[arr.size()]);
#ifdef _PROFILE_SORT
			seq_sort.stop();
#endif
				
      std::vector<T> sendSplits(npes-1);
      splitters.resize(npes);

      #pragma omp parallel for
      for(int i = 1; i < npes; i++)         {
        sendSplits[i-1] = arr[i*nelem/npes];        
      }//end for i

#ifdef _PROFILE_SORT
 		  sample_sort_splitters.start();
#endif
      // sort sendSplits using bitonic ...
      par::bitonicSort<T>(sendSplits,comm);
#ifdef _PROFILE_SORT
 		  sample_sort_splitters.stop();
#endif
				
				
#ifdef _PROFILE_SORT
	 		sample_prepare_scatter.start();
#endif				
      // All gather with last element of splitters.
      T* sendSplitsPtr = NULL;
      T* splittersPtr = NULL;
      if(sendSplits.size() > static_cast<unsigned int>(npes-2)) {
        sendSplitsPtr = &(*(sendSplits.begin() + (npes -2)));
      }
      if(!splitters.empty()) {
        splittersPtr = &(*(splitters.begin()));
      }
      par::Mpi_Allgather<T>(sendSplitsPtr, splittersPtr, 1, comm);

      sendSplits.clear();

      int *sendcnts = new int[npes];
      assert(sendcnts);

      int * recvcnts = new int[npes];
      assert(recvcnts);

      int * sdispls = new int[npes];
      assert(sdispls);

      int * rdispls = new int[npes];
      assert(rdispls);

      #pragma omp parallel for
      for(int k = 0; k < npes; k++){
        sendcnts[k] = 0;
      }

      //To be parallelized
/*      int k = 0;
      for (DendroIntL j = 0; j < nelem; j++) {
        if (arr[j] <= splitters[k]) {
          sendcnts[k]++;
        } else{
          k = seq::UpperBound<T>(npes-1, splittersPtr, k+1, arr[j]);
          if (k == (npes-1) ){
            //could not find any splitter >= arr[j]
            sendcnts[k] = (nelem - j);
            break;
          } else {
            assert(k < (npes-1));
            assert(splitters[k] >= arr[j]);
            sendcnts[k]++;
          }
        }//end if-else
      }//end for j
*/

      {
        int omp_p=omp_get_max_threads();
        int* proc_split = new int[omp_p+1];
        DendroIntL* lst_split_indx = new DendroIntL[omp_p+1];
        proc_split[0]=0;
        lst_split_indx[0]=0;
        lst_split_indx[omp_p]=nelem;
        #pragma omp parallel for
        for(int i=1;i<omp_p;i++){
          //proc_split[i] = seq::BinSearch(&splittersPtr[0],&splittersPtr[npes-1],arr[i*nelem/omp_p],std::less<T>());
          proc_split[i] = std::upper_bound(&splittersPtr[0],&splittersPtr[npes-1],arr[i*(size_t)nelem/omp_p],std::less<T>())-&splittersPtr[0];
          if(proc_split[i]<npes-1){
            //lst_split_indx[i]=seq::BinSearch(&arr[0],&arr[nelem],splittersPtr[proc_split[i]],std::less<T>());
            lst_split_indx[i]=std::upper_bound(&arr[0],&arr[nelem],splittersPtr[proc_split[i]],std::less<T>())-&arr[0];
          }else{
            proc_split[i]=npes-1;
            lst_split_indx[i]=nelem;
          }
        }
        #pragma omp parallel for
        for (int i=0;i<omp_p;i++){
          int sendcnts_=0;
          int k=proc_split[i];
          for (DendroIntL j = lst_split_indx[i]; j < lst_split_indx[i+1]; j++) {
            if (arr[j] <= splitters[k]) {
              sendcnts_++;
            } else{
              if(sendcnts_>0)
                sendcnts[k]=sendcnts_;
              sendcnts_=0;
              k = seq::UpperBound<T>(npes-1, splittersPtr, k+1, arr[j]);
              if (k == (npes-1) ){
                //could not find any splitter >= arr[j]
                sendcnts_ = (nelem - j);
                break;
              } else {
                assert(k < (npes-1));
                assert(splitters[k] >= arr[j]);
                sendcnts_++;
              }
            }//end if-else
          }//end for j
          if(sendcnts_>0)
            sendcnts[k]=sendcnts_;
        }
        delete [] lst_split_indx;
        delete [] proc_split;
      }

      par::Mpi_Alltoall<int>(sendcnts, recvcnts, 1, comm);

      sdispls[0] = 0; rdispls[0] = 0;
//      for (int j = 1; j < npes; j++){
//        sdispls[j] = sdispls[j-1] + sendcnts[j-1];
//        rdispls[j] = rdispls[j-1] + recvcnts[j-1];
//      }
      omp_par::scan(sendcnts,sdispls,npes);
      omp_par::scan(recvcnts,rdispls,npes);

      DendroIntL nsorted = rdispls[npes-1] + recvcnts[npes-1];
      SortedElem.resize(nsorted);

      T* arrPtr = NULL;
      T* SortedElemPtr = NULL;
      if(!arr.empty()) {
        arrPtr = &(*(arr.begin()));
      }
      if(!SortedElem.empty()) {
        SortedElemPtr = &(*(SortedElem.begin()));
      }
#ifdef _PROFILE_SORT
	 		sample_prepare_scatter.stop();
#endif
				
#ifdef _PROFILE_SORT
	 		sample_do_all2all.start();
#endif							
      par::Mpi_Alltoallv_dense<T>(arrPtr, sendcnts, sdispls,
          SortedElemPtr, recvcnts, rdispls, comm);
#ifdef _PROFILE_SORT
	 		sample_do_all2all.stop();
#endif							
      arr.clear();

      delete [] sendcnts;
      sendcnts = NULL;

      delete [] recvcnts;
      recvcnts = NULL;

      delete [] sdispls;
      sdispls = NULL;

      delete [] rdispls;
      rdispls = NULL;

#ifdef _PROFILE_SORT
	 		seq_sort.start();
#endif
      omp_par::merge_sort(&SortedElem[0], &SortedElem[nsorted]);
#ifdef _PROFILE_SORT
	 		seq_sort.stop();
#endif


#ifdef _PROFILE_SORT
	 		total_sort.stop();
#endif
      PROF_SORT_END
    }//end function

  /********************************************************************/
  /*
   * which_keys is one of KEEP_HIGH or KEEP_LOW
   * partner    is the processor with which to Merge and Split.
   *
   */
  template <typename T>
    void MergeSplit( std::vector<T> &local_list, int which_keys, int partner, MPI_Comm  comm) {

      MPI_Status status;
      int send_size = local_list.size();
      int recv_size = 0;

      // first communicate how many you will send and how many you will receive ...

      par::Mpi_Sendrecv<int, int>( &send_size , 1, partner, 0,
          &recv_size, 1, partner, 0, comm, &status);

      std::vector<T> temp_list( recv_size );

      T* local_listPtr = NULL;
      T* temp_listPtr = NULL;
      if(!local_list.empty()) {
        local_listPtr = &(*(local_list.begin()));
      }
      if(!temp_list.empty()) {
        temp_listPtr = &(*(temp_list.begin()));
      }

      par::Mpi_Sendrecv<T, T>( local_listPtr, send_size, partner,
          1, temp_listPtr, recv_size, partner, 1, comm, &status);

      MergeLists<T>(local_list, temp_list, which_keys);

      temp_list.clear();
    } // Merge_split 

  template <typename T>
    void Par_bitonic_sort_incr( std::vector<T> &local_list, int proc_set_size, MPI_Comm  comm ) {
      int  eor_bit;
      int       proc_set_dim;
      int       stage;
      int       partner;
      int       my_rank;

      MPI_Comm_rank(comm, &my_rank);

      proc_set_dim = 0;
      int x = proc_set_size;
      while (x > 1) {
        x = x >> 1;
        proc_set_dim++;
      }

      eor_bit = (1 << (proc_set_dim - 1) );
      for (stage = 0; stage < proc_set_dim; stage++) {
        partner = (my_rank ^ eor_bit);

        if (my_rank < partner) {
          MergeSplit<T> ( local_list,  KEEP_LOW, partner, comm);
        } else {
          MergeSplit<T> ( local_list, KEEP_HIGH, partner, comm);
        }

        eor_bit = (eor_bit >> 1);
      }
    }  // Par_bitonic_sort_incr 


  template <typename T>
    void Par_bitonic_sort_decr( std::vector<T> &local_list, int proc_set_size, MPI_Comm  comm) {
      int  eor_bit;
      int       proc_set_dim;
      int       stage;
      int       partner;
      int       my_rank;

      MPI_Comm_rank(comm, &my_rank);

      proc_set_dim = 0;
      int x = proc_set_size;
      while (x > 1) {
        x = x >> 1;
        proc_set_dim++;
      }

      eor_bit = (1 << (proc_set_dim - 1));
      for (stage = 0; stage < proc_set_dim; stage++) {
        partner = my_rank ^ eor_bit;

        if (my_rank > partner) {
          MergeSplit<T> ( local_list,  KEEP_LOW, partner, comm);
        } else {
          MergeSplit<T> ( local_list, KEEP_HIGH, partner, comm);
        }

        eor_bit = (eor_bit >> 1);
      }

    } // Par_bitonic_sort_decr 

  template <typename T>
    void Par_bitonic_merge_incr( std::vector<T> &local_list, int proc_set_size, MPI_Comm  comm ) {
      int       partner;
      int       rank, npes;

      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &npes);

      unsigned int num_left  =  binOp::getPrevHighestPowerOfTwo(npes);
      unsigned int num_right = npes - num_left;

      // 1, Do merge between the k right procs and the highest k left procs.
      if ( (static_cast<unsigned int>(rank) < num_left) &&
          (static_cast<unsigned int>(rank) >= (num_left - num_right)) ) {
        partner = static_cast<unsigned int>(rank) + num_right;
        MergeSplit<T> ( local_list,  KEEP_LOW, partner, comm);
      } else if (static_cast<unsigned int>(rank) >= num_left) {
        partner = static_cast<unsigned int>(rank) - num_right;
        MergeSplit<T> ( local_list,  KEEP_HIGH, partner, comm);
      }
    }

  template <typename T>
    void bitonicSort_binary(std::vector<T> & in, MPI_Comm comm) {
      int                   proc_set_size;
      unsigned int            and_bit;
      int               rank;
      int               npes;

      MPI_Comm_size(comm, &npes);

#ifdef __DEBUG_PAR__
      assert(npes > 1);
      assert(!(npes & (npes-1)));
      assert(!(in.empty()));
#endif

      MPI_Comm_rank(comm, &rank);

      for (proc_set_size = 2, and_bit = 2;
          proc_set_size <= npes;
          proc_set_size = proc_set_size*2, 
          and_bit = and_bit << 1) {

        if ((rank & and_bit) == 0) {
          Par_bitonic_sort_incr<T>( in, proc_set_size, comm);
        } else {
          Par_bitonic_sort_decr<T>( in, proc_set_size, comm);
        }
      }//end for
    }

  template <typename T>
    void bitonicSort(std::vector<T> & in, MPI_Comm comm) {
      int               rank;
      int               npes;

      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      assert(!(in.empty()));

      //Local Sort first
      //std::sort(in.begin(),in.end());
      omp_par::merge_sort(&in[0],&in[in.size()]);

      if(npes > 1) {

        // check if npes is a power of two ...
        bool isPower = (!(npes & (npes - 1)));

        if ( isPower ) {
          bitonicSort_binary<T>(in, comm);
        } else {
          MPI_Comm new_comm;

          // Since npes is not a power of two, we shall split the problem in two ...
          //
          // 1. Create 2 comm groups ... one for the 2^d portion and one for the
          // remainder.
          unsigned int splitter = splitCommBinary(comm, &new_comm);

          if ( static_cast<unsigned int>(rank) < splitter) {
            bitonicSort_binary<T>(in, new_comm);
          } else {
            bitonicSort<T>(in, new_comm);
          }

          // 3. Do a special merge of the two segments. (original comm).
          Par_bitonic_merge_incr( in,  binOp::getNextHighestPowerOfTwo(npes), comm );

          splitter = splitCommBinaryNoFlip(comm, &new_comm);

          // 4. Now a final sort on the segments.
          if (static_cast<unsigned int>(rank) < splitter) {
            bitonicSort_binary<T>(in, new_comm);
          } else {
            bitonicSort<T>(in, new_comm);
          }
        }//end if isPower of 2
      }//end if single processor
    }//end function

  template <typename T>
    void MergeLists( std::vector<T> &listA, std::vector<T> &listB,
        int KEEP_WHAT) {

      T _low, _high;

      assert(!(listA.empty()));
      assert(!(listB.empty()));

      _low  = ( (listA[0] > listB[0]) ? listA[0] : listB[0]);
      _high = ( (listA[listA.size()-1] < listB[listB.size()-1]) ?
          listA[listA.size()-1] : listB[listB.size()-1]);

      // We will do a full merge first ...
      size_t list_size = listA.size() + listB.size();

      std::vector<T> scratch_list(list_size);

      unsigned int  index1 = 0;
      unsigned int  index2 = 0; 

      for (size_t i = 0; i < list_size; i++) {
        //The order of (A || B) is important here, 
        //so that index2 remains within bounds
        if ( (index1 < listA.size()) && 
            ( (index2 >= listB.size()) ||
              (listA[index1] <= listB[index2]) ) ) {
          scratch_list[i] = listA[index1];
          index1++;
        } else {
          scratch_list[i] = listB[index2];
          index2++;        
        }
      }

      //Scratch list is sorted at this point.

      listA.clear();
      listB.clear();
      if ( KEEP_WHAT == KEEP_LOW ) {
        int ii=0;
        while ( ( (scratch_list[ii] < _low) ||
              (ii < (list_size/2)) )
            && (scratch_list[ii] <= _high) ) {
          ii++;        
        }
        if(ii) {
          listA.insert(listA.end(), scratch_list.begin(),
              (scratch_list.begin() + ii));
        }
      } else {
        int ii = (list_size - 1);
        while ( ( (ii >= (list_size/2)) 
              && (scratch_list[ii] >= _low) )
            || (scratch_list[ii] > _high) ) {
          ii--;        
        }
        if(ii < (list_size - 1) ) {
          listA.insert(listA.begin(), (scratch_list.begin() + (ii + 1)),
              (scratch_list.begin() + list_size));
        }
      }
      scratch_list.clear();
    }//end function

		/*
	template<typename T>
		std::vector<T> GetRangeMean(std::vector<T>& arr, std::vector<unsigned int> range_min, std::vector<unsigned int> range_max, MPI_Comm comm) {
			unsigned int q = range_max.size();  // number of samples ...
			std::vector<T> local_mean(q), global_mean(q);
			std::vector<DendroIntL> n(q), N(q);
			
			// compute local means in ranges
			#pragma omp parallel for
			for(size_t i = 0; i < q; ++i) {
				local_mean[i] = T();
				for(size_t j = range_min[i]; j < range_max[i]; ++j) {
					local_mean[i] += arr[j];
				} // j
				n[i] = (range_max[i] - range_min[i]);
			} // i
			
			// global means ...
			par::Mpi_Allreduce<DendroIntL>(&(*(local_mean.begin())), &(*(global_mean.begin())), q, MPI_SUM, comm);  
			par::Mpi_Allreduce<DendroIntL>(&(*(n.begin())), &(*(N.begin())), q, MPI_SUM, comm);
			
			for(size_t i = 0; i < q; ++i) {
				global_mean[i] /= N[i];
			}  
			return global_mean;
		} // GetRangeMean()
		*/
		
		// ------------------------------

	template<typename T>
		std::vector<T> Sorted_Sample_Select(std::vector<T>& arr, unsigned int kway, std::vector<unsigned int>& min_idx, std::vector<unsigned int>& max_idx, std::vector<DendroIntL>& splitter_ranks, MPI_Comm comm) {
			int rank, npes;
      MPI_Comm_size(comm, &npes);
			MPI_Comm_rank(comm, &rank);
			
			//-------------------------------------------
      DendroIntL totSize, nelem = arr.size(); 
      par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
			
			//Determine splitters. O( log(N/p) + log(p) )        
      int splt_count = (1000*kway*nelem)/totSize; 
      if (npes>1000*kway) splt_count = (((float)rand()/(float)RAND_MAX)*totSize<(1000*kway*nelem)?1:0);
      if (splt_count>nelem) splt_count=nelem;
      std::vector<T> splitters(splt_count);
      for(size_t i=0;i<splt_count;i++) 
        splitters[i] = arr[rand()%nelem];

      // Gather all splitters. O( log(p) )
      int glb_splt_count;
      std::vector<int> glb_splt_cnts(npes);
      std::vector<int> glb_splt_disp(npes,0);
      par::Mpi_Allgather<int>(&splt_count, &glb_splt_cnts[0], 1, comm);
      omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
      glb_splt_count = glb_splt_cnts[npes-1] + glb_splt_disp[npes-1];
      std::vector<T> glb_splitters(glb_splt_count);
      MPI_Allgatherv(&    splitters[0], splt_count, par::Mpi_datatype<T>::value(), 
                     &glb_splitters[0], &glb_splt_cnts[0], &glb_splt_disp[0], 
                     par::Mpi_datatype<T>::value(), comm);

      // rank splitters. O( log(N/p) + log(p) )
      std::vector<DendroIntL> disp(glb_splt_count,0);
      if(nelem>0){
        #pragma omp parallel for
        for(size_t i=0; i<glb_splt_count; i++){
          disp[i] = std::lower_bound(&arr[0], &arr[nelem], glb_splitters[i]) - &arr[0];
        }
      }
      std::vector<DendroIntL> glb_disp(glb_splt_count, 0);
      MPI_Allreduce(&disp[0], &glb_disp[0], glb_splt_count, par::Mpi_datatype<DendroIntL>::value(), MPI_SUM, comm);
        
			splitter_ranks.clear(); splitter_ranks.resize(kway);	
			min_idx.clear(); min_idx.resize(kway);
			max_idx.clear(); max_idx.resize(kway);	
	    std::vector<T> split_keys(kway);
			#pragma omp parallel for
      for (unsigned int qq=0; qq<kway; qq++) {
				DendroIntL* _disp = &glb_disp[0];
				DendroIntL* _mind = &glb_disp[0];
				DendroIntL* _maxd = &glb_disp[0];
				DendroIntL optSplitter = ((qq+1)*totSize)/(kway+1);
        // if (!rank) std::cout << "opt " << qq << " - " << optSplitter << std::endl;
        for(size_t i=0; i<glb_splt_count; i++) {
        	if(labs(glb_disp[i] - optSplitter) < labs(*_disp - optSplitter)) {
						_disp = &glb_disp[i];
					}
        	if( (glb_disp[i] > optSplitter) && ( labs(glb_disp[i] - optSplitter) < labs(*_maxd - optSplitter))  ) {
						_maxd = &glb_disp[i];
					}
        	if( (glb_disp[i] < optSplitter) && ( labs(optSplitter - glb_disp[i]) < labs(optSplitter - *_mind))  ) {
						_mind = &glb_disp[i];
					}
				}
        split_keys[qq] = glb_splitters[_disp - &glb_disp[0]];
				min_idx[qq] = std::lower_bound(&arr[0], &arr[nelem], glb_splitters[_mind - &glb_disp[0]]) - &arr[0];
				max_idx[qq] = std::upper_bound(&arr[0], &arr[nelem], glb_splitters[_maxd - &glb_disp[0]]) - &arr[0];
				splitter_ranks[qq] = optSplitter - *_mind;
			}
			
			return split_keys;
		}	
	
  template<typename T>
    void Sorted_approx_Select_helper(std::vector<T>& arr, std::vector<size_t>& exp_rank, std::vector<T>& splt_key, int beta, std::vector<size_t>& start, std::vector<size_t>& end, size_t& max_err, MPI_Comm comm) {
      //int dbg_cnt=0; MPI_Barrier(comm);
      //std::vector<double> tt(1000,-omp_get_wtime());
      //MPI_Barrier(comm); tt[dbg_cnt]+=omp_get_wtime(); dbg_cnt++; //////////////////////////////////////////////////////////////////////

      int rank, npes;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);
      
      size_t nelem=arr.size();
      int kway=exp_rank.size();
      std::vector<size_t> locSize(kway), totSize(kway);
      for(int i=0;i<kway;i++) locSize[i]=end[i]-start[i];
      par::Mpi_Allreduce<size_t>(&locSize[0], &totSize[0], kway, MPI_SUM, comm);
      //MPI_Barrier(comm); tt[dbg_cnt]+=omp_get_wtime(); dbg_cnt++; //////////////////////////////////////////////////////////////////////

      //-------------------------------------------
      std::vector<T> loc_splt;
      for(int i=0;i<kway;i++){
        int splt_count = (totSize[i]==0?1:(beta*(end[i]-start[i]))/totSize[i]);
        if (npes>beta) splt_count = (((float)rand()/(float)RAND_MAX)*totSize[i]<(beta*locSize[i])?1:0);
        for(int j=0;j<splt_count;j++) loc_splt.push_back(arr[start[i]+rand()%(locSize[i]+1)]);
        std::sort(&loc_splt[loc_splt.size()-splt_count],&loc_splt[loc_splt.size()]);
      }
      //MPI_Comm comm_;
      //MPI_Comm_split(comm, (loc_splt.size()>0?1:0), rank, &comm_);
      //if(loc_splt.size()>0) bitonicSort<T>(loc_splt, comm_);
      int splt_count=loc_splt.size();
      //MPI_Barrier(comm); tt[dbg_cnt]+=omp_get_wtime(); dbg_cnt++; //////////////////////////////////////////////////////////////////////
      
      // Gather all splitters. O( log(p) )
      int glb_splt_count;
      std::vector<int> glb_splt_cnts(npes);
      std::vector<int> glb_splt_disp(npes,0);
      par::Mpi_Allgather<int>(&splt_count, &glb_splt_cnts[0], 1, comm);
      omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
      glb_splt_count = glb_splt_cnts[npes-1] + glb_splt_disp[npes-1];
      std::vector<T> glb_splt(glb_splt_count);
      MPI_Allgatherv(&loc_splt[0], splt_count, par::Mpi_datatype<T>::value(), 
                     &glb_splt[0], &glb_splt_cnts[0], &glb_splt_disp[0], par::Mpi_datatype<T>::value(), comm);
      //MPI_Barrier(comm); tt[dbg_cnt]+=omp_get_wtime(); dbg_cnt++; //////////////////////////////////////////////////////////////////////
      std::sort(&glb_splt[0],&glb_splt[glb_splt_count]);
      //MPI_Barrier(comm); tt[dbg_cnt]+=omp_get_wtime(); dbg_cnt++; //////////////////////////////////////////////////////////////////////

      // rank splitters. O( log(N/p) + log(p) )
      std::vector<size_t> loc_rank(glb_splt_count,0);
      if(nelem>0){
        #pragma omp parallel for
        for(size_t i=0; i<glb_splt_count; i++){
          loc_rank[i] = std::lower_bound(&arr[0], &arr[nelem], glb_splt[i]) - &arr[0];
        }
      }
      //MPI_Barrier(comm); tt[dbg_cnt]+=omp_get_wtime(); dbg_cnt++; //////////////////////////////////////////////////////////////////////
      std::vector<size_t> glb_rank(glb_splt_count, 0);
      MPI_Allreduce(&loc_rank[0], &glb_rank[0], glb_splt_count, par::Mpi_datatype<size_t>::value(), MPI_SUM, comm);
      //MPI_Barrier(comm); tt[dbg_cnt]+=omp_get_wtime(); dbg_cnt++; //////////////////////////////////////////////////////////////////////

      size_t new_max_err=0;
      std::vector<T> split_keys(kway);
      #pragma omp parallel for
      for (int i=0; i<kway; i++) {
        int ub_indx=std::upper_bound(&glb_rank[0], &glb_rank[glb_splt_count], exp_rank[i])-&glb_rank[0];
        int lb_indx=ub_indx-1; if(lb_indx<0) lb_indx=0;
        size_t err=labs(glb_rank[lb_indx]-exp_rank[i]);

        if(err<max_err){
          if(glb_rank[lb_indx]>exp_rank[i]) start[i]=0;
          else start[i] = loc_rank[lb_indx];
          if(ub_indx==glb_splt_count) end[i]=nelem;
          else end[i] = loc_rank[ub_indx];
          splt_key[i]=glb_splt[lb_indx];
          if(new_max_err<err) new_max_err=err;
        }
      }
      max_err=new_max_err;
      //MPI_Barrier(comm); tt[dbg_cnt]+=omp_get_wtime(); dbg_cnt++; //////////////////////////////////////////////////////////////////////

      //if(!rank && npes>=4096){
      //  for(int i=1;i<dbg_cnt;i++) std::cout<<tt[i]-tt[i-1]<<' ';
      //  std::cout<<'\n';
      //}
    }

  template<typename T>
    std::vector<T> Sorted_approx_Select_new(std::vector<T>& arr, unsigned int kway, MPI_Comm comm) {
      int rank, npes;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);
      
      //-------------------------------------------
      DendroIntL totSize, nelem = arr.size(); 
      par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);

      double tol=1e-2/kway;
      int beta=pow(1.0/tol,1.0/3.0)*3.0;
      std::vector<T> splt_key(kway);
      std::vector<size_t> start(kway,0);
      std::vector<size_t> end(kway,nelem);
      std::vector<size_t> exp_rank(kway);
      for(int i=0;i<kway;i++) exp_rank[i]=((i+1)*totSize)/(kway+1);
      
      //MPI_Barrier(comm);
      //double tt=omp_get_wtime();
      //int iter_cnt=0;

      size_t max_error=totSize;
      while(max_error>totSize*tol){
        //MPI_Barrier(comm);
        //double tt=omp_get_wtime();
        Sorted_approx_Select_helper(arr, exp_rank, splt_key, beta, start, end, max_error, comm);
        //MPI_Barrier(comm);
        //if(!rank) std::cout<<log(max_error*1.0/totSize)/log(0.1)<<' '<<omp_get_wtime()-tt<<'\n';
        //iter_cnt++;
      }

      //MPI_Barrier(comm);
      //if(!rank && npes>=4096) std::cout<<beta<<' '<<iter_cnt<<' '<<log(max_error*1.0/totSize)/log(0.1)<<' '<<omp_get_wtime()-tt<<'\n';
      return splt_key;
    }
    
	template<typename T>
		std::vector<T> Sorted_approx_Select(std::vector<T>& arr, unsigned int kway, MPI_Comm comm) {
			int rank, npes;
      MPI_Comm_size(comm, &npes);
			MPI_Comm_rank(comm, &rank);
			
			//-------------------------------------------
      DendroIntL totSize, nelem = arr.size(); 
      par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
			
			//Determine splitters. O( log(N/p) + log(p) )        
      int splt_count = (1000*kway*nelem)/totSize; 
      if (npes>1000*kway) splt_count = (((float)rand()/(float)RAND_MAX)*totSize<(1000*kway*nelem)?1:0);
      if (splt_count>nelem) splt_count=nelem;
      std::vector<T> splitters(splt_count);
      for(size_t i=0;i<splt_count;i++) 
        splitters[i] = arr[rand()%nelem];

      // Gather all splitters. O( log(p) )
      int glb_splt_count;
      std::vector<int> glb_splt_cnts(npes);
      std::vector<int> glb_splt_disp(npes,0);
      par::Mpi_Allgather<int>(&splt_count, &glb_splt_cnts[0], 1, comm);
      omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
      glb_splt_count = glb_splt_cnts[npes-1] + glb_splt_disp[npes-1];
      std::vector<T> glb_splitters(glb_splt_count);
      MPI_Allgatherv(&    splitters[0], splt_count, par::Mpi_datatype<T>::value(), 
                     &glb_splitters[0], &glb_splt_cnts[0], &glb_splt_disp[0], 
                     par::Mpi_datatype<T>::value(), comm);

      // rank splitters. O( log(N/p) + log(p) )
      std::vector<DendroIntL> disp(glb_splt_count,0);
      if(nelem>0){
        #pragma omp parallel for
        for(size_t i=0; i<glb_splt_count; i++){
          disp[i] = std::lower_bound(&arr[0], &arr[nelem], glb_splitters[i]) - &arr[0];
        }
      }
      std::vector<DendroIntL> glb_disp(glb_splt_count, 0);
      MPI_Allreduce(&disp[0], &glb_disp[0], glb_splt_count, par::Mpi_datatype<DendroIntL>::value(), MPI_SUM, comm);
        
	    std::vector<T> split_keys(kway);
			#pragma omp parallel for
      for (unsigned int qq=0; qq<kway; qq++) {
				DendroIntL* _disp = &glb_disp[0];
				DendroIntL optSplitter = ((qq+1)*totSize)/(kway+1);
        // if (!rank) std::cout << "opt " << qq << " - " << optSplitter << std::endl;
        for(size_t i=0; i<glb_splt_count; i++) {
        	if(labs(glb_disp[i] - optSplitter) < labs(*_disp - optSplitter)) {
						_disp = &glb_disp[i];
					}
				}
        split_keys[qq] = glb_splitters[_disp - &glb_disp[0]];
			}
			
			return split_keys;
		}	
		
	template<typename T>
		inline std::vector<T> GuessRangeMedian(std::vector<T>& arr, std::vector<unsigned int> range_min, std::vector<unsigned int> range_max, MPI_Comm comm) {
			// std::cout << "in GuessRangeMedian" << std::endl;
			int rank;
      MPI_Comm_rank(comm, &rank);
			
			std::vector<T> guess;
			
			unsigned int q = range_min.size();
			
			/*
			par::RandomPick<T> in[q], out[q]; 
			
			for(size_t i = 0; i < q; ++i) {
				in[i] = par::RandomPick<T>(rand(), arr[(range_min[i] + range_max[i])/2]);
			}
			MPI_Allreduce( in, out, q, par::Mpi_datatype<RandomPick<T> >::value(), MPI_SUM, comm );
			for(size_t i = 0; i < q; ++i) {
				guess.push_back(out[i].data());
			}
			*/
			
			// reduce to pick a non-zero proc 
			int in[2*q], out[2*q];
			for(size_t i = 0; i < q; ++i) {
				if (range_max[i] - range_min[i]) {
					in[2*i] = rand();
				} else {
					in[2*i] = 0;	
				}
				in[2*i + 1] = rank; 
			}
			MPI_Allreduce( in, out, q, par::Mpi_pairtype<int, int>::value(), MPI_MAXLOC, comm );
			
			// ranks for each proc are now present in `out`  
			for(size_t i = 0; i < q; ++i) {
				T val;
				if (rank == out[2*i+1]) {
					val = arr[(range_min[i] + range_max[i])/2];
				}
				par::Mpi_Bcast<T>(&val, 1, out[2*i+1], comm);
				guess.push_back(val);
			}
			return guess;
		}
		
		template<typename T>
		void rankSamples(std::vector<T>& arr, std::vector<T> K, MPI_Comm comm) {
			int rank, npes;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);
			
			unsigned int q = K.size();  // number of samples ...
      DendroIntL totSize, nelem = arr.size(); 
      par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
			
			DendroIntL nloc_less_or_equal[q], nloc_lesser[q];
			for(size_t i = 0; i < q; ++i) {
				nloc_lesser  [i] = 0;
				nloc_less_or_equal [i] = 0;
			}
			
			// 1. find local num elements less and greater than guess[i]
			#pragma omp parallel for
			for(size_t i = 0; i < q; ++i) {
				nloc_lesser [i] = std::lower_bound(&arr[0], &arr[nelem], K[i]) - &arr[0]; 
				nloc_less_or_equal[i] = std::upper_bound(&arr[0], &arr[nelem], K[i]) - &arr[0]; 
			} // q
			
			// 2. find global num elements less and greater than guess[i]
			DendroIntL global_lesser[q], global_less_or_equal[q];
			par::Mpi_Allreduce<DendroIntL>(nloc_lesser, global_lesser, q, MPI_SUM, comm);  
			par::Mpi_Allreduce<DendroIntL>(nloc_less_or_equal, global_less_or_equal, q, MPI_SUM, comm);  
			
			
			if (!rank) {
				std::cout << "Error in ranks : \t";
				for(size_t i = 0; i < q; ++i) {
					DendroIntL optSplitter = ((i+1)*totSize)/(q+1);
					double err_low = (optSplitter - global_lesser[i]); err_low = err_low/totSize; err_low = fabs(err_low*100);
					double err_high = (optSplitter - global_less_or_equal[i]); err_high = err_high/totSize; err_high = fabs(err_high*100);
					std::cout << "[ " << err_low <<  " ]  ";
				}
				std::cout << std::endl;
			}	
		}	
		
	template<typename T>
		std::vector<T> Sorted_k_Select(std::vector<T>& arr, std::vector<unsigned int> range_min, std::vector<unsigned int> range_max, std::vector<DendroIntL> splitter_ranks, std::vector<T> _guess, MPI_Comm comm) {
			int rank, npes;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);
			
			unsigned int q = splitter_ranks.size();  // number of samples ...
			
#ifdef __DEBUG_PAR__
			if (!rank) {
			std::cout << q << ": guesses: \t";
			/*
      for(size_t i = 0; i < q; ++i) {
				std::cout << _guess[i] << "\t";
			} 
      */
			std::cout << std::endl;
			for(size_t i = 0; i < q; ++i) {
				std::cout << "[ " << range_min[i] << " | " << splitter_ranks[i] << " | " << range_max[i] << " ]   ";
			}
			std::cout << std::endl;
			}
#endif			
			
			DendroIntL n[q], N[q];
			DendroIntL nloc_less_or_equal[q], nloc_lesser[q];

			std::vector<unsigned int> min_idx, max_idx, newMin, newMax;
			std::vector<DendroIntL> K, newK;			
			std::vector<T> guess, selects;
			
			guess = _guess;
			K = splitter_ranks;
			min_idx = range_min;
			max_idx = range_max;

			unsigned int iters=0;
			while ( selects.size() != splitter_ranks.size() ) {
				iters++;
				// if (guess.size() == splitter_ranks.size())
				// par::rankSamples(arr, guess, comm);
			
				q = K.size();
				for(size_t i = 0; i < q; ++i) {
					n[i] = max_idx[i] - min_idx[i];
					N[i] = 0; 
					nloc_lesser  [i] = 0;
					nloc_less_or_equal [i] = 0;
				}
	      par::Mpi_Allreduce<DendroIntL>(n, N, q, MPI_SUM, comm);
			
				// find K[i] ranked element between arr[min_idx[i]] and arr[max_idx[i]] ... 
			
				// 1. find local num elements less and greater than guess[i]
				#pragma omp parallel for
				for(size_t i = 0; i < q; ++i) {
					nloc_lesser [i] = std::lower_bound(&arr[min_idx[i]], &arr[max_idx[i]], guess[i]) - &arr[min_idx[i]]; 
					nloc_less_or_equal[i] = std::upper_bound(&arr[min_idx[i]], &arr[max_idx[i]], guess[i]) - &arr[min_idx[i]]; 
				} // q
			
				// 2. find global num elements less and greater than guess[i]
				DendroIntL global_lesser[q], global_less_or_equal[q];
				par::Mpi_Allreduce<DendroIntL>(nloc_lesser, global_lesser, q, MPI_SUM, comm);  
				par::Mpi_Allreduce<DendroIntL>(nloc_less_or_equal, global_less_or_equal, q, MPI_SUM, comm);  
				
				newMin.clear(); newMax.clear(); newK.clear();
				// 3. see if converged, else recurse
				for(size_t i = 0; i < q; ++i) {
					if ( ( (global_lesser[i] <= K[i]) && (K[i] <= global_less_or_equal[i]) ) || (!global_lesser[i]) || (global_lesser[i] == N[i]) ) 
					{ // this one has converged ...
						selects.push_back(guess[i]);
						// if (!rank) std::cout << i << ": Selected split " << guess[i] << " of rank " << K[i] << std::endl;
					} 
					else
					{
						if ( global_less_or_equal[i] < K[i] ) {
							// if (!rank) std::cout << "lesser: " << global_less_or_equal[i] << " " << K[i] << std::endl;							
							newMin.push_back(min_idx[i] + nloc_less_or_equal[i] ); 
							newMax.push_back(max_idx[i]); 
							newK.push_back(K[i] - global_less_or_equal[i]);
						} else if ( K[i] < global_lesser[i] ) {
							// if (!rank) std::cout << "greater" << std::endl;
							newMin.push_back(min_idx[i]); 
							newMax.push_back(min_idx[i] + nloc_lesser[i]); 
							newK.push_back(K[i]);	
						}		
					}   
				} // q
				guess = par::GuessRangeMedian<T>(arr, newMin, newMax, comm);
				min_idx = newMin;
				max_idx = newMax;
				K = newK;
			} // while ...
			// if (!rank) std::cout << "kSelect took " << iters << " iterations. " << std::endl;
			
			std::sort(selects.begin(), selects.end());
			return selects;			
		
		} // end function - kSelect

		template <class T>
		void fan_merge(int k, T** A, size_t* n, T* C_){
		  size_t totSize=0;
		  std::vector<size_t> disp(k+1,0);
		  for(int i=0;i<k;i++){
		    totSize+=n[i];
		    disp[i+1]=disp[i]+n[i];
		  }

		  T* B_=new T[totSize];
		  T* B=B_;
		  T* C=C_;
		  for(int j=1;j<k;j=j*2){
		    for(int i=0;i<k;i=i+2*j){
		      if(i+j<k){
		        //omp_par::merge(A[i],A[i]+n[i],A[i+j],A[i+j]+n[i+j],&B[disp[i]],k, std::less<T>());
		        std::merge(A[i],A[i]+n[i],A[i+j],A[i+j]+n[i+j],&B[disp[i]]);
		        A[i]=&B[disp[i]];
		        n[i]=n[i]+n[i+j];
		      }else{
		        memcpy(&B[disp[i]], A[i], n[i]*sizeof(T));
		        A[i]=&B[disp[i]];
		      }
		    }
		    B=C; //Swap buffers.
		    C=A[0];
		  }

			// Final result should be in C_;
		  if(C_!=A[0]) memcpy(C_, A[0], totSize*sizeof(T));

		  //Free memory.
		  delete[] B_;
		}


		template<typename T>
		int partitionSubArrays(std::vector<T*>& arr, std::vector<int>& a_sz) 
		{
			int rank, npes;
			MPI_Comm_size(comm, &npes);
			MPI_Comm_rank(comm, &rank);
			MPI_Request request;
			MPI_Status status;			
						
			// 1.  Compute optimal load partition ...
      int narr = a_sz.size();
			DendroIntL totSize, nelem = 0; 
      for(size_t i = 0; i < narr; ++i)
				nelem += a_sz[i];
      par::Mpi_Allreduce<DendroIntL>(&nelem, &totSize, 1, MPI_SUM, comm);
			
			DendroIntL N_opt = totSize/npes;   // desired partition ...
			
			
      // 2. perform a scan on the weights
      DendroIntL lscn[narr];
			DendroIntL zero = 0, off1, off2;
      if(narr) {
        lscn[0] = a_sz[0];
        omp_par::scan(&(a_sz[1]), lscn, narr);
        // now scan with the final members of 
        par::Mpi_Scan<DendroIntL>(lscn + narr-1, &off1, 1, MPI_SUM, comm ); 
      } else{
        par::Mpi_Scan<DendroIntL>(&zero, &off1, 1, MPI_SUM, comm ); 
      }

      // communicate the offsets ...
      if (rank < (npes-1)){
        par::Mpi_Issend<DendroIntL>( &off1, 1, rank+1, 0, comm, &request );
      }
      if (rank){
        par::Mpi_Recv<DendroIntL>( &off2, 1, rank-1, 0, comm, &status );
      }
      else{
        off2 = 0; 
      }

      // add offset to local array
      #pragma omp parallel for
      for (DendroIntL i = 0; i < narr; i++) {
        lscn[i] = lscn[i] + off2;       // This has the global scan results now ...
      }//end for
		
			// figure out to whom the subarray belongs ...
		
		}


    template <typename T>
      int bucketData(std::vector<T> &in, std::vector<T>& splitters, std::vector<T>& out, MPI_Comm comm) {
        int npes, myrank;
        MPI_Comm_size(comm, &npes);
        MPI_Comm_rank(comm, &myrank);
      
        // easier if data is locally sorted ...
        omp_par::merge_sort(&in[0], &in[in.size()]);
        
        unsigned int k = splitters.size();
       
        // locally bin the data.
        std::vector<int> bucket_size(k), bucket_disp(k+1); 
        bucket_disp[0]=0; bucket_disp[k] = in.size();

        for(int i=1; i<k; i++) bucket_disp[i] = std::lower_bound(&in[0], &in[in.size()], splitters[i]) - &in[0];
        for(int i=0; i<k; i++) bucket_size[i] = bucket_disp[i+1] - bucket_disp[i];
        
        for (int i=0; i<k; i++) {
          // load balance bucket i
          std::vector<T> bucket(bucket_size[i]);
          std::copy(&in[bucket_disp[i]], &in[bucket_disp[i+1]], bucket.begin() );
          par::partitionW<T>(bucket, NULL, comm);

          // write out ?
          // char filename[256];
          FILE* fp = fopen("/tmp/temp_array.dat", "wb");
          fwrite(&in[0], sizeof(T), in.size(), fp);

          fclose(fp);
          // update
          bucket.clear();
          bucket_prev = bucket_disp;
        }


        return 0;
      }


  //#define USE_ORIG_BUCKET_FUNC

#ifdef USE_ORIG_BUCKET_FUNC

  template <typename T>
  std::vector<int> bucketDataAndWrite(std::vector<T> &in, std::vector<T> splitters, 
				      const char* filename, MPI_Comm comm) {
        int npes, myrank;
        MPI_Comm_size(comm, &npes);
        MPI_Comm_rank(comm, &myrank);
      
        // easier if data is locally sorted ...
        omp_par::merge_sort(&in[0], &in[in.size()]);
        
        unsigned int k = splitters.size();
	std::vector<int> writeCounts(k,0);
       
        // locally bin the data.
        std::vector<int> bucket_size(k), bucket_disp(k+1); 
        bucket_disp[0]=0; bucket_disp[k] = in.size();

        for(int i=1; i<k; i++) bucket_disp[i] = std::lower_bound(&in[0], &in[in.size()], splitters[i]) - &in[0];
        for(int i=0; i<k; i++) bucket_size[i] = bucket_disp[i+1] - bucket_disp[i];
        
        for (int i=0; i<k; i++) {
          // load balance bucket i
          std::vector<T> bucket(bucket_size[i]);
          std::copy(&in[bucket_disp[i]], &in[bucket_disp[i+1]], bucket.begin() );
          par::partitionW<T>(bucket, NULL, comm);

          // write out ?
          char fname[1024];
          sprintf(fname, "%s_%03d.dat", filename, i);
          FILE* fp = fopen(fname, "wb");

          fwrite(&bucket[0], sizeof(T), bucket.size(), fp);
          fclose(fp);

	  writeCounts[i] = bucket.size();

          // update
          bucket.clear();
          // bucket_prev = bucket_disp;  (ks 4/11/13 - var not used)
        }


        //return 0;
	return(writeCounts);
     }

#else

    template <typename T>
    std::vector <int> bucketDataAndWrite(std::vector<T> &in, std::vector<T> splitters, 
					    const char* filename, MPI_Comm comm) {
        int npes, myrank;
        MPI_Comm_size(comm, &npes);
        MPI_Comm_rank(comm, &myrank);
      
        // easier if data is locally sorted ...
        omp_par::merge_sort(&in[0], &in[in.size()]);
        
        unsigned int k = splitters.size();
	std::vector<int> writeCounts(k+1,0);
       
        // locally bin the data.
        std::vector<int> bucket_size(k+1), bucket_disp(k+2); 
        bucket_disp[0]=0; bucket_disp[k+1] = in.size();

        for(int i=0; i<k; i++) bucket_disp[i+1] = std::lower_bound(&in[0], &in[in.size()], splitters[i]) - &in[0];
        //for(int i=1; i<k; i++) bucket_disp[i] = std::lower_bound(&in[0], &in[in.size()], splitters[i]) - &in[0];
        for(int i=0; i<k+1; i++) {
          bucket_size[i] = bucket_disp[i+1] - bucket_disp[i];
          // std::cout << myrank << " " << i << " " << bucket_size[i] << std::endl;
        }

	//#define USE_FWRITE

#ifdef USE_FWRITE
        FILE* fp;
#else
        int fd;
#endif
        char fname[1024];
        for (int i=0; i<k+1; i++) {
          // load balance bucket i
          std::vector<T> bucket(bucket_size[i]);
          std::copy(&in[bucket_disp[i]], &in[bucket_disp[i+1]], bucket.begin() );
          par::partitionW<T>(bucket, NULL, comm);

	  writeCounts[i] = bucket.size();

          // write out ?
          //sprintf(fname, "%s_%d_%03d.dat", filename, myrank, i);
          sprintf(fname, "%s_%03d.dat", filename, i);
#ifdef USE_FWRITE          
          fp = fopen(fname, "wb");
          fwrite(&bucket[0], sizeof(T), bucket.size(), fp);
          fclose(fp);
#else
          fd = open(fname, O_WRONLY | O_CREAT, S_IWUSR);
          if (fd == -1) {
            perror("File cannot be opened");
	    exit(1);

          }
          write(fd, &bucket[0], sizeof(T)*bucket.size() );
          close (fd);
#endif
          // update
          bucket.clear();
        }

	return(writeCounts);
     }


#endif


}//end namespace

