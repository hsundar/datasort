#include <cstdlib>
#include <iostream>
#include <cassert>

#define MAX(A_,B_) ((A_)>(B_) ? (A_) : (B_))
template <class T>
void sse<T,4>::merge(T* A_, T* A_last,T* B_, T* B_last, T* C_){
  //c = Traits<T>::_mm_min(a,b);
  //std::cout<<"size: 4\n"; //TODO: Your code here
  //float d = Traits<T>::_mm_min(A_[0],A_[1]);
  unsigned int size_1 = A_last-A_;
  unsigned int size_2 = B_last-B_;
  if(size_1==0) {memcpy(C_,B_,size_2*sizeof(T)); return;}
  if(size_2==0) {memcpy(C_,A_,size_1*sizeof(T)); return;}

  m128 L1,H1,L1p,H1p,L2,H2,L2p,H2p,L3,H3,O1,O2;
  m128 S1,S2;
  int flag1=0,flag2=0;
  m128* pDest = (m128*)C_;
  T *temp, *temp1=A_, *temp2=B_;
  unsigned int excess_1 = size_1 % 4;
  unsigned int excess_2 = size_2 % 4;
  unsigned int sizeadj_1 = size_1 - excess_1;
  unsigned int sizeadj_2 = size_2 - excess_2;
  unsigned int size1=0, size2=0;
  T max = MAX(*(A_last-1),*(B_last-1));

  if(sizeadj_1 != size_1 && size1 >= sizeadj_1 && !flag1){//we need to start padding
                flag1 = 1;
                T* pad1=temp1;
                assert(posix_memalign((void**)&temp1,16,sizeof(T)*4)==0);
                int i;
                for(i=0;i<excess_1;i++)
                        temp1[i]=pad1[i];
                for(;i<4;i++)
                        temp1[i]=max;
        }
  if(sizeadj_2 != size_2 && size2 >= sizeadj_2 && !flag2){//we need to start padding
                flag2=1;
                T* pad2=temp2;
                assert(posix_memalign((void**)&temp2,16,sizeof(T)*4)==0);
                int i;
                for(i=0;i<excess_2;i++)
                        temp2[i]=pad2[i];
                for(;i<4;i++)
                        temp2[i]=max;
  }
  S1 = Traits<T>::_mm_load(temp1);
  S2 = Traits<T>::_mm_load(temp2);
  size1 = size2 = 4;
  temp1 += 4;temp2 += 4;

  while(1){
                S2 = Traits<T>::template _mm_shuffle<_MM_SHUFFLE(0,1,2,3)>(S2,S2);
                L1 = Traits<T>::_mm_min(S1,S2);
                H1 = Traits<T>::_mm_max(S1,S2);
                L1p = Traits<T>::template _mm_shuffle<_MM_SHUFFLE(1,0,1,0)>(L1,H1);
                H1p = Traits<T>::template _mm_shuffle<_MM_SHUFFLE(3,2,3,2)>(L1,H1);
                L2 = Traits<T>::_mm_min(L1p,H1p);
                H2 = Traits<T>::_mm_max(L1p,H1p);
                L2p = Traits<T>::template _mm_shuffle<_MM_SHUFFLE(2,0,2,0)>(L2,H2);
                L2p = Traits<T>::template _mm_shuffle<_MM_SHUFFLE(3,1,2,0)>(L2p,L2p);
                H2p = Traits<T>::template _mm_shuffle<_MM_SHUFFLE(3,1,3,1)>(L2,H2);
                H2p = Traits<T>::template _mm_shuffle<_MM_SHUFFLE(3,1,2,0)>(H2p,H2p);
                L3 = Traits<T>::_mm_min(L2p,H2p);
                H3 = Traits<T>::_mm_max(L2p,H2p);
                O1 = Traits<T>::_mm_unpacklo(L3,H3);
                O2 = Traits<T>::_mm_unpackhi(L3,H3);
                {
                        T* pDest_ = (T*)pDest;
                        if(pDest_-C_+4<=size_1+size_2)
                                Traits<T>::_mm_store(pDest,O1);
                        else{
                                T* O1_=(T*)&O1;
                                for(int i=0;i<4;i++)
                                if(&pDest_[i]-C_<size_1+size_2) pDest_[i]=O1_[i];
                        }
                }
                S1 = O2;

                /*Perform comparison between two elements of arrays to see which one should be loaded*/
                if (size1 >= size_1 && size2 >= size_2){
                        pDest++;
                        T* O2_=(T*)&O2;
                        T* pDest_ = (T*)pDest;
                        for(int i=0;i<4;i++)
                                if(&pDest_[i]-C_<size_1+size_2) pDest_[i]=O2_[i];
                        //Traits<T>::_mm_store(pDest,O2);
                        return;
                }
                if(sizeadj_1 != size_1 && size1 >= sizeadj_1 && !flag1){//we need to start padding
                        flag1 = 1;
                        T* pad1=temp1;
                        assert(posix_memalign((void**)&temp1,16,sizeof(T)*4)==0);
                        int i;
                        for(i=0;i<excess_1;i++)
                                temp1[i]=pad1[i];
                        for(;i<4;i++)
                                temp1[i]=max;
                }
                if(sizeadj_2 != size_2 && size2 >= sizeadj_2 && !flag2){//we need to start padding
                        flag2=1;
                        T* pad2=temp2;
                        assert(posix_memalign((void**)&temp2,16,sizeof(T)*4)==0);
                        int i;
                        for(i=0;i<excess_2;i++)
                                temp2[i]=pad2[i];
                        for(;i<4;i++)
                                temp2[i]=max;
                }
                if(size1 >= size_1){
                        S2 = Traits<T>::_mm_load(temp2);
                        size2 += 4;
                        temp2 += 4;
                }
                else if(size2 >= size_2){
                        S2 = Traits<T>::_mm_load(temp1);
                        size1 += 4;
                        temp1 += 4;
                }
                else{
                        if(*temp1 < *temp2){
                                S2 = Traits<T>::_mm_load(temp1);
                                size1 += 4;
                                temp1 += 4;
                        }
                        else{
                                S2 = Traits<T>::_mm_load(temp2);
                                size2 += 4;
                                temp2 += 4;
                        }
                }
                pDest++;
        }
}

template <class T>
void sse<T,8>::merge(T* A_, T* A_last,T* B_, T* B_last, T* C_){
  //c = Traits<T>::_mm_min(a,b);
  std::cout<<"size: 8\n"; //TODO: Your code here
}

