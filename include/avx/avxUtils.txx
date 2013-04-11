#include <cstdlib>
#include <iostream>
#include <cassert>

#define MAX(A_,B_) ((A_)>(B_) ? (A_) : (B_))
template <class T>
//#pragma intel optimization_parameter target_arch=avx
void avx<T,4>::merge(T* A_, T* A_last,T* B_, T* B_last, T* C_){
  //printf("Performing AVX merge for floats\n");
  //c = Traits<T>::_mm_min(a,b);
  //std::cout<<"size: 4\n"; //TODO: Your code here
  //float d = Traits<T>::_mm_min(A_[0],A_[1]);
  unsigned int size_1 = A_last-A_;
  unsigned int size_2 = B_last-B_;
  if(size_1==0) {memcpy(C_,B_,size_2*sizeof(T)); return;}
  if(size_2==0) {memcpy(C_,A_,size_1*sizeof(T)); return;}

  m256 L1,H1,L1p,H1p,L2,H2,L2p,H2p,L3,H3,L3p1,H3p1,L3p,H3p,L4,H4,L4p,H4p,O1,O2;
  m256 S1,S2;
  int flag1=0,flag2=0;
  m256* pDest = (m256*)C_;
  T *temp, *temp1=A_, *temp2=B_;
  unsigned int excess_1 = size_1 % 8;
  unsigned int excess_2 = size_2 % 8;
  unsigned int sizeadj_1 = size_1 - excess_1;
  unsigned int sizeadj_2 = size_2 - excess_2;
  unsigned int size1=0, size2=0;
  T max = MAX(*(A_last-1),*(B_last-1));

  if(sizeadj_1 != size_1 && size1 >= sizeadj_1 && !flag1){//we need to start padding
                flag1 = 1;
                T* pad1=temp1;
                assert(posix_memalign((void**)&temp1,32,sizeof(T)*8)==0);
                int i;
                for(i=0;i<excess_1;i++)
                        temp1[i]=pad1[i];
                for(;i<8;i++)
                        temp1[i]=max;
        }
  if(sizeadj_2 != size_2 && size2 >= sizeadj_2 && !flag2){//we need to start padding
                flag2=1;
                T* pad2=temp2;
                assert(posix_memalign((void**)&temp2,32,sizeof(T)*8)==0);
                int i;
                for(i=0;i<excess_2;i++)
                        temp2[i]=pad2[i];
                for(;i<8;i++)
                        temp2[i]=max;
  }
  S1 = Traits_avx<T>::_mm_load(temp1);
  S2 = Traits_avx<T>::_mm_load(temp2);
  size1 = size2 = 8;
  temp1 += 8;temp2 += 8;

  while(1){
				/*Need to reverse S2 registers*/
                S2 = Traits_avx<T>::template _mm_permute2<1>(S2,S2);
                S2 = Traits_avx<T>::template _mm_permute<27>(S2);
                
		L1 = Traits_avx<T>::_mm_min(S1,S2);
                H1 = Traits_avx<T>::_mm_max(S1,S2);
                
		L1p = Traits_avx<T>::template _mm_permute2<32>(L1,H1);
                H1p = Traits_avx<T>::template _mm_permute2<49>(L1,H1);

                L2 = Traits_avx<T>::_mm_min(L1p,H1p);
                H2 = Traits_avx<T>::_mm_max(L1p,H1p);
                
		L2p = Traits_avx<T>::template _mm_shuffle<68>(L2,H2);
                H2p = Traits_avx<T>::template _mm_shuffle<238>(L2,H2);
                
		L3 = Traits_avx<T>::_mm_min(L2p,H2p);
                H3 = Traits_avx<T>::_mm_max(L2p,H2p);

		L3p1 = Traits_avx<T>::_mm_unpacklo(L3,H3);
		H3p1 = Traits_avx<T>::_mm_unpackhi(L3,H3);

		L3p = Traits_avx<T>::template _mm_shuffle<68>(L3p1,H3p1);
		H3p = Traits_avx<T>::template _mm_shuffle<238>(L3p1,H3p1);
		
		L4 = Traits_avx<T>::_mm_min(L3p,H3p);
                H4 = Traits_avx<T>::_mm_max(L3p,H3p);

		L4p = Traits_avx<T>::_mm_unpacklo(L4,H4);
		H4p = Traits_avx<T>::_mm_unpackhi(L4,H4);

                O1 = Traits_avx<T>::template _mm_permute2<32>(L4p,H4p);
                O2 = Traits_avx<T>::template _mm_permute2<49>(L4p,H4p);
                {
                        T* pDest_ = (T*)pDest;
                        if(pDest_-C_+8<=size_1+size_2)
                                Traits_avx<T>::_mm_store(pDest,O1);
                        else{
                                T* O1_=(T*)&O1;
                                for(int i=0;i<8;i++)
                                if(&pDest_[i]-C_<size_1+size_2) pDest_[i]=O1_[i];
                        }
                }
                S1 = O2;

                /*Perform comparison between two elements of arrays to see which one should be loaded*/
                if (size1 >= size_1 && size2 >= size_2){
                        pDest++;
                        T* O2_=(T*)&O2;
                        T* pDest_ = (T*)pDest;
                        for(int i=0;i<8;i++)
                                if(&pDest_[i]-C_<size_1+size_2) pDest_[i]=O2_[i];
                        //Traits_avx<T>::_mm_store(pDest,O2);
                        return;
                }
                if(sizeadj_1 != size_1 && size1 >= sizeadj_1 && !flag1){//we need to start padding
                        flag1 = 1;
                        T* pad1=temp1;
                        assert(posix_memalign((void**)&temp1,32,sizeof(T)*8)==0);
                        int i;
                        for(i=0;i<excess_1;i++)
                                temp1[i]=pad1[i];
                        for(;i<8;i++)
                                temp1[i]=max;
                }
                if(sizeadj_2 != size_2 && size2 >= sizeadj_2 && !flag2){//we need to start padding
                        flag2=1;
                        T* pad2=temp2;
                        assert(posix_memalign((void**)&temp2,32,sizeof(T)*8)==0);
                        int i;
                        for(i=0;i<excess_2;i++)
                                temp2[i]=pad2[i];
                        for(;i<8;i++)
                                temp2[i]=max;
                }
                if(size1 >= size_1){
                        S2 = Traits_avx<T>::_mm_load(temp2);
                        size2 += 8;
                        temp2 += 8;
                }
                else if(size2 >= size_2){
                        S2 = Traits_avx<T>::_mm_load(temp1);
                        size1 += 8;
                        temp1 += 8;
                }
                else{
                        if(*temp1 < *temp2){
                                S2 = Traits_avx<T>::_mm_load(temp1);
                                size1 += 8;
                                temp1 += 8;
                        }
                        else{
                                S2 = Traits_avx<T>::_mm_load(temp2);
                                size2 += 8;
                                temp2 += 8;
                        }
                }
                pDest++;
        }
}

//#pragma intel optimization_parameter target_arch=avx
template <class T>
void avx<T,8>::merge(T* A_, T* A_last,T* B_, T* B_last, T* C_){
  //std::cout<<"Performing AVX merge on doubles!\n"; //TODO: Your code here
  //c = Traits<T>::_mm_min(a,b);
  //std::cout<<"size: 4\n"; //TODO: Your code here
  //float d = Traits<T>::_mm_min(A_[0],A_[1]);
  unsigned int size_1 = A_last-A_;
  unsigned int size_2 = B_last-B_;
  if(size_1==0) {memcpy(C_,B_,size_2*sizeof(T)); return;}
  if(size_2==0) {memcpy(C_,A_,size_1*sizeof(T)); return;}

  m256 L1,H1,L1p,H1p,L2,H2,L2p,H2p,L3,H3,L3p,H3p,O1,O2;
  m256 S1,S2;
  int flag1=0,flag2=0;
  m256* pDest = (m256*)C_;
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
                assert(posix_memalign((void**)&temp1,32,sizeof(T)*4)==0);
                int i;
                for(i=0;i<excess_1;i++)
                        temp1[i]=pad1[i];
                for(;i<4;i++)
                        temp1[i]=max;
        }
  if(sizeadj_2 != size_2 && size2 >= sizeadj_2 && !flag2){//we need to start padding
                flag2=1;
                T* pad2=temp2;
                assert(posix_memalign((void**)&temp2,32,sizeof(T)*4)==0);
                int i;
                for(i=0;i<excess_2;i++)
                        temp2[i]=pad2[i];
                for(;i<4;i++)
                        temp2[i]=max;
  }
  S1 = Traits_avx<T>::_mm_load(temp1);
  S2 = Traits_avx<T>::_mm_load(temp2);
  size1 = size2 = 4;
  temp1 += 4;temp2 += 4;

  while(1){
				/*Need to reverse S2 registers*/
                S2 = Traits_avx<T>::template _mm_permute2<1>(S2,S2);
                S2 = Traits_avx<T>::template _mm_permute<5>(S2);
                
		L1 = Traits_avx<T>::_mm_min(S1,S2);
                H1 = Traits_avx<T>::_mm_max(S1,S2);
                
		L1p = Traits_avx<T>::template _mm_permute2<32>(L1,H1);
                H1p = Traits_avx<T>::template _mm_permute2<49>(L1,H1);

                L2 = Traits_avx<T>::_mm_min(L1p,H1p);
                H2 = Traits_avx<T>::_mm_max(L1p,H1p);
                
		L2p = Traits_avx<T>::template _mm_shuffle<0>(L2,H2);
                H2p = Traits_avx<T>::template _mm_shuffle<15>(L2,H2);
                
		L3 = Traits_avx<T>::_mm_min(L2p,H2p);
                H3 = Traits_avx<T>::_mm_max(L2p,H2p);

		L3p = Traits_avx<T>::_mm_unpacklo(L3,H3);
		H3p = Traits_avx<T>::_mm_unpackhi(L3,H3);

                O1 = Traits_avx<T>::template _mm_permute2<32>(L3p,H3p);
                O2 = Traits_avx<T>::template _mm_permute2<49>(L3p,H3p);
                {
                        T* pDest_ = (T*)pDest;
                        if(pDest_-C_+4<=size_1+size_2)
                                Traits_avx<T>::_mm_store(pDest,O1);
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
                        assert(posix_memalign((void**)&temp1,32,sizeof(T)*4)==0);
                        int i;
                        for(i=0;i<excess_1;i++)
                                temp1[i]=pad1[i];
                        for(;i<4;i++)
                                temp1[i]=max;
                }
                if(sizeadj_2 != size_2 && size2 >= sizeadj_2 && !flag2){//we need to start padding
                        flag2=1;
                        T* pad2=temp2;
                        assert(posix_memalign((void**)&temp2,32,sizeof(T)*4)==0);
                        int i;
                        for(i=0;i<excess_2;i++)
                                temp2[i]=pad2[i];
                        for(;i<4;i++)
                                temp2[i]=max;
                }
                if(size1 >= size_1){
                        S2 = Traits_avx<T>::_mm_load(temp2);
                        size2 += 4;
                        temp2 += 4;
                }
                else if(size2 >= size_2){
                        S2 = Traits_avx<T>::_mm_load(temp1);
                        size1 += 4;
                        temp1 += 4;
                }
                else{
                        if(*temp1 < *temp2){
                                S2 = Traits_avx<T>::_mm_load(temp1);
                                size1 += 4;
                                temp1 += 4;
                        }
                        else{
                                S2 = Traits_avx<T>::_mm_load(temp2);
                                size2 += 4;
                                temp2 += 4;
                        }
                }
                pDest++;
        }
}
/*template <class T>
void avx<T,8>::merge(T* A_, T* A_last,T* B_, T* B_last, T* C_){
  //c = Traits<T>::_mm_min(a,b);
  std::cout<<"size: 8\n"; //TODO: Your code here
}*/
