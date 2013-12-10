#ifndef __SORT_RECORD_H_
#define __SORT_RECORD_H_
#include <iostream>

class sortRecord {
private:
  /*
  long      eka;
  short     dva;
  */
  char      key[10];
  char      value[90];

public:
  sortRecord() {
    // key   = new char[10];
    // value = new char[90]; 
  }
  
  sortRecord(const sortRecord& other) {
    // key   = new char[10];
    // value = new char[90];

    memcpy(key,   other.key,   10);
    memcpy(value, other.value, 90);
  }

  // ks (4/11/13) - add ctor from raw 100-byte buffer

  sortRecord(unsigned char *buffer)
    {
      memcpy(key,  &buffer[0], 10);
      memcpy(value,&buffer[10],90);
    }

  sortRecord& operator = ( const sortRecord &other) {
    if (this != &other) // protect against invalid self-assignment
    {
      memcpy(key,   other.key,   10);
      memcpy(value, other.value, 90);
    }
    return *this;
  }
  ~sortRecord() {
    /*
    if (key != NULL) {
      delete [] key; key = NULL;
    }
    if (value != NULL) {
      delete [] value; value = NULL;
    }
    */
  }

  static sortRecord random() {
    sortRecord r;
    unsigned int   * ip;
    unsigned short * sp = (unsigned short*)(r.key);
    ip = (unsigned int *)(&(r.key[2]));
    *ip = rand();
    ip++;
    *ip = rand();
    *sp = rand()%(1<<16);
    
    return r;
  }

  // ks (4/11/13) - additional convenience function to generate
  // sortRecord from raw data

  static sortRecord fromBuffer(const unsigned char *buffer)
  {
    sortRecord record;
    memcpy(record.key,  &buffer[0] ,10);
    memcpy(record.value,&buffer[10],90);

    return(record);
  }

  // TODO : optimize using SIMD
  bool  operator == ( sortRecord const  &other) const {
    return (memcmp(this->key, other.key, 10) == 0);
    // return ( (this->eka == other.eka) && (this->dva == other.dva) );
    // return ( (*(unsigned long*)(this->key) == *(unsigned long*)(other.key)) &&  (*(unsigned short*)(this->key+8) == *(unsigned short*)(other.key+8)) );
  }
  bool  operator < ( sortRecord const  &other) const {
    return (memcmp(this->key, other.key, 10) < 0);
    
    /*
    if (*(long*)(this->key) == *(long*)(other.key)) 
      return (*(unsigned short*)(this->key+8) < *(unsigned short*)(other.key+8)); 
    else
      return (*(unsigned long*)(this->key) < *(unsigned long*)(other.key)); 
    if (this->eka == other.eka)
      return this->dva < other.dva;
    else
      return this->eka < other.eka;
      */
  }
  bool  operator <= ( sortRecord const  &other) const {
    return (memcmp(this->key, other.key, 10) <= 0);
  }
  bool  operator > ( sortRecord const  &other) const {
    return (memcmp(this->key, other.key, 10) > 0);
  }
  bool  operator >= ( sortRecord const  &other) const {
    return (memcmp(this->key, other.key, 10) >= 0);
  }
  bool  operator != ( sortRecord const  &other) const {
    return (memcmp(this->key, other.key, 100) != 0);
  }
  friend std::ostream& operator<<(std::ostream& os, const sortRecord& r1){
    os << r1.key << ' ' << r1.value << '\n';
    return os;
  }
}; 


namespace par {

  //Forward Declaration
  template <typename T>
    class Mpi_datatype;

  template <>
    class Mpi_datatype< sortRecord > {

      public:

      /**
       @return The MPI_Datatype corresponding to the datatype "sortRecord".
     */
      static MPI_Datatype value()
      {
        static bool         first = true;
        static MPI_Datatype datatype;

        if (first)
        {
          first = false;
          MPI_Type_contiguous(sizeof(sortRecord), MPI_BYTE, &datatype);
          MPI_Type_commit(&datatype);
        }

        return datatype;
      }

    };

}//end namespace par


#endif
