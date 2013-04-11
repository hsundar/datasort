#!/bin/bash

export EXEC=bin/main
export WORK_DIR=$(dirname ${PWD}/$0)/..
cd ${WORK_DIR}

make -f Makefile_jaguar clean
make -f Makefile_jaguar ${EXEC} -j20
if [ ! -f ${EXEC} ] ; then exit -1; fi;
mkdir -p tmp
rm -r -f tmp/*

k=4;
j=$((27+$k*2));                 # Number of point sources is 2^j
export NODES=$((4**($k+2)));    # Number of compute nodes.
export CORES=1;                 # Number of cores per node.
export MPI_PROC=$((4**($k+2))); # Number of MPI processes.
export THREADS=1;               # Number of threads per MPI process.
export NUM_PTS=$((2**$j));      # Number of point sources/samples.

export FNAME=tmp/sort_jaguar_${j}_p${MPI_PROC}_t${THREADS}

qsub -l size=$(($NODES*$CORES)) -o ${FNAME}.out -e ${FNAME}.err ./scripts/jaguar.pbs
if (( $? != 0 )) ; then exit -1; fi;

while ((1)) ; do 
  clear; 
  qstat -u ${USER};
  cat ${FNAME}.* 
  sleep 1; 
done
