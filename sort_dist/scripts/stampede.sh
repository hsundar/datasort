#!/bin/bash

export EXEC=bin/main
export WORK_DIR=$(dirname ${PWD}/$0)/..
cd ${WORK_DIR}

mkdir -p tmp
make -f Makefile_stampede clean
make -f Makefile_stampede ${EXEC} -j8
if [ ! -f ${EXEC} ] ; then exit -1; fi;
mkdir -p tmp
rm -r -f tmp/*

export NODES=256;           # Number of compute nodes.
export CORES=16;           # Number of cores per node.
export MPI_PROC=4096;       # Number of MPI processes.
export THREADS=1;          # Number of threads per MPI process.
export NUM_PTS=1000000;    # Number of point sources/samples.

FILE_OUT='tmp/main.out'
FILE_ERR='tmp/main.err'

#Submit Job
export OMP_NUM_THREADS=${THREADS}
sbatch -A seBiros -p normal -N${NODES} -n${MPI_PROC} -o ${FILE_OUT} -e ${FILE_ERR} -D ${PWD} ./scripts/stampede.sbatch
#sbatch -A seBiros -p normal -N${NODES} -n$(( ${MPI_PROC}*${THREADS} )) -o ${FILE_OUT} -e ${FILE_ERR} -D ${PWD} ./scripts/stampede.sbatch
if (( $? != 0 )) ; then exit -1; fi;

#Wait for job to finish and display output
while [ 1 ]
do
    clear;
    squeue -u dmalhotr
    if [ -f $FILE_OUT ]; then
      if [ -f $FILE_ERR ]; then
        cat tmp/main.out
        #cat tmp/main.err
        #break;
      fi
    fi
    sleep 1;
done

