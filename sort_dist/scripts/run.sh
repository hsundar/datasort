#!/bin/bash

export EXEC=bin/main
export WORK_DIR=$(dirname ${PWD}/$0)/..
cd ${WORK_DIR}

mkdir -p tmp
make -f Makefile_ronaldo clean
make -f Makefile_ronaldo ${EXEC} -j8
if [ ! -f ${EXEC} ] ; then exit -1; fi;
mkdir -p tmp
rm -r -f tmp/*

export NODES=6;            # Number of compute nodes.
export CORES=12;           # Number of cores per node.
export MPI_PROC=64;        # Number of MPI processes.
export THREADS=1;          # Number of threads per MPI process.

FILE_OUT='tmp/main.out'
FILE_ERR='tmp/main.err'

#Submit Job
qsub -l nodes=${NODES}:ppn=$((${MPI_PROC}/${NODES})) -o ${FILE_OUT} -e ${FILE_ERR} ./scripts/run.pbs
if (( $? != 0 )) ; then exit -1; fi;

#Wait for job to finish and display output
while [ 1 ]
do
    if [ -f $FILE_OUT ]; then
      if [ -f $FILE_ERR ]; then
        cat tmp/main.out
        cat tmp/main.err
        break;
      fi
    fi
    sleep 1;
done

