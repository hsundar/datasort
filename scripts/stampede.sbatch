#!/bin/bash
#SBATCH -J par_sort
# SBATCH --mail-type=end
# SBATCH --mail-user=<email_address>@<domain>
#SBATCH --export=ALL
#SBATCH --time=00:05:00
 
WORK_DIR=${PWD}
export TMPDIR=${WORK_DIR}/tmp

export OMP_NUM_THREADS=${THREADS}
time ibrun tacc_affinity ${EXEC} ${THREADS} l128MB UNIF

