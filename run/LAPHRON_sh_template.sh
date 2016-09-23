#!/bin/csh
#$ -M vrathee@nd.edu
#$ -m abe
#$ -N lAPHRON
#$ -t 1-10
#$ -r y
#$ -q long
#$ -pe smp 1

module purge
module load cmake/3.2.2            
module load gcc/4.9.2              
module load python/2.7.8           
module load boost/1.58             
module load ompi/1.8.7-gcc-4.4.7   
module load intel/15.0

set args = `head -n $SGE_TASK_ID args.list | tail -n 1`
fsync $SGE_STDOUT_PATH &
mpiexec -np $NSLOTS ./LAPHRON 1 $args
