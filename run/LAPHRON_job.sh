#!/bin/csh
#$ -M vrathee@nd.edu
#$ -m abe
#$ -N lAPHRON
#$ -q long
#$ -pe smp 1

module purge
module load cmake/3.2.2            
module load gcc/4.9.2              
module load python/2.7.8           
module load boost/1.58             
module load ompi/1.8.7-gcc-4.4.7   
module load intel/15.0

mpiexec -np $NSLOTS ./LAPHRON 1 in.polymer_new 1