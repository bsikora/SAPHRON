#!/bin/csh
#$ -M vrathee@nd.edu
#$ -m abe
#$ -N lAPHRON
#$ -q long
#$ -pe smp 1

mpiexec -np $NSLOTS ./LAPHRON 1 in.polymer_new 10000