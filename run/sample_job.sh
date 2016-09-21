#!/bin/csh
#$ -M vrathee@nd.edu
#$ -m abe
#$ -N sammps
#$ -q long
#$ -pe smp 8

module load lammps 
mpirun -np $NSLOTS /afs/crc.nd.edu/user/v/vrathee/lammpsStuff/lammps-master/src/lmp_mpi < in.trial
