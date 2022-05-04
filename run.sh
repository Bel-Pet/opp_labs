#!/bin/bash
#PBS -l walltime=00:05:00
#PBS -l select=2:ncpus=8:mpiprocs=4
#PBS -m n
cd $PBS_O_WORKDIR
echo
mpirun -machinefile $PBS_NODEFILE -np 16 ./ex
