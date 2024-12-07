#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00
#SBATCH --mem=30G
#SBATCH --account=cecam_premium
#SBATCH --output="log_N100_%j.out"
#SBATCH --mail-user=federica.troni@epfl.ch
#SBATCH --mail-type=END

# Load necessary modules
module load gcc mvapich2 libxc fftw openblas netlib-scalapack

# Compile the shared library
gcc -O3 --fast-math -shared ../laplace.c -o ../maze_poisson/liblaplace.so -fopenmp -fPIC

# Set the number of OpenMP threads dynamically
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the code
maze run main-maze-md input_N100.yml
