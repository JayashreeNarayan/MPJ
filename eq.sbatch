#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00
#SBATCH --mem=30G
#SBATCH --account=cecam_premium
#SBATCH --output="log_N100_%j.out"
#SBATCH --mail-user=jayashree.narayan@epfl.ch
#SBATCH --mail-type=END

# Load necessary modules
module load gcc libxc openmpi fftw openblas netlib-scalapack

# Compile the shared library
gcc -O3 --fast-math -shared laplace.c -o maze_poisson/liblaplace.so -fopenmp -fPIC

# Set the number of OpenMP threads dynamically
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the code
source venv/bin/activate maze run main-maze-md ../../examples/test_input_density_1.379.yml

