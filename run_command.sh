#!/bin/bash

# Set the values for N and stride (s)
N_max=101
N_min=10
N_steps=10
N_p=2
s=10

for ((i=N_min; i<=N_max; i+=s)); do
    echo "Running python main_MaZe.py for N = $i and N_steps = $N_steps"
    python main_Maze.py  $N_p $i $N_steps > data_md/test/output_N$i.out
    
    echo "Iteration $i finished"
done

echo "All iterations completed"
