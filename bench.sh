#!/bin/bash
#SBATCH --job-name=nbody
#SBATCH --error=nbody_%j.err
#SBATCH --partition=GPU
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

./GPU/nbody_gpu 100000 0.01 50 101 128
./GPU/nbody_gpu 10000 0.01 50 101 128
./GPU/nbody_gpu 1000 0.01 50 101 128
./sequential/nbody 100000 0.01 50 101
./sequential/nbody 10000 0.01 50 101
./sequential/nbody 1000 0.01 50 101
