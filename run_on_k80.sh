#!/bin/bash
#SBATCH --job-name=diffusion_k80
#SBATCH --output=diffusion_k80_%j.out
#SBATCH --error=diffusion_k80_%j.err
#SBATCH --partition=k80    # Specify the partition or queue, adjust if necessary
#SBATCH --gres=gpu:1        # Request one K80 GPU
#SBATCH --time=01:00:00     # Set a limit to the run time
#SBATCH --nodes=1           # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of tasks per node

# Load the CUDA module, adjust version as necessary
module load CUDA/10.1.105

# Execute your compiled CUDA program
./diffusion_part_3
