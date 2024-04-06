1. module load the following
    GCC/7.3.0-2.30
    GNU/7.3.0-2.30
    CUDA/9.2.88
2. nvcc diffusion.cu -o diffusion
3. ./diffusion blocksize nsteps
    for example: ./diffusion 256 100
4. make sure a folder named "data" exists
5. python3 plotting.py to plot