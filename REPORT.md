1. module load GCC/7.3.0-2.30 GNU/7.3.0-2.30 CUDA/9.2.88
2. nvcc diffusion.cu -o diffusion
3. ./diffusion blocksize nsteps
    for example: ./diffusion 256 100
4. make sure a folder named "data" exists
5. python3 plotting.py to plot


# Report: Project 5

1. Report your timings for the host, naive CUDA kernel, shared memory CUDA kernel, and the excessive memory copying case, using block dimensions of 256, 512, and 1024. Use a grid size of 2^15+2*NG (or larger) and run for 100 steps (or shorter, if it's taking too long). Remember to use -O3!

2. How do the GPU implementations compare to the single threaded host code. Is it faster than the theoretical performance of the host if we used all the cores on the CPU?

3. For the naive kernel, the shared memory kernel, and the excessive memcpy case, which is the slowest? Why? How might you design a larger code to avoid this slow down?

4. Do you see a slow down when you increase the block dimension? Why? Consider that multiple blocks may run on a single multiprocessor simultaneously, sharing the same shared memory.