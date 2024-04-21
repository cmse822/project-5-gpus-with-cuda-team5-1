/// Diffusion for part 3

// Remember to properly set up the environment

#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <cmath>
#include <cassert>
#include "get_walltime.h"

using namespace std;

// Number of ghost cells to use
const unsigned int NG = 2;

// Block sizes
//const unsigned int BLOCK_DIM_X = 128;
//const unsigned int BLOCK_DIM_X = 256;
//const unsigned int BLOCK_DIM_X = 512;
//const unsigned int BLOCK_DIM_X = 1024;
//const unsigned int BLOCK_DIM_X = 2048;
const unsigned int BLOCK_DIM_X = 4096;


// Finite difference weights stored in device constant memory
__constant__ float c_a, c_b, c_c;

/******************************************************************************
  Error checking function for CUDA
 *****************************************************************************/
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
//https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/
//        finite-difference/finite-difference.cu
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n",
                cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

/******************************************************************************
  Do one diffusion step, on the host in host memory
 *****************************************************************************/
void host_diffusion(float* u, float *u_new, const unsigned int n,
                    const float dx, const float dt)
{
  //First, do the diffusion step on the interior points
    for(int i = NG; i < n-NG;i++){
        u_new[i] = u[i] + dt/(dx*dx) *(
                        - 1./12.f* u[i-2]
                        + 4./3.f * u[i-1]
                        - 5./2.f * u[i]
                        + 4./3.f * u[i+1]
                        - 1./12.f* u[i+2]);
    }

    //Apply the dirichlet boundary conditions
    u_new[0] = -u_new[NG+1];
    u_new[1] = -u_new[NG];

    u_new[n-NG]   = -u_new[n-NG-1];
    u_new[n-NG+1] = -u_new[n-NG-2];
}
/******************************************************************************
  Do one diffusion step, with CUDA
 *****************************************************************************/
__global__ 
void cuda_diffusion(float* u, float *u_new, const unsigned int n)
{
    //Do the diffusion
    //FIXME

    // Global index in `u` array
    int i = blockIdx.x*blockDim.x + threadIdx.x + 2;

    // Perform the finite difference; negative signs are included in the
    // constant terms
    u_new[i] = u[i] + ( c_a*u[i-2]
                       +c_b*u[i-1]
                       +c_c*u[i]
                       +c_b*u[i+1]
                       +c_a*u[i+2]);

    //Apply the dirichlet boundary conditions
    //HINT: Think about which threads will have the data for the boundaries
    //FIXME

    // The first two cells, indices 2 and 3, will each fill their corresponding
    // ghost cell with the negative of their own value
    if(i < 4)
        u_new[(i + 1)%2] = -u_new[i];

    // The last two cells, indices n-4 and n-3, will each fill their
    // corresponding ghost cell with the negative of their own value
    else if(i >= n - 4)
        u_new[2*(n - NG) - (i + 1)] = -u_new[i];
}

/******************************************************************************
  Do one diffusion step, with CUDA, with shared memory
 *****************************************************************************/
__global__ 
void shared_diffusion(float* u, float *u_new, const unsigned int n)
{
    //Allocate the shared memory
    //FIXME

    // Since we know how many elements are in each block, we can statically
    // allocated the shared memory for this block.  The array needs its own
    // four ghost cells that will be filled with the global elements immediate
    // to the left and right of the blocks sections of the global array.
    __shared__ float s_u[BLOCK_DIM_X + 4];

    // Width of this block
    int nx = blockDim.x;

    // Local index for shared memory
    int si = threadIdx.x + 2;

    // Global index
    int gi = blockIdx.x*nx + si;

    //Fill shared memory with the data needed from global memory
    //HINT: 
    //What data does each block need from global memory?
    //When do the threads in the block need to sync?
    //FIXME

    // Fill each local element with its corresponding global element
    s_u[si] = u[gi];

    // The left two most ghost cells need the two global elements immediately
    // to the left; since the input global array has the boundary ghost cells
    // filled no extra steps need to be taken to fill these correctly.
    if(si < 2*NG)
        s_u[si - NG] = u[gi - NG];
    // Same for right two ghost cells
    else if(si >= nx - 2*NG)
        s_u[si + NG] = u[gi + NG];

    // Make sure all threads have completed filling the elements of shared
    // memory that they are responsible for
    __syncthreads();

    //Do the diffusion
    //FIXME

    // Same finite difference stencil as before, but not pulls from shared
    // memory and stores in global
    u_new[gi] = s_u[si] + ( c_a*s_u[si-2]
                          + c_b*s_u[si-1]
                          + c_c*s_u[si]
                          + c_b*s_u[si+1]
                          + c_a*s_u[si+2]);

    //Apply the dirichlet boundary conditions
    //HINT: Think about which threads will have the data for the boundaries
    //FIXME

    // Update the global boundary ghost cells in the output array
    if(gi < 2*NG)
        u_new[(gi + 1)%2] = -u_new[gi];
    if(gi >= n - 2*NG)
        u_new[2*(n - NG)-(gi + 1)] = -u_new[gi];
}

/******************************************************************************
  Dump u to a file
 *****************************************************************************/
void outputToFile(string filename, float* u, unsigned int n)
{
    ofstream file;
    file.open(filename.c_str());
    file.precision(8);
    file << std::scientific;
    for(int i =0; i < n;i++){
        file<<u[i]<<endl;
    }
    file.close();
};

/******************************************************************************
  main
 *****************************************************************************/
int main(int argc, char** argv)
{
    //Number of steps to iterate
    //const unsigned int n_steps = 10;
    const unsigned int n_steps = 10000; // -- going with this for part 3

    //const unsigned int n_steps = 1000000;

    //Whether and how ow often to dump data
    //const bool outputData = true;
    const bool outputData = true;
    const unsigned int outputPeriod = n_steps/10;

    //Size of u
    //const unsigned int n = (1<<11) +2*NG;
    const unsigned int n = (1<<15) +2*NG; // using this for part 3
    //const unsigned int n = (1<<20) +2*NG;

    //Block and grid dimensions
    const unsigned int blockDim = BLOCK_DIM_X;
    //const unsigned int gridDim = (n-2*NG)/blockDim;
    const unsigned int gridDim = (n + BLOCK_DIM_X - 1 - 2*NG) / BLOCK_DIM_X;


    //Physical dimensions of the domain
    const float L = 2*M_PI;
    const float dx = L/(n-2*NG-1);
    const float dt = 0.25*dx*dx;

    //Create constants for 6th order centered 2nd derivative
    // NOTE: I added the appropriate negative signs here
    float const_a = -1./12.f * dt/(dx*dx);  
    float const_b = 4./3.f  * dt/(dx*dx);
    float const_c = -5./2.f  * dt/(dx*dx);

    //Copy these the cuda constant memory
    //FIXME

    // Copy each finite difference weight to the device constants
    checkCuda(cudaMemcpyToSymbol(c_a, &const_a, sizeof(float),
                                 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(c_b, &const_b, sizeof(float),
                                 0, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpyToSymbol(c_c, &const_c, sizeof(float),
                                 0, cudaMemcpyHostToDevice));

    //iterator, for later
    int i;

    //Create cuda timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Timing variables
    float milliseconds;
    double startTime,endTime;

    //Filename for writing
    char filename[256];

    //Allocate memory for the initial conditions
    float* initial_u = new float[n];

    //Initialize with a periodic sin wave that starts after the left hand
    //boundaries and ends just before the right hand boundaries
    for( i = NG; i < n-NG; i++)
    {
        initial_u[i] = sin( 2*M_PI/L*(i-NG)*dx);
    }
    //Apply the dirichlet boundary conditions
    initial_u[0] = -initial_u[NG+1];
    initial_u[1] = -initial_u[NG];

    initial_u[n-NG]   = -initial_u[n-NG-1];
    initial_u[n-NG+1] = -initial_u[n-NG-2];

    /**************************************************************************
    Test the host kernel for diffusion
    **************************************************************************/

    //Allocate memory in the host's heap
    float* host_u  = new float[n];
    float* host_u2 = new float[n];//buffer used for u_new

    //Initialize the host memory
    for( i = 0; i < n; i++)
    {
        host_u[i] = initial_u[i];
    }

    outputToFile("data_blockdim_128/host_uInit.dat",host_u,n);
    //outputToFile("data_blockdim_256/host_uInit.dat",host_u,n);
    //outputToFile("data_blockdim_512/host_uInit.dat",host_u,n);
    //outputToFile("data_blockdim_1024/host_uInit.dat",host_u,n);


    get_walltime(&startTime);
    //Perform n_steps of diffusion
    for( i = 0 ; i < n_steps; i++){

       // if(outputData && i%outputPeriod == 0)
       // {
       //     sprintf(filename, "data/host_u%08d.dat", i);
       //     outputToFile(filename,host_u,n);
       // }

        if(outputData && (i % outputPeriod == 0 || i == n_steps - 1)) {
          //sprintf(filename, "data_blockdim_128/host_u%08d.dat", i);
          sprintf(filename, "data_blockdim_256/host_u%08d.dat", i);
          //sprintf(filename, "data_blockdim_512/host_u%08d.dat", i);
          //sprintf(filename, "data_blockdim_1024/host_u%08d.dat", i);

          outputToFile(filename, host_u, n);
        }
        

        host_diffusion(host_u, host_u2, n, dx, dt);

        //Switch the buffer with the original u
        float* tmp = host_u;
        host_u = host_u2;
        host_u2 = tmp;

    }
    get_walltime(&endTime);

    cout << "Host function took: " << (endTime-startTime)*1000./n_steps 
         << " ms per step" << endl;

    //outputToFile("data_blockdim_128/host_uFinal.dat", host_u, n);
    outputToFile("data_blockdim_256/host_uFinal.dat", host_u, n);
    //outputToFile("data_blockdim_512/host_uFinal.dat", host_u, n);
    //outputToFile("data_blockdim_1024/host_uFinal.dat", host_u, n);

    /**************************************************************************
    Test the cuda kernel for diffusion
    **************************************************************************/
    //Allocate a copy for the GPU memory in the host's heap
    float* cuda_u  = new float[n];

    //Initialize the cuda memory
    for( i = 0; i < n; i++)
    {
        cuda_u[i] = initial_u[i];
    }
    //outputToFile("data_blockdim_128/cuda_uInit.dat", cuda_u, n);
    outputToFile("data_blockdim_256/cuda_uInit.dat", cuda_u, n);
    //outputToFile("data_blockdim_512/cuda_uInit.dat", cuda_u, n);
    //outputToFile("data_blockdim_1024/cuda_uInit.dat", cuda_u, n);

    //Allocate memory on the GPU
    float* d_u, *d_u2;
    //FIXME Allocate d_u,d_u2 on the GPU, and copy cuda_u into d_u

    // Allocate the input and output buffers on the device
    checkCuda(cudaMalloc((void**)&d_u, n*sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_u2,n*sizeof(float)));

    // Copy the initial data (with filled ghost zones) to the device
    checkCuda(cudaMemcpy(d_u, cuda_u, n*sizeof(float),
                         cudaMemcpyHostToDevice));

    cudaEventRecord(start);//Start timing
    //Perform n_steps of diffusion
    for( i = 0 ; i < n_steps; i++)
    {
        if(outputData && i%outputPeriod == 0)
        {
            //Copy data off the device for writing
            //sprintf(filename, "data_blockdim_128/cuda_u%08d.dat", i);
            sprintf(filename, "data_blockdim_256/cuda_u%08d.dat", i);
            //sprintf(filename, "data_blockdim_512/cuda_u%08d.dat", i);
            //sprintf(filename, "data_blockdim_1024/cuda_u%08d.dat", i);
            
            //FIXME

            // Copy the current solution back to the host
            checkCuda(cudaMemcpy(cuda_u, d_u, n*sizeof(float), 
                                 cudaMemcpyDeviceToHost));

            outputToFile(filename,cuda_u,n);
        }

        //Call the cuda_diffusion kernel
        //FIXME

        // Invoke the finite difference kernel on the GPU
        cuda_diffusion<<<gridDim, blockDim>>>(d_u, d_u2, n);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error in cuda_diffusion kernel: " << cudaGetErrorString(error) << std::endl;
        }
        //Switch the buffer with the original u
        //FIXME

        // Only need to swap the pointers so the device knows which buffer to
        // read from and which to write to
        float* tmp = d_u;
        d_u = d_u2;
        d_u2 = tmp;
    }
    cudaEventRecord(stop);//End timing

    //Copy the memory back for one last data dump
    //sprintf(filename, "data_blockdim_128/cuda_u%08d.dat", i);
    sprintf(filename, "data_blockdim_256/cuda_u%08d.dat", i);
    //sprintf(filename, "data_blockdim_512/cuda_u%08d.dat", i);
    //sprintf(filename, "data_blockdim_1024/cuda_u%08d.dat", i);

    //FIXME

    // Copy the final solution back to the host
    checkCuda(cudaMemcpy(cuda_u, d_u, n*sizeof(float),
                         cudaMemcpyDeviceToHost));

    //outputToFile("data_blockdim_128/cuda_uFinal.dat",cuda_u,n);
    outputToFile("data_blockdim_256/cuda_uFinal.dat",cuda_u,n);
    //outputToFile("data_blockdim_512/cuda_uFinal.dat",cuda_u,n);
    //outputToFile("data_blockdim_1024/cuda_uFinal.dat",cuda_u,n);

    //Get the total time used on the GPU
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Cuda Kernel took: " << milliseconds/n_steps 
         << " ms per step" << endl;


    /**************************************************************************
    Test the cuda kernel for diffusion with shared memory
    **************************************************************************/

    //Allocate a copy for the GPU memory in the host's heap
    float* shared_u  = new float[n];

    //Initialize the cuda memory
    for( i = 0; i < n; i++)
    {
        shared_u[i] = initial_u[i];
    }
    //outputToFile("data_blockdim_128/shared_uInit.dat",shared_u,n);
    outputToFile("data_blockdim_256/shared_uInit.dat",shared_u,n);
    //outputToFile("data_blockdim_512/shared_uInit.dat",shared_u,n);
    //outputToFile("data_blockdim_1024/shared_uInit.dat",shared_u,n);

    //Copy the initial memory onto the GPU
    //FIXME copy shared_u to d_u

    // Copy the initial data to the device
    checkCuda(cudaMemcpy(d_u, shared_u, n*sizeof(float),
                         cudaMemcpyHostToDevice));


    cudaEventRecord(start);//Start timing
    //Perform n_steps of diffusion
    for( i = 0 ; i < n_steps; i++)
    {
        if(outputData && i%outputPeriod == 0)
        {
            //Copy data off the device for writing
            //sprintf(filename,"data_blockdim_128/shared_u%08d.dat",i);
            sprintf(filename,"data_blockdim_256/shared_u%08d.dat",i);
            //sprintf(filename,"data_blockdim_512/shared_u%08d.dat",i);
            //sprintf(filename,"data_blockdim_1024/shared_u%08d.dat",i);

            //FIXME

            // Copy the current solution back to the host
            checkCuda(cudaMemcpy(shared_u, d_u, n*sizeof(float),
                                 cudaMemcpyDeviceToHost));

            outputToFile(filename,shared_u,n);
        }

        //Call the shared_diffusion kernel
        //FIXME

        // Invoke the shared diffusion kernel on the GPU
        shared_diffusion<<<gridDim, blockDim>>>(d_u, d_u2, n);


        //Switch the buffer with the original u
        //FIXME

        // Only need to swap the pointers
        float* tmp = d_u;
        d_u = d_u2;
        d_u2 = tmp;

    }
    cudaEventRecord(stop);//End timing

    //Copy the memory back for one last data dump
    //sprintf(filename,"data_blockdim_128/shared_u%08d.dat",i);
    sprintf(filename,"data_blockdim_256/shared_u%08d.dat",i);
    //sprintf(filename,"data_blockdim_512/shared_u%08d.dat",i);
    //sprintf(filename,"data_blockdim_1024/shared_u%08d.dat",i);

    //FIXME

    // Copy final solution back to the device
    checkCuda(cudaMemcpy(shared_u, d_u, n*sizeof(float),
                         cudaMemcpyDeviceToHost));

    //Get the total time used on the GPU
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Shared Memory Kernel took: " << milliseconds/n_steps
         << " ms per step" << endl;

    /**************************************************************************
    Test the cuda kernel for diffusion, with excessive memcpys
    **************************************************************************/

    //Initialize the cuda memory
    for( i = 0; i < n; i++)
    {
        shared_u[i] = initial_u[i];
    }

    cudaEventRecord(start);//Start timing
    //Perform n_steps of diffusion
    for( i = 0 ; i < n_steps; i++)
    {
        //Copy the data from host to device
        //FIXME copy shared_u to d_u

        // Copy last solution data from the host back to the device
        checkCuda(cudaMemcpy(d_u, shared_u, n*sizeof(float),
                             cudaMemcpyHostToDevice));

        //Call the shared_diffusion kernel
        //FIXME

        // Invoke the shared diffusion kernel on the GPU
        shared_diffusion<<<gridDim, blockDim>>>(d_u, d_u2, n);

        //Copy the data from host to device
        //FIXME copy d_u2 to cuda_u

        // Copy the solution back to the host
        checkCuda(cudaMemcpy(shared_u, d_u2, n*sizeof(float),
                             cudaMemcpyDeviceToHost));
    }
    cudaEventRecord(stop);//End timing

    //Get the total time used on the GPU
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "Excessive cudaMemcpy took: " << milliseconds/n_steps 
         << " ms per step" << endl;

    //Clean up the data
    delete[] initial_u;
    delete[] host_u;
    delete[] host_u2;

    delete[] cuda_u;
    delete[] shared_u;

    //FIXME free d_u and d_2

    // De-allocate the memory on the device
    checkCuda( cudaFree(d_u) );
    checkCuda( cudaFree(d_u2) );
}

/////////////////// HOW TO RUN with GPU for cuda and shared parts: //////////////////////
// 1. Go to Command terminal
// 2. go to .ssh
// 3. give 'touch config'
// 4. get into dev node k80 with: ssh k80
// 5. get into your correct project 5 directory
// 6. do 'module load CUDA/10.1.105'
// 7. compile with: nvcc -arch=sm_37 -O3 diffusion_part_3.cu -o diffusion_part_3
// 8. run with: './diffusion_part_3

//////////////////// Next steps: /////////////////////////////////
// Specify 100 steps, then we'll plot the data over 100 steps for
// different block dimensions of 256, 512, and 1024 and for the 
// different CPU/GPU set ups -- so, you'll plot three plots for host (grid size = 256, 512, 1024)
// and three for CUDA set up and shared set up (grid size = 256, 512, 1024)
// explain results ... done