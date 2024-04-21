In this folder, we have the difference between the host kernel, cuda kernel and cuda shared memory kernel performances for fixed timesteps. 

The plotting has been done using the `plotting.py` file in the main directory. The plots starting with `fig1` prefix are the difference between cuda and host
kernels. 
And `fig2` prefixed files are the difference between all 3 kernels.

Also, the `max_diff.csv` file includes the maximum differences between corresponding kernels for specific block sizes and step sizes. Here we can see that the difference between cuda and host kernel becomes larger for bigger step and block sizes. 
