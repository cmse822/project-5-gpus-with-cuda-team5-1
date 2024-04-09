import numpy as np
import matplotlib.pyplot as plt


#Plot the difference between the first timesteps for the host kernel and cuda
#kernel, using n_steps=10

# data1 = np.loadtxt("data/host_u00000001.dat")
# data2 = np.loadtxt("data/cuda_u00000001.dat")
data1 = np.loadtxt("data/host_u00000100.dat")
data2 = np.loadtxt("data/cuda_u00000100.dat")
plt.plot(data1-data2)
plt.savefig("fig1.png")
plt.clf()


#Find the maximum difference between the host kernel and cuda, using n_steps=10
max_diff = 0
for i in np.arange(0,10):
    data1 = np.loadtxt("data/host_u00000{}00.dat".format(i))
    data2 = np.loadtxt("data/cuda_u00000{}00.dat".format(i))
    max_diff = np.max((np.max(np.abs(data1-data2)),max_diff))
print("Maximum cuda and host diff: ",max_diff)

#Find the maximum difference between the cuda kernel and cuda kernel using shared memory
#using n_steps=10

max_diff = 0
for i in np.arange(0,10):
    data1 = np.loadtxt("data/shared_u00000{}00.dat".format(i))
    data2 = np.loadtxt("data/cuda_u00000{}00.dat".format(i))
    max_diff = np.max((np.max(np.abs(data1-data2)),max_diff))
print("Maximum cuda and cuda shared diff: ",max_diff)


#Plot 

# for i in np.arange(0,9):
#     data = np.loadtxt("data/host_u00000{}00.dat".format(i))
#     p = plt.plot(data)
#     color = p[0].get_color()
#     data = np.loadtxt("data/cuda_u00000{}00.dat".format(i))
#     plt.plot(data,linestyle=":",color=color)
#     data = np.loadtxt("data/shared_u00000{}00.dat".format(i))
#     plt.plot(data,linestyle="--",color=color)
# plt.savefig("fig2.png")
# plt.clf()

for i in np.arange(0, 10):
    data_host = np.loadtxt("data/host_u00000{}00.dat".format(i))
    plt.plot(data_host, linestyle="--", color='red', label='Host' if i == 0 else "")
    data_cuda = np.loadtxt("data/cuda_u00000{}00.dat".format(i))
    plt.plot(data_cuda, linestyle="--", color='blue', label='CUDA' if i == 0 else "")
    data_shared = np.loadtxt("data/shared_u00000{}00.dat".format(i))
    plt.plot(data_shared, linestyle="--", color='green', label='Shared' if i == 0 else "")
plt.legend()
plt.savefig("fig2.png")
plt.clf()