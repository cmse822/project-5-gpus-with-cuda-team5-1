import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

#Plot the difference between the first timesteps for the host kernel and cuda
#kernel, using n_steps=10
blocksize = sys.argv[1]  # First command line argument after the script name
n_steps = sys.argv[2]

# data1 = np.loadtxt("data/host_u00000001.dat")
# data2 = np.loadtxt("data/cuda_u00000001.dat")
data1 = np.loadtxt("data/host_u00000000.dat")
data2 = np.loadtxt("data/cuda_u00000000.dat")
plt.plot(data1-data2)
plt.title(f"blocksize {blocksize}, n_steps {n_steps}")
filename = f"fig1_bs{blocksize}_ns{n_steps}.png"
plt.savefig(filename)
plt.clf()


#Find the maximum difference between the host kernel and cuda, using n_steps=10
max_diff1 = 0
for i in np.arange(0,10):
    data1 = np.loadtxt(f"data/host_u0000{i:02}00.dat")
    data2 = np.loadtxt(f"data/cuda_u0000{i:02}00.dat")
    max_diff1 = np.max((np.max(np.abs(data1-data2)),max_diff1))
print("Maximum cuda and host diff: ",max_diff1)

#Find the maximum difference between the cuda kernel and cuda kernel using shared memory
#using n_steps=10

max_diff = 0
for i in np.arange(0,10):
    data1 = np.loadtxt(f"data/shared_u0000{i:02}00.dat")
    data2 = np.loadtxt(f"data/cuda_u0000{i:02}00.dat")
    max_diff = np.max((np.max(np.abs(data1-data2)),max_diff))
print("Maximum cuda and cuda shared diff: ",max_diff)

csv_file_path = 'results/max_diff.csv'
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([blocksize, n_steps, max_diff1, max_diff])

for i in np.arange(0, 10):
    plt.title(f"blocksize {blocksize}, n_steps {n_steps}")
    data_host = np.loadtxt(f"data/host_u0000{i:02}00.dat")
    plt.plot(data_host, linestyle="--", color='red', label='Host' if i == 0 else "")
    data_cuda = np.loadtxt(f"data/cuda_u0000{i:02}00.dat")
    plt.plot(data_cuda, linestyle="--", color='blue', label='CUDA' if i == 0 else "")
    data_shared = np.loadtxt(f"data/shared_u0000{i:02}00.dat")
    plt.plot(data_shared, linestyle="--", color='green', label='Shared' if i == 0 else "")
plt.legend()
filename = f"fig2_bs{blocksize}_ns{n_steps}.png"
plt.savefig(filename)
plt.clf()