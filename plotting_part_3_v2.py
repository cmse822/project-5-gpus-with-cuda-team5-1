import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Define the directory where your data files are stored
data_dir = "Project 5/data_blockdim_256"

# Define regular expressions to match the filenames and extract step numbers
regex = re.compile(r"(host_u|cuda_u|shared_u)(\d+).dat")

# Dictionary to store the filenames organized by type and step
data_files = {'host_u': {}, 'cuda_u': {}, 'shared_u': {}}

# Populate the data_files dictionary
for filename in os.listdir(data_dir):
    match = regex.match(filename)
    if match:
        data_type = match.group(1)
        step = int(match.group(2))
        data_files[data_type][step] = os.path.join(data_dir, filename)

# Get all unique step numbers across all types
all_steps = set()
for data_type in data_files:
    all_steps.update(data_files[data_type].keys())
all_steps = sorted(all_steps)

plt.figure(figsize=(10, 6))  # Adjust figure size as needed

# For each data type, plot data for each step if available
for data_type, files in data_files.items():
    for step in sorted(files.keys()):
        data = np.loadtxt(files[step])
        label = f'{data_type} Step {step}'
        if data_type == 'host_u':
            plt.plot(data, linestyle="--", color='red', label=label)
        elif data_type == 'cuda_u':
            plt.plot(data, linestyle="--", color='blue', label=label)
        elif data_type == 'shared_u':
            plt.plot(data, linestyle="--", color='green', label=label)

plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.title("Data Comparison Across Steps")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{data_dir}/all_comparisons.png")
plt.show()
