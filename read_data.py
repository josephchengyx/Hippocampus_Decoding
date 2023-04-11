import os
import numpy as np
from scipy.io import loadmat
import h5py
from preproc import *

prefix = "/Volumes/Hippocampus/Data/picasso-misc/"
day_dir = "20181102"

# Get list of cells under the day directory
os.system(f"sh ~/Documents/neural_decoding/scripts/get_cells.sh {day_dir}")
cell_list = list()
with open("cell_list.txt", "r") as file:
    for line in file.readlines():
        cell_list.append(line.strip())
os.system("rm cell_list.txt")

# Load data from vmpv.mat object
pv = h5py.File(prefix + day_dir + "/session01/1vmpv.mat")
pv = pv.get('pv').get('data')

# Extract session time and position bins
session_data = np.array(pv.get('sessionTimeC'))
timepoints, pos_bins = session_data[0,:], session_data[1,:]

# Load data and extract spike times from all spiketrain.mat objects
spike_times = list()
for cell_dir in cell_list:
    spk = loadmat(prefix + day_dir + "/session01/" + cell_dir + "/spiketrain.mat")
    spk = spk.get('timestamps').flatten() # spike timestamps is loaded in as a column vector
    spike_times.append(spk)

# Get start/end indices of all trials in the session
trial_indices = get_trial_indices(session_data)

# Convert position bins to coordinates, spike timestamps to spiketrains
pos_coords = pos_bins_to_coords(pos_bins)
spiketrains = spike_trains_from_times(spike_times, timepoints)

# Split data into individual trials
timepoints = split_by_trials(timepoints, trial_indices)
pos_coords = split_by_trials(pos_coords, trial_indices)
spiketrains = split_by_trials(spiketrains, trial_indices)
