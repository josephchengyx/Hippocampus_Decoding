import numpy as np

def rebin_data(timepoints: np.array, timeseries: np.array, interval: float) -> np.array:
    # Re-bins timepoints and timeseries into evenly spaced time bins of size 'interval'
    # Interval should be specified in units of sec
    # Returns 1D arrays of re-binned timepoints and timeseries

    def time_series_to_intervals(timepoints: np.array, timeseries: np.array) -> np.array:
        # Compresses data points in timeseries into [start, end) time intervals per different data point
        # Returns a (k, 3) shaped array where k is the number of different data points
        # Return format is (start_time, end_time, data_point)
        start_time, curr_val = timepoints[0], timeseries[0]
        res = list()
        for idx in range(timepoints.shape[0]):
            if timeseries[idx] != curr_val:
                end_time = timepoints[idx]
                res.append([start_time, end_time, curr_val])
                start_time, curr_val = end_time, timeseries[idx]
        end_time = timepoints[-1]
        res.append([start_time, end_time, curr_val])
        return np.array(res)
    
    time_ranges = time_series_to_intervals(timepoints, timeseries)
    start_time, end_time = timepoints[0], timepoints[-1]
    new_timepoints = np.arange(start_time, end_time, interval)
    new_timeseries = np.zeros(new_timepoints.shape)
    ptr = 0
    window_start, window_end, data_pt = time_ranges[ptr,:]
    for idx in range(new_timepoints.shape[0]):
        timept = new_timepoints[idx]
        while not (window_start <= timept < window_end) and (ptr < time_ranges.shape[0] - 1):
            ptr += 1
            window_start, window_end, data_pt = time_ranges[ptr,:]
        new_timeseries[idx] = data_pt
    return new_timepoints, new_timeseries

def get_place_intervals(pv) -> np.array:
    # Obtains place intervals from pv object and arranges them in chronological order
    # Returns a (l, 3) shaped array where l is the number of timestamps,
    # Col 1 gives start time, col 2 gives end time, col 3 gives place bin
    place_intervals, place_intervals_count = np.array(pv.get('place_intervals')).T, np.array(pv.get('place_intervals_count')).flatten()
    arranged = np.empty((int(np.sum(place_intervals_count)), 3))
    cur = 0
    for bin, num_obs in enumerate(place_intervals_count):
        for obs in range(int(num_obs)):
            st_time, ed_time = place_intervals[bin, obs]
            arranged[cur,:] = [st_time, ed_time, bin+1]
            cur += 1
    arranged = arranged[np.argsort(arranged[:,1])]
    return arranged

def slot_in_spikes(time_intervals: np.array, cell: np.array) -> np.array:
    # Slots spikes from cell array containing spike timings into time intervals for one cell
    # Returns array of spike counts with length l (number of observations)
    spike_counts = np.zeros(time_intervals.shape[0])
    ptr = 0
    for obs in range(time_intervals.shape[0]):
        if ptr == cell.shape[0]:
            # No more spikes to slot
            break
        st, ed = time_intervals[obs, 0:2]
        while ptr < cell.shape[0] and cell[ptr] < st:
            # Skip spikes that fall into time gaps between observations
            ptr += 1
        while ptr < cell.shape[0] and st <= cell[ptr] < ed:
            # Slot spikes into current observation if it falls within current time interval
            spike_counts[obs] += 1
            ptr += 1
    return spike_counts

def slot_in_spike_times(time_intervals: np.array, cell: np.array) -> list:
    # Slots spikes from cell array containing spike timings into time intervals for one cell
    # Returns array of spike timings relative to start of each time bin, with length l (number of observations)
    spike_counts = [np.array([], dtype=np.float64) for bin in range(time_intervals.shape[0])]
    ptr = 0
    for obs in range(time_intervals.shape[0]):
        if ptr == cell.shape[0]:
            # No more spikes to slot
            break
        st, ed = time_intervals[obs, 0:2]
        while ptr < cell.shape[0] and cell[ptr] < st:
            # Skip spikes that fall into time gaps between observations
            ptr += 1
        while ptr < cell.shape[0] and st <= cell[ptr] < ed:
            # Slot spikes into current observation if it falls within current time interval
            spike_time = cell[ptr] - st
            spike_counts[obs] = np.append(spike_counts[obs], spike_time)
            ptr += 1
    return spike_counts

def spike_counts_per_observation(time_intervals: np.array, spike_times: list) -> np.array:
    # Returns (l, k) shaped array of spike counts (total k cells) for each observation (total l observations)
    spikecounts = np.empty((time_intervals.shape[0], len(spike_times)))
    for num, cell in enumerate(spike_times):
        spikecounts[:,num] = slot_in_spikes(time_intervals, cell)
    return spikecounts

def spike_rates_per_observation(time_intervals: np.array, spike_times: list) -> np.array:
    # Returns (l, k) shaped array of spike rates (total k cells) for each observation (total l observations)
    durations = time_intervals[:,1] - time_intervals[:,0]
    spikerates = np.empty((time_intervals.shape[0], len(spike_times)))
    for num, cell in enumerate(spike_times):
        spike_counts = slot_in_spikes(time_intervals, cell)
        spikerates[:,num] = np.divide(spike_counts, durations, out=np.zeros_like(spike_counts), where=durations!=0)
    return spikerates

def spike_times_per_observation(time_intervals: np.array, spike_times: list) -> list:
    # Returns (l, k) shaped list of spike times, relative to start of each time bin (total k cells)
    # for each observation (total l observations)
    spiketimes = list()
    for num, cell in enumerate(spike_times):
        spiketimes.append(slot_in_spike_times(time_intervals, cell))
    # Transpose list of spiketimes
    spiketimes = list(map(list, zip(*spiketimes)))
    return spiketimes

def get_trial_indices(session_data: np.array) -> np.array:
    # Obtains start (inclusive) and end (exclusive) trial indices for all trials in a session,
    # Returns a (k, 2) shaped array where k is the number of trials in the session
    is_trial = False
    indices = list()
    start_idx = 0
    # Session data is a 4 x n array, where n is the number of timepoints
    for idx in range(session_data.shape[1]):
        # Row 2 is for position bin data, set to 0 inbetween trials
        if not is_trial and session_data[2, idx] > 0:
            start_idx = idx
            is_trial = True
        elif is_trial and session_data[2, idx] == 0:
            indices.append([start_idx, idx])
            is_trial = False
    return np.array(indices)

def pos_bins_to_coords(pos_bins: np.array) -> np.array:
    # Converts bin numbers to positional (x, y) coordinates, returns a (n, 2) shaped array
    # Default maze dimensions
    num_bins = 40
    coord_min, size = -12.5, 25
    bin_width = size / num_bins
    # Need to offset bin number by -1 for zero-based indexing of bin horizontal/vertical index
    h, v = (pos_bins - 1) % num_bins, ((pos_bins - 1) // num_bins) + 1
    # Take center of bin as position coordinate
    x, y = coord_min + (h + 0.5) * bin_width, coord_min + (v + 0.5) * bin_width
    return np.vstack((x, y)).T

def pos_coords_to_bins(coords: np.array) -> int:
    # Converts positional (x, y) coordinates to bin numbers, returns an int
    # Default maze dimensions
    num_bins = 40
    coord_min, size = -12.5, 25
    bin_width = size / num_bins
    x, y = coords
    # Convert to row/column number for each axis
    h, v = int(np.floor((x - coord_min)/bin_width)), int(np.floor((y - coord_min)/bin_width))
    # Combine to get actual bin number
    return (v - 1) * num_bins + h

def spike_trains_from_times(spike_times: list, timepoints: np.array) -> np.array:
    # Converts spike times to spiketrain, returns a (n, l) shaped array
    # l is the number of recorded cells in the session
    # timepoints is a (n,) shaped array
    spiketrain = np.empty((timepoints.shape[0], len(spike_times)), dtype=np.int_)
    for cell, spike_time in enumerate(spike_times):
        # Spikes that fall between timepoints i and i+1 should be slotted into timepoint i,
        # i.e. when timepoint[i] <= spike_time < timepoint[i+1], spiketrain[i]++
        spiketrain[:-1,cell] = np.histogram(spike_time, timepoints)[0]
    # Fill last timepoint with 0 spikes across all cells
    spiketrain[-1,:] = 0
    return spiketrain

def spike_rates_from_trains(spiketrains: np.array, timepoints: np.array) -> np.array:
    # Divides all spike counts in spiketrains by the time interval for each respective bin,
    # returns (n, l) array of spike rates
    time_dur = np.empty_like(spiketrains, dtype=float)
    intervals = np.diff(timepoints)
    intervals[np.where(intervals==0)] = 1
    time_dur[:-1,:] = np.stack([intervals for _ in range(spiketrains.shape[1])], axis=1)
    time_dur[-1,:] = np.ones(spiketrains.shape[1])
    return np.divide(spiketrains, time_dur)

'''
def spike_trains_from_times_old(spike_times: list, timepoints: np.array) -> np.array:
    # Deprecated version which manually slots spikes in between timepoints, has a slow runtime
    # Converts spike times to spiketrain, returns a (n, l) shaped array
    # l is the number of recorded cells in the session
    # timepoints is a (n,) shaped array
    spiketrain = np.empty((timepoints.shape[0], len(spike_times)), dtype=np.int_)
    ptrs = np.zeros(len(spike_times), np.int_)
    for idx, timept in enumerate(timepoints):
        for cell, spike_time in enumerate(spike_times):
            if ptrs[cell] == spike_time.shape[0]:
                # No more spikes to process
                continue
            while spike_time[ptrs[cell]] >= timept:
                # Slot spike(s) into current index if it is between current and next timepoint
                spiketrain[idx, cell] += 1
                ptrs[cell] += 1
                if ptrs[cell] == spike_time.shape[0]:
                # No more spikes to process
                    break
    return spiketrain
'''

def split_by_trials(timeseries: np.array, indices: np.array) -> list:
    # Returns a length k list of length n arrays for each trial,
    # where k is the number of trials and n is the number of timepoints per trial
    # timeseries is n x something array, indices is k x 2 array
    trials = list()
    for trial in indices:
        # trial is a 1 x 2 array of start (inclusive) and end (exclusive) index for that trial
        st, ed = trial
        if len(timeseries.shape) > 1:
            trials.append(timeseries[st:ed,:])
        else:
            trials.append(timeseries[st:ed])
    return trials

def combine_trials(trials: list) -> np.array:
    # Concatenates trials back-to-back with each other,
    # returns a 1D array of length corresponding to the total number of time bins
    timeseries = trials[0]
    for trial in trials[1:]:
        timeseries = np.concatenate((timeseries, trial), axis=0)
    return timeseries

def normalize_dataset(data: np.array) -> np.array:
    # Takes in a 1-D array and scales the values to a maximum of the 95th percentile
    # Returns 1-D array of same length containing scaled values
    res = np.empty_like(data)
    lim = np.percentile(data, 95)
    if lim == 0:
        lim = 1
    for idx, val in enumerate(data):
        res[idx] = min(val / lim, 1)
    return res

def bin_firing_rates_3(cell: np.array, stats=None) -> np.array:
    # Takes in 1-D array of firing rates/counts of a single cell, and bins into
    # [0, 1, 2] states of firing rates (silent: bottom 20%, low firing: 20 - 60%, high firing: 60% - max)
    # Returns same length 1-D array of binned firing rates for cell
    res = np.empty(cell.shape, dtype=np.int8)
    if stats is not None:
        lo, hi = stats
    else:
        lo, hi = np.percentile(cell, 20), np.percentile(cell, 60)
    for idx, val in enumerate(cell):
        if val < lo:
            cat = 0
        elif val >= lo and val < hi:
            cat = 1
        else:
            cat = 2
        res[idx] = cat
    return res

def get_binning_stats_4(cell: np.array) -> tuple:
    lo, md, hi = np.percentile(cell, 25), np.median(cell), np.percentile(cell, 75)
    return (lo, md, hi)

def bin_firing_rates_4(cell: np.array, stats=None) -> np.array:
    # Takes in 1-D array of firing rates/counts of a single cell, and bins into
    # [0, 1, 2, 3] states of firing rates (low: bottom 25%, mid-low: 25% - 50%, mid-high: 50% - 75%, high: 75% - max)
    # Returns same length 1-D array of binned firing rates for cell
    res = np.empty(cell.shape, dtype=np.int8)
    if stats is not None:
        lo, md, hi = stats
    else:
        lo, md, hi = np.percentile(cell, 25), np.median(cell), np.percentile(cell, 75)
    for idx, val in enumerate(cell):
        if val < lo:
            cat = 0
        elif val >= lo and val < md:
            cat = 1
        elif val >= md and val < hi:
            cat = 2
        else:
            cat = 3
        res[idx] = cat
    return res

def hash_response(response: np.array) -> str:
        # Converts a 1-D population response (length l corresponding to number of cells)
        # into str format for hashing in a dictionary
        res = ''
        for cell in response:
            res += f'{cell:.0f}'
        return res

def map_response_distribution_cell(responses: np.array, dist=None) -> np.array:
    # Takes in (n x l) 2-D array of (l cells) population responses across (n number of) observations
    # Returns a 2-D array mapping of each cell's binned responses to its number of occurences for all cells
    # Format of mapping is dist[cell, cat] = occurence count, each row corresponds to the mapping of one cell
    if dist is None:
        dist = np.zeros((responses.shape[1], 4), dtype=np.int32)
    for (obs, cell), cat in np.ndenumerate(responses):
        dist[int(cell), int(cat)] += 1
    return dist

def map_response_distribution_popl(responses: np.array, dist=None) -> dict:
    # Takes in (n x l) 2-D array of (l cells) population responses across (n number of) observations
    # Returns a dictionary mapping of each unique population response to its number of occurences
    if dist is None:
        dist = dict()
    for obs in range(responses.shape[0]):
        response = hash_response(responses[obs,:])
        if response in dist:
            dist[response] += 1
        else:
            dist[response] = 1
    return dist 
