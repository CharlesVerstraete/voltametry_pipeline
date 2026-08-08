# -*- coding: utf-8 -*-
# -*- python 3.11.6 -*-

# Author : Charles Verstraete
# Date : 2026

""" 
test script to check if the preprocessing pipeline works
"""

#%%
import sys, os
PROJECT_ROOT = os.path.abspath("/home/cverstraete/nasShare/projects/cverstraete/voltametry_pipeline")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.utils.config import *
from preprocessing.utils.import_helper import *
from preprocessing.utils.align_helper import *
import pyabf
from scipy.ndimage import gaussian_filter1d
# %%

#### Define subject and run
subject = 3
fsub = f"sub-{subject:02d}"
run = 1

### Load data
# Trigger log from machine
trigger_path = os.path.join(RAW_DIR, f"{fsub}", "beh", f"{fsub}_task-stratinfvolta_trigTable.csv")
trigger_log = pd.read_csv(trigger_path)

# Triggers from EEG acquisition system
event_path = os.path.join(RAW_DIR, f"{fsub}", "ieeg", f"{fsub}_task-stratinfvolta_run-{run:02d}_events.tsv")
event_log = pd.read_csv(event_path, sep="\t") 

# Sweep data from voltammetry acquisition system
sweep_path = os.path.join(RAW_DIR, f"{fsub}", "volta", f"{fsub}_task-stratinfvolta_run-{run:02d}_MA1_MI1.abf")
sweep = pyabf.ABF(sweep_path)

# Raw behavioral data
beh_path = os.path.join(DERIVATIVES_DIR, f"{fsub}", "beh", f"{fsub}_task-stratinfvolta_beh.csv")
beh = pd.read_csv(beh_path)

# Fitted simulation data
simu_path = os.path.join(DERIVATIVES_DIR, f"{fsub}", "beh", f"{fsub}_task-stratinfvolta_simu-forceranked.csv")
simu = pd.read_csv(simu_path)

# Amine concentration predictions
pred_path = os.path.join(DERIVATIVES_DIR, f"{fsub}", "volta", f"{fsub}_task-stratinfvolta_run-{run:02d}_pred.csv")
pred = pd.read_csv(pred_path)

#%%

# sweep.sweepC
# print(sweep.headerText)
plt.plot(sweep.sweepX, sweep.sweepY, lw=.5)


# %%

def extract_signal_sweep(sweep, ch, thr=1):
    """
    Extracts the signal from a sweep object and detects events based on a threshold.
    Returns the signal and a DataFrame with event information.
    """
    n_sweeps = sweep.sweepCount
    n_samples = sweep.sweepY.size
    signal = np.zeros((n_sweeps, n_samples))
    df = pd.DataFrame(index=np.arange(n_sweeps), columns=["n_events", "event_start", "event_end"])
    for sweep_idx in range(n_sweeps):
        sweep.setSweep(sweep_idx, channel=ch)
        y = sweep.sweepY
        signal[sweep_idx, :] = y
        idx_start = np.where((y[:-1] < thr) & (y[1:] >= thr))[0]
        idx_end = np.where((y[:-1] >= thr) & (y[1:] < thr))[0] 
        n_events = len(idx_start)
        df.loc[sweep_idx, "n_events"] = n_events
        df.loc[sweep_idx, "event_start"] = idx_start.tolist()
        df.loc[sweep_idx, "event_end"] = idx_end.tolist()
    return signal, df

#%%

def transform_sweep_to_tc(signal, thr=1):
    """
    Transforms a sweep signal into a time course (tc) and detects events based on a threshold.
    Returns the time course and the indices of detected events.
    """
    tc = np.reshape(signal, (signal.size,))
    event_idx = np.where((tc[:-1] < thr) & (tc[1:] >= thr))[0]
    return tc, event_idx


# %%

volta_signal, volta_events = extract_signal_sweep(sweep, 4)


# %%
tc, event_idx = transform_sweep_to_tc(volta_signal, thr=1)
# %%

def plot_sweep_with_events(signal, 
                           events, 
                           x_offset=300, 
                           y_offset=10,
                           figsize=(15, 5),
                           lw=3,
                           color=cmc.batlow):
    """
    Plots the sweep signal with detected events, applying offsets for better visualization.
    """
    cm = plt.get_cmap(color)
    colors = [cm(x/len(events)) for x in range(len(events))]
    plt.figure(figsize=figsize)
    for i, idx in enumerate(events):
        tc = signal[idx]
        x = np.arange(len(tc)) + i * x_offset
        y = tc + i * y_offset
        plt.plot(x, y, lw=lw, color=colors[i])
    plt.gca().axis('off') 
    plt.show()


# %%

def find_event_indices(event_idx, thr=1000):
    events_interval = np.concatenate(([event_idx[0]], np.diff(event_idx)))
    consecutive_events = np.where(events_interval > thr)[0]
    n_consecutive_events = np.diff(consecutive_events)
    other_idx = consecutive_events[np.where(n_consecutive_events == 5)[0]]
    stim_idx = consecutive_events[np.where(n_consecutive_events == 1)[0]]
    action_idx = consecutive_events[np.where(n_consecutive_events == 2)[0]]
    fb_idx = consecutive_events[np.where(n_consecutive_events == 3)[0]]
    return other_idx, stim_idx, action_idx, fb_idx

# %%


# %%
# diff_tc_idx


# %%

stim_idx
