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

def find_event_indices(event_idx, thr=5000):
    df = pd.DataFrame({
    "sample": event_idx,
    })
    df["sweep_idx"] = df["sample"] // 10000
    df["event_interval"] = np.concatenate(([0], np.diff(event_idx)))
    df["new_event"] = df["event_interval"] > thr
    df["n_consecutive_events"] = 0
    df.loc[df["new_event"], "n_consecutive_events"] = np.diff(np.concatenate(([0],np.where(df["new_event"])[0])))
    df["event_id"] = ""
    df.loc[df["n_consecutive_events"] == 1, "event_id"] = "stim"
    df.loc[df["n_consecutive_events"] == 2, "event_id"] = "action"
    df.loc[df["n_consecutive_events"] == 3, "event_id"] = "fb"
    df.loc[df["n_consecutive_events"] == 5, "event_id"] = "other"
    return df

# %%

voltatc_events = find_event_indices(event_idx)



# %%

trigger_log = trigger_log.loc[:749]
fb_voltatc_events = voltatc_events.loc[voltatc_events["event_id"] == "fb"].copy().reset_index(drop=True)
fb_log_events = trigger_log.loc[np.isin(trigger_log["value"], fb_ids)].copy().reset_index(drop=True)
fb_eeg_log = event_log.loc[event_log["value"].isin(fb_ids)].copy().reset_index(drop=True)
# %%
fb_log_events["diff_time"] = np.concatenate(([0], np.diff(fb_log_events["time"])))
fb_voltatc_events["diff_time"] = np.concatenate(([0], np.diff(fb_voltatc_events["sample"])))/100000
fb_eeg_log["diff_time"] = np.concatenate(([0], np.diff(fb_eeg_log["sample"])))/2048

#%%
# plt.plot(fb_log_events["diff_time"], label="log")
# plt.plot(fb_voltatc_events["diff_time"], label="volta")
plt.plot((fb_log_events["diff_time"].values - fb_eeg_log["diff_time"].values)*1000, label="diff")
plt.legend()


#%%
#### label amine in matrix 0 = DA ; 1 = NE ; 2 = 5HT ; 3 = pH

pred_matrix = np.zeros((4, len(pred)))
for i, amine in enumerate(["DA", "NE", "5HT", "pH"]):
    pred_matrix[i, :] = pred[amine].values

n_out = len(tc)
interpolated_pred_matrix = np.zeros((4, n_out))
for i in range(4):
    pred_tc = pred_matrix[i, :]
    x_in = np.linspace(0, 1, len(pred_tc))
    x_out = np.linspace(0, 1, n_out)
    interpolated_pred_matrix[i, :] = np.interp(x_out, x_in, pred_tc)




# %%

colors = cmc.batlow(np.linspace(0, 1, interpolated_pred_matrix.shape[0]))
for i, c in enumerate(colors):
    plt.plot(interpolated_pred_matrix[i], lw=2, color=c)


# %%
from scipy.signal import butter, filtfilt

sample_rate = 10000
high_pass_cutoff = 0.5

filtered_pred_matrix = filtfilt(*butter(2, high_pass_cutoff/(sample_rate/2), btype='high'), interpolated_pred_matrix - np.mean(interpolated_pred_matrix, axis=1, keepdims=True))
# %%

# colors = cmc.batlow(np.linspace(0, 1, filtered_pred_matrix.shape[0]))
fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
for i, ax in enumerate(axs):
    ax.plot(filtered_pred_matrix[i])
    ax.set_ylabel(f"Amine {i+1}")
    ax.grid()
axs[-1].set_xlabel("Time (samples)")
plt.tight_layout()
plt.show()

# %%


epoch_length = 50000
fb_event_indices = voltatc_events.loc[voltatc_events["event_id"] == "fb", "sample"].values




# %%

epoch_matrix = np.zeros((len(fb_event_indices), 4, epoch_length))
for i, fb_idx in enumerate(fb_event_indices):
    start_idx = np.max([0, fb_idx - 30000])
    end_idx = np.min([len(filtered_pred_matrix[0]), fb_idx + 20000])
    epoch = filtered_pred_matrix[:, start_idx:end_idx]
    baseline = np.mean(epoch, axis=1, keepdims=True)
    std_epoch = np.std(epoch, axis=1, keepdims=True)
    epoch_matrix[i] = (epoch - baseline) / std_epoch
    


# %%

fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
ax = axs.flatten()
for i in range(4):
    mat = epoch_matrix[:, i, :]
    limit = np.max(np.abs(mat))*0.8
    im = ax[i].imshow(mat, aspect='auto', cmap=cmc.vik, vmin=-limit, vmax=limit)
    plt.colorbar(im, ax=ax[i])
    ax[i].axvline(x=30000, color='k', linestyle='--', lw=2)
    ax[i].set_title(f"Amine {i+1}")
    ax[i].set_xlabel("Time (samples)")
    ax[i].set_ylabel("Trials")

plt.tight_layout()
plt.show()


# %%
