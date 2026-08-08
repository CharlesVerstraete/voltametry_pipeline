# -*- coding: utf-8 -*-
# -*- python 3.11.6 -*-

# Author : Charles Verstraete
# Date : 2026


"""
Import helper functions
"""

from preprocessing.utils.config import *

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

def transform_sweep_to_tc(signal, thr=1):
    """
    Transforms a sweep signal into a time course (tc) and detects events based on a threshold.
    Returns the time course and the indices of detected events.
    """
    tc = np.reshape(signal, (signal.size,))
    event_idx = np.where((tc[:-1] < thr) & (tc[1:] >= thr))[0]
    return tc, event_idx

