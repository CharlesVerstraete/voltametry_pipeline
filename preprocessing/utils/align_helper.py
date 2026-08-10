# -*- coding: utf-8 -*-
# -*- python 3.11.6 -*-

# Author : Charles Verstraete
# Date : 2026


"""
Import helper functions
"""

from preprocessing.utils.config import *


def find_event_indices(event_idx, thr=5000):
    df = pd.DataFrame({
    "sample": event_idx,
    })
    df["sweep_idx"] = df["sample"] // 10000
    df["event_interval"] = np.concatenate(([0], np.diff(event_idx)))
    df["new_event"] = df["event_interval"] > thr
    df["n_consecutive_events"] = 0
    df.loc[df["new_event"], "n_consecutive_events"] = np.concatenate((np.diff(np.where(df["new_event"])[0]), [0]))
    df["event_id"] = ""
    df.loc[df["n_consecutive_events"] == 1, "event_id"] = "stim"
    df.loc[df["n_consecutive_events"] == 2, "event_id"] = "action"
    df.loc[df["n_consecutive_events"] == 3, "event_id"] = "fb"
    df.loc[df["n_consecutive_events"] == 5, "event_id"] = "other"
    return df

