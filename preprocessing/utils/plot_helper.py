# -*- coding: utf-8 -*-
# -*- python 3.11.6 -*-

# Author : Charles Verstraete
# Date : 2026


"""
Plot helper functions
"""

from preprocessing.utils.config import *


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
