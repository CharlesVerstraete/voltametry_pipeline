# -*- coding: utf-8 -*-
# -*- python 3.11.6 -*-

# Author : Charles Verstraete
# Date : 2026

""" 
Configuration file for global variables and constants
"""

import os
from posixpath import expanduser
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import mne
import os
from copy import deepcopy
import gc
import json
import cmcrameri.cm as cmc



RAW_DIR = os.path.expanduser(os.path.join("~", "nasShare", "INM-GlobalShare", "Domenech_Lousada_2026_Cogamine", "data_STRATINF_VOLTA_bids"))

ROOT_DIR = os.path.expanduser(os.path.join("~", "nasShare", "projects", "cverstraete", "voltametry_pipeline"))
DERIVATIVES_DIR = os.path.join(ROOT_DIR, "data", "derivatives")

stim_ids = [10, 20, 30, 11, 21, 31, 12, 22, 32]
action_ids = [100, 101, 102, 110, 111, 112, 200, 201, 202, 210, 211, 212]
fb_ids = [151, 152, 153, 154]

