
import sys, os
PROJECT_ROOT = os.path.abspath("/home/cverstraete/nasShare/projects/cverstraete/voltametry_pipeline")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.utils.config import *
from preprocessing.utils.import_helper import *
from preprocessing.utils.align_helper import *
import pyabf


for subject in range(3, 6):
    fsub = f"sub-{subject:02d}"
    print(f"Processing {fsub} ")
    for run in range(1, 4):
        ### Load data
        print(f"Loading data for {fsub}, run {run}")
        # Sweep data from voltammetry acquisition system
        sweep_path = os.path.join(RAW_DIR, f"{fsub}", "volta", f"{fsub}_task-stratinfvolta_run-{run:02d}_MA1_MI{run}.abf")
        sweep = pyabf.ABF(sweep_path)

        print(f"Extracting signal for {fsub}, run {run}")

        volta_signal, volta_events = extract_signal_sweep(sweep, 4)
        save_extracted_signal(volta_signal, volta_events, subject_id=subject, run=run)

        print(f"Subject {fsub}, run {run} processed and saved successfully.")