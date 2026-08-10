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
subject = 4
fsub = f"sub-{subject:02d}"
run = 1

### Load data
# Trigger log from machine
trigger_path = os.path.join(RAW_DIR, f"{fsub}", "beh", f"{fsub}_task-stratinfvolta_trigTable.csv")
trigger_log = pd.read_csv(trigger_path)

# Signal stim channel from voltammetry acquisition system
signal_path = os.path.join(DERIVATIVES_DIR,f"sub-{fsub}", "volta", f"sub-{fsub}_run-{run:02d}_signal-triggers.npy")
volta_signal = np.load(signal_path)

# Events from voltammetry acquisition system
events_path = os.path.join(DERIVATIVES_DIR, f"sub-{fsub}", "events", f"sub-{fsub}_run-{run:02d}_events-triggers.csv")
volta_events = pd.read_csv(events_path)

# Raw behavioral data
beh_path = os.path.join(DERIVATIVES_DIR, f"{fsub}", "beh", f"{fsub}_task-stratinfvolta_beh.csv")
beh = pd.read_csv(beh_path)

# Fitted simulation data
simu_path = os.path.join(DERIVATIVES_DIR, f"{fsub}", "beh", f"{fsub}_task-stratinfvolta_simu-forceranked.csv")
simu = pd.read_csv(simu_path)

# Amine concentration predictions
pred_path = os.path.join(DERIVATIVES_DIR, f"{fsub}", "pred", f"{fsub}_task-stratinfvolta_run-{run:02d}_pred.csv")
pred = pd.read_csv(pred_path)



# %%
tc, event_idx = transform_sweep_to_tc(volta_signal, thr=1)
# %%

voltatc_events = find_event_indices(event_idx, thr=3000)

#%%

# trigger_log[trigger_log["value"] == 240]
# trigger_log.loc[:827]

#%%

# trigger_log = trigger_log.loc[:827]
fb_voltatc_events = voltatc_events.loc[voltatc_events["event_id"] == "fb"].copy().reset_index(drop=True)
# fb_voltatc_events = fb_voltatc_events.drop(index=[0]).reset_index(drop=True)
fb_log_events = trigger_log.loc[np.isin(trigger_log["value"], fb_ids)].copy().reset_index(drop=True)
# fb_eeg_log = event_log.loc[event_log["value"].isin(fb_ids)].copy().reset_index(drop=True)
# %%
fb_log_events["diff_time"] = np.concatenate(([0], np.diff(fb_log_events["time"])))
fb_voltatc_events["diff_time"] = np.concatenate(([0], np.diff(fb_voltatc_events["sample"])))/VOLTA_SR
# fb_eeg_log["diff_time"] = np.concatenate(([0], np.diff(fb_eeg_log["sample"])))/IEEG_SR

#%%
#insert missingrows
for i in [69, 70, 71]:
    missing_row = pd.DataFrame({
        "onset": 0,
        "duration": 0,
        "trial_type": "",
        "value": 0,
        "sample": 0,
        "diff_time": np.nan
    }, index=[i])
    fb_eeg_log = pd.concat([fb_eeg_log.iloc[:i], missing_row, fb_eeg_log.iloc[i:]]).reset_index(drop=True)
fb_eeg_log.loc[72, "diff_time"] = fb_log_events.loc[72, "diff_time"]
# print(fb_eeg_log.loc[65:75])
# print(fb_voltatc_events.loc[65:75])
#%%

# for i, idx in enumerate(fb_voltatc_events.index):
#     print(f"FB Volta {i}: {fb_voltatc_events.loc[idx, 'diff_time']}, FB Event {i}: {fb_log_events.loc[i, 'diff_time']} diff : {fb_voltatc_events.loc[idx, 'diff_time'] - fb_log_events.loc[i, 'diff_time']}")

# plt.plot(fb_log_events["diff_time"], label="log")
# plt.plot(fb_voltatc_events["diff_time"], label="volta")
plt.hist((fb_log_events["diff_time"].values - fb_voltatc_events["diff_time"].values)*1000, bins=30, density=True, alpha=0.5, label="log-volta")
plt.hist((fb_log_events["diff_time"].values - fb_eeg_log["diff_time"].values)*1000, bins=30, density=True, alpha=0.5, label="log-eeg")
# plt.hist((fb_eeg_log["diff_time"].values - fb_voltatc_events["diff_time"].values)*1000, bins=30, density=True, alpha=0.5, label="eeg-volta")

# plt.hist((fb_log_events["diff_time"].values[:200] - fb_voltatc_events["diff_time"].values[:200])*1000, bins=30, density=True)
plt.legend()
plt.show()

plt.plot((fb_log_events["diff_time"].values - fb_voltatc_events["diff_time"].values)*1000, label="log-volta", alpha=0.5)
plt.plot((fb_log_events["diff_time"].values - fb_eeg_log["diff_time"].values)*1000, label="log-eeg", alpha=0.5)
# plt.plot((fb_voltatc_events["diff_time"].values - fb_eeg_log["diff_time"].values)*1000, label="volta-eeg", alpha=0.5)
plt.legend()

plt.show()

#%%

fb_voltatc_events["resampled"] = np.round(fb_voltatc_events["sample"].values / 100000 * 500).astype(int)


#%%
#### label amine in matrix 0 = DA ; 1 = NE ; 2 = 5HT ; 3 = pH

pred_matrix = np.zeros((4, len(pred)))
for i, amine in enumerate(["DA", "NE", "5HT", "pH"]):
    pred_matrix[i, :] = pred[amine].values

n_out = int(len(tc)/100000*500)
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

sample_rate = 500
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


epoch_length = 2000
fb_event_indices = fb_voltatc_events["resampled"].values




# %%

epoch_matrix = np.zeros((len(fb_event_indices), 4, epoch_length))
for i, fb_idx in enumerate(fb_event_indices):
    start_idx = np.max([0, fb_idx - 1500])
    end_idx = np.min([len(filtered_pred_matrix[0]), fb_idx + 500])
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
    ax[i].axvline(x=1500, color='k', linestyle='--', lw=2)
    ax[i].set_title(f"Amine {i+1}")
    ax[i].set_xlabel("Time (samples)")
    ax[i].set_ylabel("Trials")

plt.tight_layout()
plt.show()


# %%
fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
ax = axs.flatten()

for i, color in zip([0, 1], ["firebrick", "forestgreen"]):
    for k, style in zip([0, 1], ["-", "--"]):
        for j in range(4):
            idx = test_beh.loc[(test_beh["fb"] == i) & (test_beh["trap"] == k)].index
            mat = epoch_matrix[idx, j, :]
            mean_mat = gaussian_filter1d(np.mean(mat, axis=0), sigma=200)
            sem_mat = gaussian_filter1d(np.std(mat, axis=0)/np.sqrt(len(idx)), sigma=200)
            ax[j].plot(mean_mat, color=color, lw=0.5, linestyle=style, label=f"FB {i} - Trap {k}")
            ax[j].fill_between(np.arange(len(mean_mat)), mean_mat - sem_mat, mean_mat + sem_mat, color=color, alpha=0.1)
            ax[j].legend()
            ax[j].axvline(x=1500, color='k', linestyle='--', lw=2)
            ax[j].set_title(f"Amine {j+1}")
            ax[j].set_xlabel("Time (samples)")
            ax[j].set_ylabel("Trials")

# %%


from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from scipy.stats import mode, pearsonr, linregress
from joblib import Parallel, delayed

from sklearn.svm import LinearSVC
from sklearn.frozen import FrozenEstimator
# %%


pipeline = make_pipeline(
    StandardScaler(),
    # RidgeCV(alphas=np.logspace(-3, 6, 10), cv=5)
    Ridge(alpha=1.0e6, random_state=42)
)
cv = KFold(n_splits=3, shuffle=True, random_state=42)



# %%

score = np.zeros(2000)
# score2 = np.zeros(2000)
metrics = np.zeros(2000)
betas = np.zeros((2000, 4))

y = test_beh["update_best_qval"].values

for t in range(2000):
    X = epoch_matrix[:, :, t].T
    X = X.reshape(len(y), -1)

    fold_metrics = []
    fold_betas = []
    fold_score = []
    tmp_betas = []
    all_y_true = np.zeros((len(y),))
    all_y_pred = np.zeros((len(y),))
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r, _ = pearsonr(y_test, y_pred)
        all_y_true[test_idx] = y_test
        all_y_pred[test_idx] = y_pred
        fold_score.append(np.arctanh(r))
        fold_metrics.append(mse)
        tmp_betas.append(pipeline.named_steps["ridge"].coef_)
    tmp_betas = np.array(tmp_betas)

    score[t] = np.mean(fold_score)
    metrics[t] = np.mean(fold_metrics)
    betas[t] = np.mean(tmp_betas, axis=0)

# %%

fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
axs[0].plot(score, label='R Score')
axs[0].axvline(x=1500, color='k', linestyle='--', label='Feedback Event')
axs[0].set_title('Decoding Performance Over Time')
axs[0].set_ylabel('Z-Score')


lim = np.max(np.abs(betas))
im = axs[1].imshow(betas.T, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='none')
plt.colorbar(im, ax=axs[1], label='Beta Coefficients', orientation='horizontal')

axs[1].set_title('Beta Coefficients for Ridge Regression')
axs[1].set_xlabel('Time (samples)')
axs[1].set_ylabel('Amine Type')
axs[1].axvline(x=1500, color='k', linestyle='--', label='Feedback Event')
axs[1].set_yticks(ticks=np.arange(4), labels=["DA", "NE", "5HT", "pH"])

# %%
