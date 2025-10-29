#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
FIR modeling of BOLD data from ds000030 stop signal task
Dr. Drew E. Winters
Created on 9/16/2025
"""

#---------
# Setup
#---------

# Package Imports
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn
import.nilearn.plotting
import.nilearn.image
from nilearn.interfaces.bids import get_bids_files
from nilearn.maskers import NiftiLabelsMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.datasets import fetch_atlas_schaefer_2018
import warnings
from scipy import stats as spstats
import gc



#  Set up paths
base_bid = "/home/dwinters/ascend-lab/data/pcsm/ds000030"
der_bid = os.path.join(base_bid, "derivatives", "fmriprep")
out_dir = Path("/home/dwinters/ascend-lab/data/pcsm/ds000030/derivatives/fir_outputs")
out_dir.mkdir(exist_ok=True, parents=True)


# Load BIDS files
    # evants
event_files=get_bids_files(
    base_bid,
    file_tag="task-stopsignal_events",
    file_type="tsv",
    modality_folder="func",
    # filters=('task','stopsignal')
    )

    # confounds
confound_files=get_bids_files(
    der_bid,
    file_tag="task-stopsignal_bold_confounds",
    file_type="tsv",
    modality_folder="func",
    # filters=('task','stopsignal')
    )

    # minimally preprocessed bold
bold_files=get_bids_files(
    der_bid,
    file_tag="task-stopsignal_bold_space-MNI152NLin2009cAsym_preproc",
    file_type="nii.gz",
    modality_folder="func"
    )

# getting subject ID from file path
sub_ids=[]
for i in range(len(bold_files)):
    sub_ids.append(bold_files[i].split("/")[9]) 


# ensure event and confound files match subjects with bold files
event_files_sub=[]
event_ids=[]
for i in range(len(event_files)):
    id = event_files[i].split("/")[9].split("_")[0]
    if pd.Series(id).isin(sub_ids).any():
        event_ids.append(id)
        event_files_sub.append(event_files[i])


confound_files_sub=[]
confound_ids=[]
for i in range(len(confound_files)):
    id = confound_files[i].split("/")[9]
    if pd.Series(id).isin(sub_ids).any():
        confound_ids.append(id)
        confound_files_sub.append(confound_files[i])

    # test to ensure lengths match
len(confound_files_sub) == len(bold_files) # true 259
len(bold_files) == len(event_files_sub) # True 259
len(event_files_sub) == len(confound_files_sub) # True 259

    # test to ensure IDs match and are in same order
sub_ids == event_ids # True
sub_ids == confound_ids # True
event_ids == confound_ids # True


    # tripple check a few files to ensure they match
# bold_files[10]
# event_files_sub[10]
# confound_files_sub[10]
    ## Yes files are in the same order and match - we are good to go! No need to sort


# Load atlas for parcellation
atlas_files = fetch_atlas_schaefer_2018(
    n_rois = 200, 
    yeo_networks = 17, 
    data_dir = os.path.join(base_bid[0:36], "atlases"))
# loading atlas
atlas = nilearn.image.load_img(atlas_files["maps"])
# getting region names
labels = atlas_files["labels"]
# formatting into a form we can use
node_names = []
for label_byte in labels[1:]:
  # Decode the bytes object to a string and then slice it
  node_names.append(label_byte.split("s_")[1])

# plotting atlas
# nilearn.plotting.plot_roi(atlas, title="Schaefer atlas")
# plt.show(), plt.close(), plt.clf(), plt.cla()




# ---------------
# FIR modeling
# ---------------

# suppress warnings that are not relevant to our analysis

warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
  # We suppress this becasue we are not working with z_scores from the GLM - the GLM is having issues with the background zscore calculation but we are not reguesting them - we are requesting the betas
  # These betas are not the cause of the sqrt error - that is from zscoring in the GLM. We instead zscore outside the model using a model to that does cause this issue.
  # We also just checked to make sure there were no 0 values - and there were none - indicating that we do not have to be concerned about this in our model - so we just suppress it.

warnings.filterwarnings("ignore", message="memory_level is currently set to 0")
    # suppress warning about memory level - we are ok with this

warnings.filterwarnings("ignore", message="Resampling labels at transform time")
    # suppress warning about resampling labels at transform time - we are ok with this

# FIR parameters
fir_delays = [0, 1, 2, 3, 4]  # 4 FIR bins
n_bins = len(fir_delays)

# Use one masker fitted ONCE (e.g., on first BOLD) and then .transform thereafter
    #  IMPORTANT: do not re-fit per contrast/step
label_masker = NiftiLabelsMasker(
    labels_img=atlas_img,
    resampling_target="labels",   # always resample BOLD to atlas
    standardize=False,
    detrend=False,
    verbose=0
)
label_masker.fit()
# number of parcels
n_parcels = getattr(label_masker, "n_elements_", None)
n_parcels = len(node_names) # True 200 parcels

# lists for FIR modeling
clean_img_fir = []
clean_img_fir_z = []
clean_img_fir_predicted = []
clean_img_fir_predicted_z = []
clean_img_fir_resid = []
clean_img_fir_resid_z = []

# FIR modeling loop
for ii in sub_ids:
    # indices
    cf_ind = int(np.where(np.array(confound_ids) == ii)[0][0])
    ev_ind = int(np.where(np.array(event_ids) == ii)[0][0])
    bd_ind = int(np.where(np.array(sub_ids) == ii)[0][0])
    # confounds
    confounds = pd.read_csv(
        confound_files_sub[cf_ind],
        sep="\t",
        usecols=["FramewiseDisplacement","X","Y","Z","RotX","RotY","RotZ","WhiteMatter"],
        dtype=np.float32,
    )
    censor = (pd.read_csv(confound_files_sub[cf_ind], sep="\t", usecols=["FramewiseDisplacement"])
                .squeeze("columns").astype(np.float32))
    confounds["censor"] = (censor > 0.5).astype(np.float32)
    confounds.fillna(0, inplace=True)
    # events
    events = pd.read_csv(event_files_sub[ev_ind], sep="\t")
    events.loc[np.where(events.TrialOutcome == "JUNK")[0], "TrialOutcome"] = "UnsuccessfulGo"
    # bold (mmap)
    bold = nib.load(bold_files[bd_ind], mmap=True)
    print(f"Subject {ii}: confound {cf_ind}, event {ev_ind}, bold {bd_ind}, "
          f"bold shape {bold.shape}, n events {len(events)}, n confounds {len(confounds)}")
    # FIR Modeling
    ev = events[["onset","duration","TrialOutcome"]].copy()
    ev.columns = ["onset","duration","trial_type"]
    ev = ev.loc[ev["trial_type"].notna()].copy()
    trial_ids, trial_counter = [], {}
    for t in ev["trial_type"]:
        trial_counter[t] = trial_counter.get(t, 0) + 1
        trial_ids.append(f"{t}__{trial_counter[t]:03d}")
    ev["trial_type"] = trial_ids
    model = nilearn.glm.first_level.FirstLevelModel(
        t_r=2,
        hrf_model='fir',
        fir_delays=fir_delays,
        drift_model=None,
        high_pass=None,
        standardize=True,
        signal_scaling=False,
        minimize_memory=False,   # keep to get predicted
        n_jobs=1,
        memory='nilearn_cache',
        memory_level=1
    )
    model_fit = model.fit(bold, events=ev, confounds=confounds)
    # design / FIR columns
    dm = model_fit.design_matrices_[0]
    fir_cols = [c for c in dm.columns if "__" in c and "delay_" in c]
    col2idx = {c: i for i, c in enumerate(dm.columns)}
    print("   Design matrix shape:", dm.shape)
    print(f"   Number of FIR regressors: {len(fir_cols)}")
    # contrasts -> parcel values
    region_fir_vals = np.empty((len(fir_cols), 1, n_parcels), dtype=np.float32)
    for j, cname in enumerate(fir_cols):
        contrast = np.zeros(dm.shape[1], dtype=np.float32)
        contrast[col2idx[cname]] = 1.0
        try:
            map_img = model_fit.compute_contrast(contrast, output_type="effect_size")
            region_vals = label_masker.transform(map_img)
            region_fir_vals[j, 0, :] = region_vals.astype(np.float32, copy=False)
        except np.linalg.LinAlgError:
            print(f"Singular matrix for {cname} — skipping.")
            region_fir_vals[j, 0, :] = np.nan
        del map_img
    # (n_trials * n_bins, n_nodes) -> (n_trials, n_bins, n_nodes)
    region_fir_array = region_fir_vals.reshape(len(fir_cols), n_parcels)
    print("   NaN parcels: ", np.isnan(region_fir_array).sum())
    n_trials = len(ev)
    region_fir_array = region_fir_array.reshape(n_trials, n_bins, -1)
    # zscore betas
    z_node = spstats.zscore(region_fir_array, axis=2, ddof=0)
    z_time = spstats.zscore(z_node.reshape(-1, z_node.shape[2]), axis=0, ddof=0)
    z_rs = z_time.reshape(region_fir_array.shape).astype(np.float32)
    print("   zscore NaN parcels: ", np.isnan(z_rs).sum())
    # predicted (parcel space)
    pred_img = model_fit.predicted[0]
    pred_ts = label_masker.transform(pred_img).astype(np.float32)
    pred_z_node = spstats.zscore(pred_ts, axis=1, ddof=0)
    pred_z_time = spstats.zscore(pred_z_node, axis=0, ddof=0)
    # residuals (parcel space)
    bold_ts = label_masker.transform(bold).astype(np.float32)
    res_vals = bold_ts[:pred_ts.shape[0], :] - pred_ts
    res_z_node = spstats.zscore(res_vals, axis=1, ddof=0)
    res_z_time = spstats.zscore(res_z_node, axis=0, ddof=0)
    # -------- save to disk; append *paths* --------
    sid = str(ii)
    f_betas   = out_dir / f"{sid}_fir_betas.npz"
    f_betas_z = out_dir / f"{sid}_fir_betas_z.npz"
    f_pred    = out_dir / f"{sid}_pred_ts.npz"
    f_pred_z  = out_dir / f"{sid}_pred_ts_z.npz"
    f_res     = out_dir / f"{sid}_res_ts.npz"
    f_res_z   = out_dir / f"{sid}_res_ts_z.npz"
    np.savez_compressed(f_betas,   data=region_fir_array)
    np.savez_compressed(f_betas_z, data=z_rs)
    np.savez_compressed(f_pred,    data=pred_ts)
    np.savez_compressed(f_pred_z,  data=pred_z_time.astype(np.float32))
    np.savez_compressed(f_res,     data=res_vals)
    np.savez_compressed(f_res_z,   data=res_z_time.astype(np.float32))
    clean_img_fir.append(str(f_betas))
    clean_img_fir_z.append(str(f_betas_z))
    clean_img_fir_predicted.append(str(f_pred))
    clean_img_fir_predicted_z.append(str(f_pred_z))
    clean_img_fir_resid.append(str(f_res))
    clean_img_fir_resid_z.append(str(f_res_z))
    # cleanup AFTER saving - this minimizes RAM use
    del region_fir_array, z_rs, pred_ts, pred_z_time, res_vals, res_z_time
    del region_fir_vals, dm, ev, events, confounds, bold_ts
    gc.collect()


# All outputs suggested no errors in FIR modeling and no NAN values in betas or zscored betas
# Files written to disk for later use


# loading the fir z scored data for modeling
out_dir

all_fir_files = os.listdir(out_dir)
fir_zbeta_files = [f for f in all_fir_files if "fir_betas_z.npz" in f]

fir_zbeta_subs = []
for ii in fir_zbeta_files:
    data = np.load(os.path.join(out_dir, ii))['data']
    print(f"{ii}: {data.shape}")

k = np.load(os.path.join(out_dir,fir_zbeta_files[2]))["data"]












