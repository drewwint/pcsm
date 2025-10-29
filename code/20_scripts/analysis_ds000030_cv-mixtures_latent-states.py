#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Cross-validation of latent states from ds000030 stop signal task
Dr. Drew E. Winters
Created on 9/16/2025
first activate environment:
source /home/dwinters/ascend-lab/publications/pcsm/code/env/venv_pcsm/bin/activate
"""

import os
import numpy as np
import pandas as pd

from pathlib import Path
from joblib import Memory

from projlib import cross_validation_verification as cvv

import pickle # for saving details dictionary as pickle
import json # for saving details as json

#---------------
#  Set up paths
#---------------

## cache for joblib
memory = Memory(location="/home/dwinters/ascend-lab/publications/pcsm/work/joblib_cache", verbose=0)
## output directory for fir z scored betas (already calculated)
out_dir = Path("/home/dwinters/ascend-lab/data/pcsm/ds000030/derivatives/fir_outputs")
out_dir.mkdir(exist_ok=True, parents=True)
## save director
save_dir = Path("/home/dwinters/ascend-lab/data/pcsm/ds000030/derivatives/cv_latent_states")
save_dir.mkdir(exist_ok=True, parents=True)


#--------------------------
# Loading / preparing data
#--------------------------

# loading the fir z scored data for modeling
all_fir_files = os.listdir(out_dir)
fir_zbeta_files = [f for f in all_fir_files if "fir_betas_z.npz" in f]

fir_zbeta_subs = []
for ii in fir_zbeta_files:
    data = np.load(os.path.join(out_dir, ii))['data']
    print(f"{ii}: {data.shape}")
    fir_zbeta_subs.append(data)

# preparing data for models
Y = np.concatenate(fir_zbeta_subs,axis=0).reshape(-1,200)
lengths = [int(Y.shape[0]/len(fir_zbeta_subs))]*len(fir_zbeta_subs)


#------------------
# Subject-Level CV
#------------------


# Running joint function ------------------------------------------------------------------------------
summary_df, details = cvv.select_states_and_mixtures_strict_gain(
    Y=Y,
    lengths=lengths,
    state_range=range(2, 6),
    mixture_range=(1, 2, 3),
    n_splits=5,
    n_starts_cv=1,
    n_iter_cv=60,
    tol_cv=1e-2,
    covariance_type="diag",
    random_state=5745,
    n_jobs=-1,
    pre_dispatch="all",
    batch_size=1,
    delta_thresh=2.0,
    wins_thresh=0.70,
    se_factor=2.0,
    rel_gain_states=0.10,
    gain_thresh_mixtures=5.0,
)

## results
summary_df
        #    mixtures  K_star     mean_LL     SE_LL     delta  selected
        # 0         1       3 -269.932795  0.331372       NaN     False
        # 1         2       3 -264.874885  1.001493  5.057911      True
        # 2         3       3 -261.609800  0.730968  3.265084     False

details[2]["cv_table"] # t mixtures latent state testing
        #    latent_states     mean_LL     SE_LL  delta_LL  perc_folds  selected_model
        # 0              2 -267.429966  0.751157       NaN         NaN           False
        # 1              3 -264.874885  1.001493  2.555081         1.0            True
        # 2              4 -262.622416  1.071757  2.252469         1.0           False
        # 3              5 -260.272286  1.323421  2.350130         1.0           False

# Save results ------------------------------------------------------------------------------------------
    # Results for mixtures dataframe as tsv
summary_df.to_csv(os.path.join(save_dir, "gmmhmm_cv_results_subj-mixtures.tsv"), sep="\t", index=False)

    # Results for latent states for 2 mixtures as tsv
details[2]["cv_table"].to_csv(os.path.join(save_dir, "gmmhmm_cv_results_subj-latent-states_mix-2.tsv"), sep="\t", index=False)

    # details dict as pickle and json
with open(os.path.join(save_dir, "gmmhmm_cv_details_subj.pkl"), "wb") as f:
    pickle.dump(details, f)

    # details dict as json
with open(os.path.join(save_dir, "gmmhmm_cv_details_subj.json"), "w") as f:
    json.dump(details, f, indent=4)



#-----------------------------------
# Sample-Level CV
#-----------------------------------

# running sample CV ---------------------------------------------------------------------------------
results_df_all, details_all = cvv.run_gmmhmm_cv(
    Y,
    state_range=range(2,6),    # k = 2..5
    n_mixtures=2,
    n_splits=5,
    n_starts=3,
    random_state=5745
)

## Results
results_df_all
    # >>> results_df
    #    latent_states     mean_LL     SE_LL  delta_LL  perc_folds  selected_model
    # 0              2 -266.952581  0.113347       NaN         NaN           False
    # 1              3 -263.776055  0.117484  3.176526         1.0            True
    # 2              4 -261.425234  0.087933  2.350821         1.0           False
    # 3              5 -259.454424  0.103845  1.970810         1.0           False

# Save results--------------------------------------------------------------------------------------
    # results dataframe as tsv
results_df_all.to_csv(os.path.join(save_dir, "gmmhmm_cv_results_all.tsv"), sep="\t", index=False)

    # details dict as pickle and json
with open(os.path.join(save_dir, "gmmhmm_cv_details.pkl"), "wb") as f:
    pickle.dump(details, f)

    # details dict as json
with open(os.path.join(save_dir, "gmmhmm_cv_details.json"), "w") as f:
    json.dump(details, f, indent=4)



