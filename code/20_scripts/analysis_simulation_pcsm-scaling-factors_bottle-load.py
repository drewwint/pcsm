#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Ground-truth recovery for PCSM
Dr. Drew E. Winters
Created on 9/16/2025
"""

#------------------
# Setup/loading packages 
# -----------------

# packages -----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import scipy.stats

import projlib # local package for project specific functions
from projlib import metric_calculation as mc # loading the metric calculation functions specifically


#----------------------------
# Gausian Consistency Factors
#----------------------------

gaussian_cons = 1.4826
gaussian_iqr_cons = 1.349


#----------------------------------
# Setting up paths and loading data
#-----------------------------------

# Paths-------------------------------------------------------------------
load_dat = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/simulated_participants"
load_mod = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/derivatives/models"
base_out = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/derivatives/thresholds"


# Loading BOLD data-------------------------------------------------------
bold_v = []
events_l = []

for ii in range(15000):
  bb = np.load(os.path.join(load_dat,f"sim_subject-{ii+1}_BOLD_full.npy"))
  bold_v.append(bb)
  # ev
  ev = pd.read_csv(os.path.join(load_dat,f"sim_subject-{ii+1}_events.csv"))
  events_l.append(ev)


# Loading GMMHMM posterior values------------------------------------------
mean_mat_l = []
posterior_probs_l = []
# alpha_1_l = []
for ii in range(15000):
  # posterior_probs_l
  pp = np.load(os.path.join(load_mod, f"sim_subj_model-{ii+1}_posterior_probs.npy"))
  posterior_probs_l.append(pp)
  # mean_mat_l
  mml = np.load(os.path.join(load_mod, f"sim_subj_model-{ii+1}_state_mean_mat.npy"))
  mean_mat_l.append(mml)


#--------------------------------------
# Aquring Values Across All Simulations
#--------------------------------------

# Loop to extract values -------------------------------------------------------
bottleneck_idx = []
imp_cumulative = []
imp_duration = []
for ii in range(len(posterior_probs_l)):
    bnode = mc.compute_B_node(
       posterior_probs_l[ii],
       mean_mat_l[ii],
       bold_v[ii]
    )[0]
    dsp = mc.compute_Dsp(
       bnode
    )
    demand,resource_level, impacts,components = mc.compute_transition_load(
       posterior_probs_l[ii],
       dsp
    )
    cum = impacts["impact_cumulative"]
    dur = impacts["impact_duration"]
    bnidx = mc.compute_serial_bottleneck(
       components['labels']['mode'],
       demand,
       verbose=False,
    )
    bottleneck_idx.append(bnidx)
    imp_cumulative.append(cum)
    imp_duration.append(dur)


# Deriving Scaling Factors --------------------------------------------------------
# Load Impact cumulative effect scaling values_______________________
imp_cumulative_center = np.median(imp_cumulative)
imp_cumulative_gaussian_scale = gaussian_cons*scipy.stats.median_abs_deviation(imp_cumulative)
imp_cumulative_gaussian_iqr_scale = ((np.percentile(imp_cumulative,75) - np.percentile(imp_cumulative,25))/gaussian_iqr_cons)

# Load Duration decay effect scaling values__________________________
imp_duration_center = np.median(imp_duration)
imp_duration_gaussian_scale = gaussian_cons*scipy.stats.median_abs_deviation(imp_duration)
imp_duration_gaussian_iqr_scale = ((np.percentile(imp_duration,75) - np.percentile(imp_duration,25))/gaussian_iqr_cons)

# Bottleneck scaling values__________________________________________
bottleneck_center = np.median(bottleneck_idx)
bottleneck_gaussian_scale = gaussian_cons*scipy.stats.median_abs_deviation(bottleneck_idx)
bottleneck_gaussian_iqr_scale = ((np.percentile(bottleneck_idx,75) - np.percentile(bottleneck_idx,25))/gaussian_iqr_cons)


# Creating value dataframe ---------------------------------------------------------
scaling_factors_df = pd.DataFrame(
   {
      "center":np.array(
         (
            imp_cumulative_center, 
            imp_duration_center, 
            bottleneck_center
         )
      ),
      "scale":np.array(
         (
            imp_cumulative_gaussian_scale,
            imp_duration_gaussian_scale,
            bottleneck_gaussian_scale
         )
      ),
      "scale_iqr":np.array(
         (
            imp_cumulative_gaussian_iqr_scale,
            imp_duration_gaussian_iqr_scale,
            bottleneck_gaussian_iqr_scale
         )
      )
   },
   index=["impact_cumulative","impact_duration","bottleneck"]
)

# Saving Values -----------------------------------------------------------------------

scaling_factors_df.to_csv(os.path.join(base_out,"metric_scaling_factors_robust.tsv"),sep="\t")



