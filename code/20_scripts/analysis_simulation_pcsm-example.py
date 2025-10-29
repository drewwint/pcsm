#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
PCSM example with simulated data
Dr. Drew E. Winters
Created on 9/27/2025

first activate environment:
source /home/dwinters/ascend-lab/publications/pcsm/code/env/venv_pcsm/bin/activate
"""

#------------------
# Setup/loading packages 
#----------

# packages 
## file handling
import os

## data handling
import numpy as np
import pandas as pd
import nilearn.datasets

## plotting
import matplotlib.pyplot as plt
import seaborn as sns


## PCSM specific calculations
import projlib # local package for project specific functions
from projlib import metric_calculation as mc # loading the metric calculation functions specifically
from projlib import characterization_help as ch
from projlib import plotting as pj_plt



#-------------------------------------
# Setting up paths and loading data
#-------------------------------------

# Paths-------------------------------------------------------------------
atlas_loc = r"/home/dwinters/ascend-lab/resources/atlases"
load_dat = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/simulated_participants"
load_mod = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/derivatives/models"
load_th = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/derivatives/thresholds"
base_out = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/derivatives/example"

# Loading BOLD data-------------------------------------------------------
bold_v = []
events_l = []
subs= [2055, 7178, 12050]

for ii in subs: # loading 3 subjects for example, one from each transition group at the high noise level. 
  bb = np.load(os.path.join(load_dat,f"sim_subject-{ii}_BOLD_full.npy"))
  bold_v.append(bb)
  # ev
  ev = pd.read_csv(os.path.join(load_dat,f"sim_subject-{ii}_events.csv"))
  events_l.append(ev)


# Loading GMMHMM posterior values------------------------------------------
mean_mat_l = []
posterior_probs_l = []
alpha_1_l = []
for ii in subs: # loading 3 subjects for example, one from each transition group at the high noise level. 
  # posterior_probs_l
  pp = np.load(os.path.join(load_mod, f"sim_subj_model-{ii}_posterior_probs.npy"))
  posterior_probs_l.append(pp)
  # mean_mat_l
  mml = np.load(os.path.join(load_mod, f"sim_subj_model-{ii}_state_mean_mat.npy"))
  mean_mat_l.append(mml)
  # alpha_1_l
  a1 = np.load(os.path.join(load_mod, f"sim_subj_model-{ii}_alpha_1.npy"))
  alpha_1_l.append(a1)


# Loading thresholds
dsp_tau= pd.read_csv(os.path.join(load_th, "dsp_threshold.tsv"), sep="\t").loc[:,["tau_lo","tau_hi"]].values



#---------------
# diagnostics
#---------------

# calculating flagged trials----------------------------------------
flag_time = []
flag_trial = []
flag_info = []
for ii in range(len(bold_v)):
    time, trial, info = mc.flag_trials(
       Dmaha_val = mc.compute_D_maha_fir_mahalanobis(
          posterior_probs_l[ii], 
            mean_mat_l[ii], 
            bold_v[ii]
        ), 
        rho_val = mc.rho_ts(
            posterior_probs_l[ii]
        ), 
        pt_val = mc.compute_pt(
            posterior_probs_l[ii],
            alpha_1_l[ii]
        )    
    )
    flag_time.append(time)
    flag_trial.append(trial)
    flag_info.append(info)


# Plotting timeseries with markers for flagged trials--------
trans_l = ["High Transitions", "Medium Transitions", "Low Transitions"]
trans_ind = [4,3,3]
llg = [True, False, False]
for ii in range(len(bold_v)):
    pj_plt.plot_subject_fir(
    ii,
    bold_v,
    events_l,
    legend_anchor = (-0.025,0.15),
    flagged_idx = flag_time[ii],
    legend_show = llg[ii],
    title = "Timeseries with Flagged Timepoints: " + trans_l[ii],
    show = True,
    save_loc = os.path.join(base_out, f"figure_flagged_{trans_l[ii][0:trans_ind[ii]]}_{trans_l[ii][-11:]}.tiff")
    )


#--------------
# Decoding
#--------------

# Node responding--------------------------------------------------------
bnode_l = []
for ii in range(len(bold_v)):
    bn = mc.compute_B_node(posterior_probs_l[ii], mean_mat_l[ii], bold_v[ii])[0]
    bnode_l.append(bn)


# Plotting timeseries heatmap of Bnode across time_______________
## All nodes
for ii in range(len(bold_v)):
    pj_plt.plot_bnode_heatmap(
        bnode_l[ii], 
        n_nodes=200, 
        n_bins=5,
        save_loc = os.path.join(base_out, f"figure_bnode-heatmap_{trans_l[ii][0:trans_ind[ii]]}_{trans_l[ii][-11:]}.tiff")
        )


# Serial/Parallel-------------------------------------------------------
dsp_l = []
for ii in range(len(bold_v)):
    dsp = mc.compute_Dsp(bnode_l[ii])
    dsp_l.append(dsp)


# Plotting timeseries and heatmap of Dsp across time______________
for ii in range(len(bold_v)):
    plt.plot(dsp_l[ii])
    plt.axhline(dsp_tau[0][0], color = "red")
    plt.axhline(dsp_tau[0][1], color = "red", linestyle = "--")
    plt.savefig(os.path.join(base_out, f"figure_dsp-timeseries_{trans_l[ii][0:trans_ind[ii]]}_{trans_l[ii][-11:]}.tiff"))
    pj_plt.plot_dsp_timeseries_heatmaps(
       dsp_l[ii], 
       dsp_tau[0][0], 
       dsp_tau[0][1],
       save_loc = os.path.join(base_out, f"figure_dsp-heatmap_{trans_l[ii][0:trans_ind[ii]]}_{trans_l[ii][-11:]}.tiff")
       )


# Transition Impact/Load -------------------------------------------------------
demand_l = []
resource_l = []
impact_l = []
info_l = []
corr_dsp_demand_l = []
corr_dsp_resource_l = []
corr_demand_resource_l = []
for ii in range(len(posterior_probs_l)):
    demand, resource_level, impact, impact_severity, demand_info, resource_info, components = mc.compute_transition_load(
        posterior_probs_l[ii],
        dsp_l[ii]
    )
    demand_l.append(demand)
    resource_l.append(resource_level)
    impact_l.append(impact)
    info_l.append(components)
    corr_dsp_demand_l.append(np.corrcoef(dsp_l[ii],demand)[0,1])
    corr_dsp_resource_l.append(np.corrcoef(dsp_l[ii],resource_level)[0,1])
    corr_demand_resource_l.append(np.corrcoef(demand,resource_level)[0,1])

# Plotting and saving resources/demand by trial
for ii in range(len(demand_l)):
    pj_plt.plot_timeseries_with_trials(
        resource_l[ii],
        events_l[ii].trial_type,
        fir_bins=5,
        line_kwargs = {"color":"black"},
        include_line_labels=False,
        trial_alpha=0.30,
        xlabel="Time",
        ylabel="Level",
        title=f"Subject {ii+1} - Resource Level: {trans_l[ii]}",
        save_loc=os.path.join(base_out, f"figure_resource-trial-lineplot_{trans_l[ii][0:trans_ind[ii]]}_{trans_l[ii][-11:]}.tiff")
    )
    pj_plt.plot_timeseries_with_trials(
        demand_l[ii],
        events_l[ii].trial_type,
        fir_bins=5,
        line_kwargs = {"color":"black"},
        include_line_labels=False,
        trial_alpha=0.30,
        xlabel="Time",
        ylabel="Magnitude",
        title=f"Subject {ii+1} - Demand: {trans_l[ii]}",
        save_loc=os.path.join(base_out, f"figure_demand-trial-lineplot_{trans_l[ii][0:trans_ind[ii]]}_{trans_l[ii][-11:]}.tiff")
    )


impact_df = pd.DataFrame(impact_l,index=trans_l)
impact_df.to_csv(os.path.join(base_out, "table_impact.tsv"), sep="\t")

impact_corr_dict = {}
impact_corr_dict["dsp_resource"] = corr_dsp_resource_l
impact_corr_dict["dsp_demand"] = corr_dsp_demand_l
impact_corr_dict["demand_resource"] = corr_demand_resource_l

dsp_impact_corr_df = pd.DataFrame(impact_corr_dict,index=trans_l)

dsp_impact_corr_df.to_csv(os.path.join(base_out, "table_corr_dsp-impact.tsv"), sep="\t")


# compute_serial_bottleneck -------------------------------------------------------------------
bottelneck_l = []
for ii in range(len(demand_l)):
    bn=mc.compute_serial_bottleneck(
        info_l[ii]["labels"]["mode"],
        demand_l[ii]
    )[0]
    bottelneck_l.append(bn)


bottelneck_df = pd.DataFrame(bottelneck_l, columns = ["bottelneck"], index=trans_l)

bottelneck_df.to_csv(os.path.join(base_out, "table_bottleneck-index.tsv"), sep="\t")


#--------------------------
# Reporting/ plotting help
#--------------------------


# Identify nodes by mode/trial --------------------------------------------------------------
node_idxs = []
node_trial_idxs = []
for ii in range(len(bold_v)):
    [node_idx_pmode, node_pmode_mostprob_hmmstate,
        trial_node_idx_pmode, trial_node_pmode_mostprob_hmmstate] = ch.pcsm_nodes_and_states(
        bnode_l[ii],
        dsp_l[ii],
        posterior_probs_l[ii],
        trial_labels = events_l[0].trial_type,
        trial_onsets = events_l[0].onset,
        selection_mode = "scored",
        verbose=False
        )
    node_idxs.append(node_idx_pmode)
    node_trial_idxs.append(trial_node_idx_pmode)

# Plot spatial brain locations ____________________________________________
sub_labs = ["high-transition", "medium-transition", "low-transition"]
for ii in range(len(sub_labs)):
    pj_plt.plot_node_process_state_trans_simulation(
        node_idxs[ii], 
        bold_v[ii], 
        plt_type='stat',
        trial_node_idx = node_trial_idxs[ii],
        plt_save_dir = base_out,
        plt_subj_lab=sub_labs[ii],
        plt_show= False 
        )

sub_labs = ["high-transition", "medium-transition", "low-transition"]
for ii in range(len(sub_labs)):
    pj_plt.plot_node_process_state_trans_simulation(
        node_idxs[ii], 
        bold_v[ii], 
        plt_type="glass", 
        trial_node_idx = node_trial_idxs[ii],
        plt_save_dir = base_out,
        plt_subj_lab=sub_labs[ii],
        plt_show= False 
        )

# Helper Function Reporting -------------------------------------------------------------------
sch_atl = nilearn.datasets.fetch_atlas_schaefer_2018(
    n_rois = 200, 
    yeo_networks= 7, 
    resolution_mm= 2, 
    data_dir= atlas_loc
    )

descriptive_tables = []
descriptive_summary = []
for ii in range(len(posterior_probs_l)):
    out = ch.profile_hmm_states(
        posterior_probs_l[ii],
        dsp_l[ii],
        events_l[ii].onset,
        events_l[ii].trial_type,
        events_l[ii].Accuracy,
        events_l[ii].GoRT,
        demand=demand_l[ii],
        node_network_labels = sch_atl.labels[1:],
        bnode = bnode_l[ii],
        return_pandas = True
    )
    summ = ch.summarize_hmm_profile(out)
    descriptive_tables.append(out["tables"])
    descriptive_summary.append(summ)

ind = 0
lab_trans = ["high", "medium", "low"]
for ii in descriptive_tables:
    for jj in ii.keys():
        ii[jj].to_csv(os.path.join(base_out, "table-" + str(jj) + "_" + str(lab_trans[ind]) + "-transition.tsv"), sep="\t")
    ind += 1

