#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Ground-truth Simulation for PCSM recovery
Dr. Drew E. Winters
Created on 9/16/2025
"""

#-----------------------
# Setup/loading packages 
# ----------------------

# packages -----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import itertools
from projlib import simulation as sml

# settign save base filename -----------------------------------------------------------
base = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/simulated_participants"
fig_save = r"/home/dwinters/ascend-lab/publications/pcsm/code/90_figures/"



#-----------------
# Data Simulation 
# ----------------

# Genearting list of transition and noise profiles -----------------------
t_prob = ["high", "medium", "low"]
noise_profile = ["veryhigh", "high", "medium", "low"]
combinations = list(itertools.product(t_prob, noise_profile))

combinations_df = pd.DataFrame(np.vstack(combinations), columns = ["Transition Probability", "BOLD Noise"])

# Saving combinations
combinations_df.to_csv(os.path.join(base,"support_file-combinations_all_possible.csv"), index=False)

print(f"Possible combinations = {len(combinations_df)} \n",combinations_df)



# Simulating data ---------------------------------------------------------
## Initiating Lists
bold_v = []
events_v = []
true_states_v = []
per_comb_subj = 1250 # define how many subjects to simulate per combination of transition probability and noise level
  # 12 total combinations of noise and transition probabilities
  # 1250 subjects per combination
  # 12 * 1250 = 15000 total participants are simulated
n_nodes=200
duration=400

## Running simulation
for ii in combinations:
  for i in range(per_comb_subj):
    bo, be, trs = sml.simulate_sst_with_ground_truth(
        n_nodes=200,
        duration=400,
        random_state=42+i,
        transmat_state_change=ii[0],       # 'high', 'medium', or 'low'
        noise_profile=ii[1],               # 'low', 'medium', 'high', 'veryhigh'
        state_assignment_strategy="random" # "random" or "by_trial_type"
    )
    bold_v.append(bo)
    events_v.append(be)
    true_states_v.append(trs)

# Saving outputs
for ii in range(len(bold_v)):
  np.save(os.path.join(base,f"sim_subject-{ii+1}_BOLD_full"),bold_v[ii])
  np.save(os.path.join(base,f"sim_subject-{ii+1}_STATES"),true_states_v[ii])
  pd.DataFrame(bold_v[ii].mean(axis=1)).to_csv(os.path.join(base,f"sim_subject-{ii+1}_BOLD_avg-fir.csv"))
  events_v[ii].to_csv(os.path.join(base,f"sim_subject-{ii+1}_events.csv"))



## Visualizing Simulated BOLD _________________________________________

### Initalizing lists
go_trials_concatenated = []
stop_trials_concatenated = []

max_sub = input(f"Enter the number of subjects you want to plot in total out of max {len(bold_v)} =  ")
max_sub = int(max_sub)
random_sub_ts = np.random.choice(range(0,len(bold_v)), max_sub)


### Plotting Code

for subject_idx in random_sub_ts:
    plt.figure(figsize=(20, 4))
    subject_fir_data = bold_v[subject_idx] # shape: (n_trials, n_bins, n_parcels)
    subject_labels = events_v[subject_idx].trial_type # Series of labels ("GO", "STOP", "WAIT")
    # subject_events_original = events_v[subject_idx].loc[events_v[subject_idx]["TrialOutcome"].notna()].copy().reset_index(drop=True) # Original events for onset/duration
    # Concatenate all trials' FIR responses along the bins dimension for a single timeseries
    # Shape: (n_trials * n_bins, n_parcels)
    all_trials_concatenated = subject_fir_data.reshape(-1, subject_fir_data.shape[-1])
    # Select data for the specified nodes
    node=np.array(range(15))
    selected_nodes_data = all_trials_concatenated[:, node] # Shape: (n_trials * n_bins, n_selected_nodes)
    # Plot individual lines for each selected node
    for node_idx in range(selected_nodes_data.shape[1]):
        plt.plot(np.arange(selected_nodes_data.shape[0]),
                 selected_nodes_data[:, node_idx],
                 label=f'Node {node_idx}') # Use node index and name for label
    # Add background coloring for trial periods
    trial_type_colors = {
        'Go': 'olivedrab',
        'Stop': 'orangered'
    }
    current_onset_in_concatenated_fir = 0
    # Create lists to store handles and labels for the legend
    legend_handles = []
    legend_labels = []
    for trial_idx in range(len(subject_labels)):
        trial_type = subject_labels.iloc[trial_idx]
        duration_in_fir_bins = 5 #n_bins
        # Get color for the trial type
        color = trial_type_colors.get(trial_type, 'white') # Default to white if type not in dict
        # Add the background span and store the handle and label
        if trial_type not in legend_labels:
            span = plt.axvspan(current_onset_in_concatenated_fir,
                        current_onset_in_concatenated_fir + duration_in_fir_bins,
                        facecolor=color, alpha=0.2, label=trial_type)
            legend_handles.append(span)
            legend_labels.append(trial_type)
        else:
             plt.axvspan(current_onset_in_concatenated_fir,
                        current_onset_in_concatenated_fir + duration_in_fir_bins,
                        facecolor=color, alpha=0.2)
        # Move to the onset of the next trial in the concatenated timeseries
        current_onset_in_concatenated_fir += duration_in_fir_bins
    plt.xlabel("Concatenated Timepoints (FIR Bins)")
    plt.ylabel("FIR Response (Effect Size)")
    plt.title(f"Subject {subject_idx + 1} Individual Node FIR Responses Across All Trials")
    # Create a legend using the collected handles and labels
    plt.legend(legend_handles, legend_labels, loc= 'lower left')
    plt.grid(True)
    # Saving figures
    plt.savefig(os.path.join(fig_save, "timeseries-raw_figs", f"sim_subject-{subject_idx+1}_BOLD_figure_rand_select.tiff"), dpi=400)
    plt.show()
    plt.close()


