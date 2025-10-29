#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for simulating fMRI time series and behavioral data for a stop signal task with known attentional states.
Dr. Drew E. Winters
Created on 9/16/2025
"""

import numpy as np
import pandas as pd
import scipy.stats


# function to vary transition probabilities -----------------------------------------------
def generate_state_sequence(n, change='low'):
    """
    Probability matrix of changing attentional state
    Values:
    - probs: an array that stores the probability of staying in the current state. The values in this array depend on the stability parameter passed to the function.
    - In the line if np.random.rand() < probs[last]:, last is the index of the previously visited state (0, 1, or 2). So, probs[last] retrieves the probability of remaining in the state represented by the last index.
    """
    if change == 'high': # a lower liklihood of staying in any state = higher liklihood of changing states
        probs = np.array([0.3, 0.25, 0.3])
    elif change == 'medium': # a moderate liklihood of staying in any state
        probs = np.array([0.5, 0.55, 0.5])
    elif change == 'low': # A higher liklihood of staying in the same state for any state = higher liklihood of staying in current state
        probs = np.array([0.9, 0.95, 0.9])
    else:
        raise ValueError("Invalid stability")
    states = [np.random.choice([0, 1, 2])]
    for _ in range(1, n):
        last = states[-1]
        if np.random.rand() < probs[last]:
            states.append(last)
        else:
            states.append(np.random.choice([s for s in [0, 1, 2] if s != last]))
    return np.array(states)


# function to run simulation -------------------------------------------------------------
def simulate_sst_with_ground_truth(
    n_nodes=5,
    go_prop=0.6,
    go_rt_dist=(0.4, 0.6),
    stop_process_dist=(0.2, 0.3),
    ssd_range=(0.1, 0.4),
    transmat_state_change ='low',                # 'high', 'medium', or 'low'
    state_assignment_strategy="random",          # or "by_trial_type"
    duration=500,
    tr=2.0,
    min_isi=2.0,
    max_isi=4.0,
    hrf_duration=15,
    noise_profile="default",                     # "low", "medium", "high", or "veryhigh" can also use "custom" and input value in noise_level
    noise_level=0.2,
    random_state=42
    ):
    """
    Simulate fMRI time series and behavioral data for a stop signal task with known attentional states.
    """
    # Setting random state
    if random_state is not None:
        np.random.seed(random_state)
    total_trials_estimate = int(duration / ((min_isi + max_isi) / tr))
    # Generate trial types
    trial_types = np.random.choice(
        [0, 1],  # 0 = Go, 1 = Stop
        size=total_trials_estimate,
        p=[go_prop, 1 - go_prop]
    )
        # setting alpha profile as specified         
    if noise_profile == 'veryhigh':  # Very High Noise (SNR <1, likely partial or failed recovery)
        noise_level = 0.55
    elif noise_profile == 'high':    # High Noise (SNR ~1, challenging)
        noise_level = 0.35
    elif noise_profile == 'medium':  # Moderate Noise (SNR moderate, still feasible) this is the default
        noise_level = 0.2
    elif noise_profile == 'low':     # Low Noise (SNR high, easy decoding)
        noise_level = 0.1
    elif noise_profile == 'custom':  # Custom allows you to inset any value into noise_level - if no noise level is entered then the default of 0.2 is used
        noise_level = noise_level
    elif noise_profile == 'default':
        noise_level = 0.2
    elif noise_profile == 'none':
        noise_level = 0
    else:
        raise ValueError("Invalid noise_profile")
    # Generate ISIs and trial onsets
    isis = np.random.uniform(min_isi, max_isi, total_trials_estimate)
    isis = isis * (duration - hrf_duration) / np.sum(isis)
    trial_onsets = np.cumsum(isis)
    valid_idx = trial_onsets < duration - hrf_duration
    trial_onsets = trial_onsets[valid_idx]
    trial_types = trial_types[valid_idx]
    total_trials = len(trial_onsets)
    trial_duration = []
    for ii in range(len(trial_onsets)):
      if ii+1 < len(trial_onsets):
        trial_duration.append(trial_onsets[ii+1]-trial_onsets[ii])
      else:
        trial_duration.append(np.array(trial_duration).mean())
    # Assign ground-truth attentional states per trial
    if state_assignment_strategy == "random":
        true_states = generate_state_sequence(total_trials, transmat_state_change)
    elif state_assignment_strategy == "by_trial_type":
        true_states = np.where(trial_types == 0, 0, np.random.choice([1, 2], size=total_trials))
    else:
        raise ValueError("Invalid state_assignment_strategy")
    behavioral_events = []
    trial_list_data = []
    def sample_go_rt(): return np.random.uniform(*go_rt_dist)
    def sample_stop_process(): return np.random.uniform(*stop_process_dist)
    for i in range(total_trials):
        onset_time = trial_onsets[i]
        duration_trial = trial_duration[i]
        is_stop = trial_types[i] == 1
        state = true_states[i]
        go_rt = sample_go_rt()
        go_time = onset_time + go_rt
        ssd = np.nan
        stop_success_time = np.nan
        if is_stop:
            ssd = np.random.uniform(*ssd_range)
            stop_time = onset_time + ssd
            stop_process_time = sample_stop_process()
            stop_success_time = stop_time + stop_process_time
            if stop_success_time < go_time:
                outcome = "SuccessfulStop"
                behavioral_events.append({'time': onset_time, 'label': 0})
                behavioral_events.append({'time': stop_time, 'label': 1})
            else:
                outcome = "UnsuccessfulStop"
                behavioral_events.append({'time': onset_time, 'label': 0})
                behavioral_events.append({'time': stop_time, 'label': 1})
                behavioral_events.append({'time': go_time, 'label': 2})
        else:
            outcome = "Go"
            behavioral_events.append({'time': onset_time, 'label': 0})
            behavioral_events.append({'time': go_time, 'label': 2})
        trial_list_data.append({
            'onset': onset_time,
            'duration': duration_trial,
            'trial_type': 'Stop' if is_stop else 'Go',
            'TrialOutcome': outcome,
            'Accuracy': int(outcome in ["Go", "SuccessfulStop"]),
            'GoRT': go_rt,
            'SSD': ssd,
            'State': state
        })
    behavioral_df = pd.DataFrame(trial_list_data)
    behavioral_df['trial_type_Code'] = behavioral_df['trial_type'].map({'Go': 0, 'Stop': 1})
    behavioral_events.sort(key=lambda x: x['time'])
    # Simulate BOLD
    fir_delays = [0, 2, 4, 6, 8]  # in seconds
    n_bins = len(fir_delays)
    fir_data = np.zeros((total_trials, n_bins, n_nodes))
    np.random.seed(random_state)
    state_node_weights = {
        0: np.random.rand(n_nodes) * 0.6 + 0.2,  # High and positive amplitude
        1: np.random.rand(n_nodes) * 0.2 + 0.1,  # Low amplitude
        2: np.random.rand(n_nodes) * 0.4 - 0.2   # Mixed amplitude positive/negative)
    }
    for trial in range(total_trials):
        state = true_states[trial]
        node_weights = state_node_weights[state]
        for bin_idx in range(n_bins):
            for node in range(n_nodes):
                fir_data[trial, bin_idx, node] = node_weights[node] + np.random.normal(0, noise_level)
    return scipy.stats.zscore(fir_data), behavioral_df, true_states

