#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Ground-truth recovery for PCSM
Dr. Drew E. Winters
Created on 9/16/2025

first activate environment:
source /home/dwinters/ascend-lab/publications/pcsm/code/env/venv_pcsm/bin/activate
"""

#------------------
# Setup/loading packages 
# -----------------

# packages -----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from hmmlearn import hmm
import scipy
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold, GroupKFold
from projlib import cross_validation_verification as cvv
import pickle # for saving hmmgmm model after fitted

from joblib import Parallel, delayed, Memory

from sklearn.mixture import GaussianMixture

# settign save base filename -----------------------------------------------------------
memory = Memory(location="/home/dwinters/ascend-lab/publications/pcsm/work/joblib_cache", verbose=0)
base = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/simulated_participants"
base_save = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/derivatives/recovery"


#----------------------
# Loading data
#----------------------

bold_v = []
true_states_v = []
events_v = []
for ii in range(15000):
  bb = np.load(os.path.join(base,f"sim_subject-{ii+1}_BOLD_full.npy"))
  bold_v.append(bb)
  ts = np.load(os.path.join(base,f"sim_subject-{ii+1}_STATES.npy"))
  true_states_v.append(ts)
  edf = pd.read_csv(os.path.join(base,f"sim_subject-{ii+1}_events.csv"))
  events_v.append(edf)


#----------------------------------------------
# Verifying recovery of mixtures/latent states
#----------------------------------------------

# selecting 100 nodes at random
nodes = [np.random.randint(0,200) for _ in range(100)]
nodes_df = pd.DataFrame(nodes,columns=["node_idx"])
nodes_df.to_csv(os.path.join(base_save,"nodes_100_random.tsv"), sep="\t")

# Recovering the number of expected latent states____________________
Y = np.array(np.concatenate(bold_v,axis=0), dtype= np.float32)
Y = Y.reshape(-1,200)[:,np.r_[nodes]]


results_df, details = cvv.run_gmmhmm_cv(
  Y,
  state_range=range(2,6),
  n_mixtures=2,
  n_splits=5,
  n_starts=3,
  covariance_type="diag",
  random_state=5745,
  n_jobs=-1,
  n_iter = 50
)


# given that we only used 1/2 of the nodes due to computing efficiency - we adjust for accurate values by *2
results_df[["mean_LL","SE_LL","delta_LL"]] = results_df[["mean_LL","SE_LL","delta_LL"]]*2

for ii in details.keys():
  for jj in ['LL_per_obs_folds', 'mean_LL_per_obs', 'SE_LL_per_obs']:
    if details[ii][jj].size > 1:
      for kk in range(0,np.array(details[ii][jj]).size): 
        details[ii][jj][kk] = (details[ii][jj][kk]*2)
    else:
      details[ii][jj] = (details[ii][jj]*2)

# rerun select_best_k
results_df = cvv.select_best_k(results_df)
    # >>> results_df
    #    latent_states     mean_LL     SE_LL  delta_LL  perc_folds  selected_model
    # 0              2 -245.774754  0.012767       NaN         NaN           False
    # 1              3 -242.300903  0.013635  3.473851         1.0            True
    # 2              4 -240.991421  0.092862  1.309482         1.0           False
    # 3              5 -240.145536  0.014384  0.845885         1.0           False


## Saving generlizability stats
  # overall generlizability across folds
results_df.to_csv(os.path.join(base_save,"verify_latent-state_cv-generlizability_mix-state.csv"))

  # details per fold
for ii in details.keys():
  ddf = pd.DataFrame({k: pd.Series(v) for k, v in details[ii].items()}).iloc[0:5,:]
  ddf["fold"] = [1,2,3,4,5]
  ddf = ddf.loc[:,["fold", "LL_per_obs_folds", "mean_LL_per_obs", "SE_LL_per_obs", "fold_sizes"]]
  print(ddf)
  ddf.to_csv(os.path.join(base_save,f"verify_latent-state_cv-details_fold-{ii}_mix-state.csv"))




#------------------------------
# Evaluating Emperical Priors
#------------------------------

## Estimating Global GMMHMM model
  # Setting up values for model
    # Number of hidden states C_t
n_states = 3 
    # Number of “attended‐stimulus” states X_{i,t} is an element of {0,1}
n_mixtures = 2
    # Number of fir delays
n_delays = 5
    # epislon to ensure there are not division errors (by 0) when regularizing
epsilon = 1e-6

  # GMMHMM
global_model = hmm.GMMHMM(
    n_components=n_states,
    n_mix = n_mixtures,
    covariance_type='diag',
    n_iter=100, 
    verbose=False,
    random_state = 42,
      # We dont specify any initialized or learned parameters here becasue the default is to have all parameters initialize before training and update during trainining
        ## We want to learn parameters from the global model so we will use these for the subject specific model
        ## Specifically
          ### We want to fix covarainces from the global model to the individual model
          ### But we will use the other parameters as starting points
)

  # Pull all subjects data together to obtain sample level attentional states
ts_concat = np.concatenate(np.concatenate(bold_v,axis=0),axis=0)
global_len = np.array([n_delays] * int(ts_concat.shape[0]/n_delays))

  # Fit global (all subjects) model
global_model.fit(ts_concat, lengths = global_len)

# Saving model/data
with open(os.path.join(base,"models","sim_global_gmmhmm_model.pkl"), "wb") as file:
  pickle.dump(global_model, file)

np.save(os.path.join(base,"models","global_model_concatenated_ts"),ts_concat)
np.save(os.path.join(base,"models","global_model_concatenated_lengths"),global_len)


# # to later load:
# # with open(os.path.join(base,"models","sim_global_gmmhmm_model.pkl"), "rb") as file:
# #   global_model = pickle.load(file)

# with open(r"D:\CU OneDrive Sync\1 Publications\Methods_cognitive_state_decoding\cog_state_simulation_output\models\sim_subj_model-14990.pkl", "rb") as file:
#   modd = pickle.load(file)



## Cross validation for using global model as emperical priors
  # Cross validating across entire sample
    # getting indicies for folds
folds = 5
n_subjects = len(bold_v)
kf = KFold(n_splits=folds, shuffle=True, random_state=42)

splits = list(kf.split(np.arange(n_subjects)))

    # running Cross Validation 
fold_models = []
fold_scores = []

for train_idx, test_idx in splits:
    # Build training data
    train_data = np.concatenate([
        np.concatenate(bold_v[i], axis=0) for i in train_idx
    ], axis=0)
    train_lengths = np.array([n_delays] * (train_data.shape[0] // n_delays))
    # Fit global model
    model = hmm.GMMHMM(
        n_components=n_states,
        n_mix=n_mixtures,
        covariance_type='diag',
        n_iter=100,
        verbose=False,
        random_state=42
    )
    model.fit(train_data, lengths=train_lengths)
    fold_models.append(model)
    # Evaluate log-likelihood on held-out test data
    test_data = np.concatenate([
        np.concatenate(bold_v[i], axis=0) for i in test_idx
    ], axis=0)
    test_lengths = np.array([n_delays] * (test_data.shape[0] // n_delays))
    score = model.score(test_data, lengths=test_lengths)
    fold_scores.append(score)


## Investigating output of cross validation
  # Normalizing liklihood values by per observation n observations
    # This helps interpretabilty and more accuratly defines metrics
      # (n_delays * n_nodes + duration of timeseries) * n_subj-in-fold (total_n/folds_n)
obs_norm = (n_delays * n_nodes + duration) * (15000/5)
cv_df = pd.DataFrame({"Fold": range(5), "log-likelihood": (np.array(fold_scores)/obs_norm)})
print(f"CV per-observation Values by fold: \n{cv_df.to_string()}")
          #>> CV per-observation Values by fold: 
          #>>    Fold  log-likelihood
          #>>       Fold  log-likelihood
          #>>    0     0     -113.513204
          #>>    1     1     -114.077999
          #>>    2     2     -113.785938
          #>>    3     3     -114.047833
          #>>    4     4     -114.356937
print(f"CV per-observation Descriptives: \n{cv_df["log-likelihood"].describe().loc[["mean", "50%","std", "min","max"]].to_string()}")
          #>> CV per-observation Descriptives: 
          #>> mean   -113.956382
          #>> 50%    -114.047833
          #>> std       0.319787
          #>> min    -114.356937
          #>> max    -113.513204
### Test of vs values being within 2SDs
m_cv = -113.956382
sd_cv = 0.319787
min_cv =  -114.356937
max_cv =  -113.513204
m_cv + sd_cv*2 > max_cv # True
m_cv - sd_cv*2 < min_cv # True
      # This suggests that the values are consistent across folds and with in an expected range - so good performance
      # because this is only for data checking we will now only use the global model for the bext steps


  # test that bold responses are similar between CV model and global model bold by state
    ## calculating cv metrics across folds
means_all = np.stack([m.means_ for m in fold_models], axis=0)
means_avg = np.mean(means_all, axis=0)

    ## mean bold by state
      # - we use a dice coefficient for improves interpretability of distance between models
mean_dist = cdist(means_avg.mean(axis=1), global_model.means_.mean(axis=1), 'dice').mean(axis=1)
if all(mean_dist) >0.9:
  print(f"Good - Close BOLD between CV and global model! Average Dice = {round(mean_dist.mean())}")
else:
  print(f"Issue - **NOT CLOSE** BOLD between CV and Global model :( Average Dice = {round(mean_dist.mean())}")

          #>> Good - Close BOLD between CV and global model! Average Dice >0.9


    ## standard deviation by state
      # - we use a dice coefficient for improves interpretability of distance between models 
std_dist = cdist(means_avg.std(axis=1), global_model.means_.std(axis=1), 'dice').mean(axis=1)
if all(std_dist) >0.9:
  print(f"Good - Close variance between CV and global model! Average Dice = {round(mean_dist.mean())}")
else:
  print(f"Issue - **NOT CLOSE** variance between CV and Global model :( Average Dice = {round(mean_dist.mean())}")

          #>> Good - Close variance between CV and global model! Average Dice > 0.9


## Saving files
  # splits
for ii in range(len(splits)):
  np.save(os.path.join(base,"recovery",f"verify_cv_indicies_fold-{ii+1}_train"),splits[ii][0])
  np.save(os.path.join(base,"recovery",f"verify_cv_indicies_fold-{ii+1}_test"),splits[ii][1])


  # cv_df
cv_df.to_csv(os.path.join(base,"recovery","verify_cv_log-like_df.csv"))

  # fold_models
for ii in range(len(fold_models)):
  with open(os.path.join(base,"recovery",f"verify_sim_cv_model_fold_{ii+1}.pkl"), "wb") as file:
    pickle.dump(fold_models[ii], file)


  # mean_dist
np.save(os.path.join(base,"recovery","verify_cv_distance_mean"),mean_dist)
# mean_dist = np.load(os.path.join(base,"recovery","verify_cv_distance_mean.npy"))

  # std_dist
np.save(os.path.join(base,"recovery","verify_cv_distance_std"),std_dist)
# std_dist = np.load(os.path.join(base,"recovery","verify_cv_distance_std.npy"))

cv_dist_df = pd.DataFrame({"mean_dist": mean_dist, "std_dist": std_dist})
# cv_dist_df.to_csv(os.path.join(base,"recovery","verify_cv_distance_dice-avg_df.csv"))

### Distance reporting
pd.DataFrame({"dice": round(cv_dist_df.mean(),3), "std": round(cv_dist_df.std(),3)}, index=["dice","std"])
  #               dice    std
  # mean_dist    0.977   0.040
  # std_dist     0.934   0.026

## CONCLUSION:
    ##> All in all it looks like we have a good relationship between the global model and CV estimates
    ##> suggests this alignment would align with out of sample predictions





