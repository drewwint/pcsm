#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Recovery of GMM-HMM Latent States and Transitions
Dr. Drew E. Winters
Created on 9/16/2025
"""



#------------------
# Setup
#------------------

# Loading packages -----------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
import statistics
from sklearn.metrics import confusion_matrix, classification_report
import pickle # for saving / loading hmmgmm model

from projlib import aligning_gmmhmm as al
from projlib import recovery_simulation as resim

# settign save base filename ------------------------------------------------------
base = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data"
base_fig = r"/home/dwinters/ascend-lab/publications/pcsm/code/90_figures"


# Loading Data and Global Model --------------------------------------------------

## Loading combinations of noise and transition profiles _______________
combinations = np.array(pd.read_csv(os.path.join(base,"simulated_participants","support_file-combinations_all_possible.csv")))

## loading data ________________________________________________________
bold_v = []
true_states_v = []
events_v = []
for ii in range(15000):
  bb = np.load(os.path.join(base,"simulated_participants",f"sim_subject-{ii+1}_BOLD_full.npy"))
  bold_v.append(bb)
  ts = np.load(os.path.join(base,"simulated_participants",f"sim_subject-{ii+1}_STATES.npy"))
  true_states_v.append(ts)
  edf = pd.read_csv(os.path.join(base,"simulated_participants",f"sim_subject-{ii+1}_events.csv"))
  events_v.append(edf)

# Loading global model from verification ________________________________
with open(os.path.join(base,"derivatives/models","sim_global_gmmhmm_model.pkl"), "rb") as file:
  global_model = pickle.load(file)



#---------------------------
# Running Individual Models
#---------------------------

# #*# NOTE #*# This approach 
#      # 1) constrains variance to global model but
#      # 2) frees individual parameters of individual means, transitions, start parameters, and weights to individually vary
#   # the advantage of this:
#       ## Models individual cognitive states but grounds it in shared dynamics
#       ## Variance retains a more reliable value without distorting estimates based on a single case  
#   # Interpretation is improved across individuals
#       ## States can be aligned to global model expectations which allows for between subject comparison more easily
#       ## Without this latent states have no intrinsic value and are unable to be compared across subjects
#   # The disadvantage
#     ## Individual BOLD related to each state is constrained to the global model - which constrains individual level BOLD estimates for each state -
#     ### HOWEVER:
#       #*# This is reasonable for our current purposes - 
#           # to identify cognitive states it is important to have a reasonable expected space (based on a global model)
#           # Decoding information processing depends on the parameters that are allowed to vary individually - not the individual variance
#           # allowing individual varinace to vary procuded unrealistic values and compromises estimates
#       #*# This does not impact the models ability to learn cognitive states from the brain data we input into it.
#       #*# Individual parameters that are most relevant are allowed to vary - individulal estimates are still learned by gmmhmm

# Setting up values for model
  # Number of hidden attention‐states C_t
n_states = 3 
  # Number of “response mixtures” states M_{i,t} is an element of {0,1}
n_mixtures = 2
  # Number of fir delays
n_delays = 5
  # epislon to ensure there are not division errors (by 0) when regularizing
epsilon = 1e-6

# Setting up lists to append to for outputs
  # Subject specific values
c_t_hat_l = []
log_lik_l = []
posterior_probs_l = []
alpha_mat_l = []
alpha_1_l = []
alpha_1_adj_l = []
trans_mat_l = []
lambda_vec_l = []
mean_mat_l = []
cov_mat_l = []
ebold_state_n = []
ebold_attn_n = []
individual_models = []

# Regularization of means and gamma (transition matrix)
    # To improve model estimation and state alignment we regularize the transition and bold intensity by state
    # Specifically - Soft regularization: blend with a weak Dirichlet-like prior for minor regulaization and numerical stability
  # Gamma transition matrix
global_reg_transmat = global_model.transmat_.copy()
global_reg_transmat = (1 - epsilon) * global_reg_transmat + epsilon * (1 / n_states)
  # Bold intensity by state
global_reg_means = global_model.means_.copy()
global_reg_means = (1 - epsilon) * global_reg_means + epsilon * (1 / n_states)

for ii in range(0,len(bold_v)):
  # fitting individual level models updated with starting parameters from the global_model to infomr individual model fitting
  subj_model = hmm.GMMHMM(
    n_components=n_states,
    n_mix = n_mixtures,
    covariance_type='diag',
    n_iter=100,
    random_state = 42,
                      # Specifying parameters so we can learn subject-specific cognitive dynamics
    init_params = '', # We have '' so we do not initialize anything before training - this is becasue we use the global model values as starting places (If we had these here it would throw an error)
    params = 'mtsw' # here we allow individual values to be updated during training for each individual 
                    ## so we learn these values from the individual subjects data
                    ## to derive subject-specific cognitive dynamics
                    ## The values we learn from each subject include:
                      #> m= means (Bold) by state and responding over time
                      #> t= transition (gamma), 
                      #> s= start (lambda), and 
                      #> w= weights (aka mixture weights)
                    ## The global model values are used as a prior for individual estimates 
                      # these are uses as a soft regularization to stay in expected ranges
                      # We intentionally fix covarainces from the global model so that:
                        #> shorter sequences of invidual timeseries don't cause erroneous values 
                        #> as individual variances are less stable 
                        #> that can cause unstable estimates and infinately small log transormations
    )

    # For individual models we use the parameters learned from the global_model
      # lambda
  subj_model.startprob_ = global_model.startprob_.copy()
      # mixture weights
  subj_model.weights_ = global_model.weights_.copy()
      # covariance
  subj_model.covars_ = global_model.covars_.copy()
      # transmat (with regularized global)
  subj_model.transmat_ = global_reg_transmat
      # bold intensity by state
  subj_model.means_ = global_reg_means

  # Fit via GMMHMM to each subject individually
  subj_len = np.array([n_delays] * bold_v[ii].shape[0])
    # Pulling each individuals data
  Y = np.concatenate(bold_v[ii],axis = 0)
    # Fitting individual models
  subj_model.fit(Y, lengths = subj_len)
      ## aligning states
  subj_model, order = al.align_model_to_expected(subj_model, global_model, wt_var=True)
    # saving individual model parameters
  c_t_hat = subj_model.predict(Y, lengths = subj_len)
  log_lik, posterior_probs = subj_model.score_samples(Y, lengths = subj_len)
  alpha_mat = np.column_stack(
      (
          subj_model.weights_[:,1], 1 - subj_model.weights_[:,1]
          )
      )
  alpha_1 = subj_model.weights_[:,1]
  alpha_1_adj = subj_model.weights_ @ subj_model.weights_.mean(axis=0) # adjusting alpha by probability of responding to either stimuli
  trans_mat = subj_model.transmat_
  lambda_vec = subj_model.startprob_
  mean_mat = subj_model.means_                # Note this will be the same for all subjects because we use the global model values to fix this across subjects
  cov_mat = subj_model.covars_
  expected_bold_state_by_node = (subj_model.means_ * subj_model.weights_[:,:,None]).mean(axis=1)  #(latent state, nodes)
  expected_bold_respond_by_node = (subj_model.means_ * subj_model.weights_[:,:,None]).mean(axis=0) #(respond, nodes)
    # appending values to lists
  c_t_hat_l.append(c_t_hat)
  log_lik_l.append(log_lik)
  posterior_probs_l.append(posterior_probs)
  alpha_mat_l.append(alpha_mat)
  alpha_1_l.append(alpha_1)
  alpha_1_adj_l.append(alpha_1_adj)
  trans_mat_l.append(trans_mat)
  lambda_vec_l.append(lambda_vec)
  mean_mat_l.append(mean_mat)
  cov_mat_l.append(cov_mat)
  ebold_state_n.append(expected_bold_state_by_node)
  ebold_attn_n.append(expected_bold_respond_by_node)
  individual_models.append(subj_model)


n_nodes = 200
duration = 15
## Saving outputs from model __________________________________________
for ii in range(len(c_t_hat_l)):
  # c_t_hat_l
  np.save(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_latent_state-hat"),c_t_hat_l[ii])
  # log_lik_l
  log_lik_per_obs = (log_lik_l[ii] / (n_delays * n_nodes + duration))
  np.save(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_log_like"),log_lik_per_obs)
  # posterior_probs_l
  np.save(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_posterior_probs"),posterior_probs_l[ii])
  pp_df = pd.DataFrame(posterior_probs_l[ii], columns = ["latent_state_1", "latent_state_2", "latent_state_3"])
  pp_df.to_csv(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_posterior_probs.csv"))
  # alpha_mat_l
  np.save(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_alpha_mat"),alpha_mat_l[ii])
  # alpha_1_l
  np.save(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_alpha_1"),alpha_1_l[ii])
  # alpha_1_adj_l
  np.save(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_alpha_1_adj"),alpha_1_adj_l[ii])
  # trans_mat_l
  np.save(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_trans_mat"),trans_mat_l[ii])
  # lambda_vec_l
  np.save(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_lambda_vec"),lambda_vec_l[ii])
  # mean_mat_l
  np.save(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_state_mean_mat"),mean_mat_l[ii])
  # cov_mat_l
  np.save(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_state_cov_mat"),cov_mat_l[ii])
  # ebold_state_n
  np.save(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_expected_bold_state_by_node"),ebold_state_n[ii])
  # ebold_attn_n
  np.save(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_expected_bold_respond_by_node"),ebold_attn_n[ii])
  # individual_models
  with open(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}.pkl"), "wb") as file:
    pickle.dump(individual_models[ii], file)

### Loading outputs from model
# c_t_hat_l = []
# log_lik_l = []
# alpha_mat_l = []
# alpha_1_adj_l = []
# trans_mat_l = []
# lambda_vec_l = []
# mean_mat_l = []
# # cov_mat_l = []
# ebold_state_n = []
# ebold_attn_n = []
# posterior_probs_l = []
# alpha_1_l = []

# for ii in range(15000):
#   # posterior_probs_l
#   pp = np.load(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_posterior_probs.npy"))
#   posterior_probs_l.append(pp)
#   # alpha_1_l
#   a1 = np.load(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_alpha_1.npy"))
#   alpha_1_l.append(a1)
#   # c_t_hat_l
#   ct = np.load(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_latent_state-hat.npy"))
#   c_t_hat_l.append(ct)
#   # log_lik_l
#   ll = np.load(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_log_like.npy"))
#   log_lik_l.append(ll)
#   # alpha_mat_l
#   aml = np.load(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_alpha_mat.npy"))
#   alpha_mat_l.append(aml)
#   # alpha_1_adj_l
#   aadj = np.load(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_alpha_1_adj.npy"))
#   alpha_1_adj_l.append(aadj)
#   # trans_mat_l
#   tm = np.load(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_trans_mat.npy"))
#   trans_mat_l.append(tm)
#   # lambda_vec_l
#   lv = np.load(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_lambda_vec.npy"))
#   lambda_vec_l.append(lv)
#   # mean_mat_l
#   mml = np.load(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_state_mean_mat.npy"))
#   mean_mat_l.append(mml)
#   # cov_mat_l
#   cm = np.load(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_state_cov_mat.npy"))
#   cov_mat_l.append(cm)
#   # ebold_state_n
#   ebs = np.load(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_expected_bold_state_by_node.npy"))
#   ebold_state_n.append(ebs)
#   # ebold_attn_n
#   eba = np.load(os.path.join(base,"derivatives/models",f"sim_subj_model-{ii+1}_expected_bold_respond_by_node.npy"))
#   ebold_attn_n.append(eba)



#---------------------------
# Recovery Reporting
#---------------------------

# Recovery from c_hat -----------------------------------------------------------------
## Recovering - Latent State_______________________________________________
ss = []
for ii in range(len(c_t_hat_l)):
  sss = []
  sss.append(resim.decode_state_from_c_hat(c_t_hat_l[ii], [5]*int(len(c_t_hat_l[ii])/5)))
  ss.append(np.array(sss[0]))

  ## Latent State - Across all subjects
    # initializing 0'ed out confusion matrix
trans_cm = np.zeros((3,3))
    # making a 0'ed out df to initialize df structure for overall calculation
trans_classify_df = pd.DataFrame(classification_report(true_states_v[0], ss[0], output_dict=True, target_names = ["State_0", "State_1", "State_2"],zero_division=1))
trans_classify_df[:] = 0 # specifying 0's for all cells
for ii in range(len(true_states_v)):
  # We add 0,1,2 to all subjects
    # becasue there are a few that do not have one of these latent states
    # this is the nature of their dat
    # however - to align to all other subjects we add one of all values
    # this ensure all states are represented but not impact the classification metrics
  true_state_i = np.append(true_states_v[ii], [0,1,2])
  predicted_state_i =np.append(ss[ii], [0,1,2])
  if ii < len(true_states_v)-1:
    trans_cm += confusion_matrix(true_state_i, predicted_state_i, normalize='true')
    trans_classify_df += pd.DataFrame(classification_report(true_state_i, predicted_state_i, output_dict=True, target_names = ["State_0", "State_1", "State_2"],zero_division=1))
  else:
    trans_cm += confusion_matrix(true_state_i, predicted_state_i, normalize='true')
    trans_cm_av = (trans_cm/(len(true_states_v)))
    trans_classify_df += pd.DataFrame(classification_report(true_state_i, predicted_state_i, output_dict=True, target_names = ["State_0", "State_1", "State_2"],zero_division=1))
    trans_classify_df_av = trans_classify_df/len(true_states_v)

print("\nAll Simulated Subjects Classification Report - Latent Attention State")
print(trans_classify_df_av)

  ## Saving classification matrix
trans_classify_df_av.to_csv(os.path.join(base,"derivatives/recovery","recovery_df_latent-state_c-hat_all-subj.csv"),index=True)

  ## Plotting/saving plot
sns.heatmap(trans_cm_av, annot=True, vmin=0, vmax=1, fmt=".3f", cmap="viridis")
plt.xlabel("Predicted State")
plt.ylabel("True State")
plt.xticks([0.5,1.5, 2.5],["State_0", "State_1", "State_2"])
plt.yticks([0.5,1.5, 2.5],["State_0", "State_1", "State_2"])
plt.title("All Simulated Subjects Recovery - Latent Attention States")
plt.savefig(os.path.join(base_fig,"recovery_figs","recovery_figure_latent-state_c-hat_all-subj.tiff"), dpi=400)
plt.show()
plt.close()

  ## Latent State - By Transition and noise type
ind = 0
for jj in np.arange(0, len(bold_v), per_comb_subj):
    # initializing 0'ed out confusion matrix
  trans_cm = np.zeros((3,3))
    # making a 0'ed out df to initialize df structure for overall calculation
  trans_classify_df = pd.DataFrame(classification_report(true_states_v[0], ss[0], output_dict=True, target_names = ["State_0", "State_1", "State_2"],zero_division=1))
  trans_classify_df[:] = 0 # specifying 0's for all cells
  for ii in range(len(true_states_v[jj:jj+per_comb_subj])):
  # We add 0,1,2 to all subjects
    # becasue there are a few that do not have one of these latent states
    # this is the nature of their data
    # however - to align to all other subjects we add one of all values
    # this ensures all states are represented but not impact the classification metrics
    true_state_i = np.append(true_states_v[jj:jj+per_comb_subj][ii], [0,1,2])
    predicted_state_i =np.append(ss[jj:jj+per_comb_subj][ii], [0,1,2])
    if ii < len(true_states_v[jj:jj+per_comb_subj])-1:
      trans_cm += confusion_matrix(true_state_i, predicted_state_i, normalize='true')
      trans_classify_df += pd.DataFrame(classification_report(true_state_i, predicted_state_i, output_dict=True, target_names = ["State_0", "State_1", "State_2"],zero_division=1))
    else:
      trans_cm += confusion_matrix(true_state_i, predicted_state_i, normalize='true')
      trans_cm_av = (trans_cm/(len(true_states_v[jj:jj+per_comb_subj])))
      trans_classify_df += pd.DataFrame(classification_report(true_state_i, predicted_state_i, output_dict=True, target_names = ["State_0", "State_1", "State_2"],zero_division=1))
      trans_classify_df_av = trans_classify_df/len(true_states_v[jj:jj+per_comb_subj])

  print("\nAll Simulated Subjects Classification Report - Latent State")
  print(trans_classify_df_av)
  ## saving classification dfs
  trans_classify_df_av.to_csv(os.path.join(base,"derivatives/recovery",f"recovery_df_latent-state_c-hat_trans-{combinations[ind][0]}_Noise-{combinations[ind][1]}.csv"),index=True)

    ## Plotting/saving plot
  sns.heatmap(trans_cm_av, annot=True, vmin=0, vmax=1, fmt=".3f", cmap="viridis")
  plt.xlabel("Predicted State")
  plt.ylabel("True State")
  plt.xticks([0.5,1.5, 2.5],["State_0", "State_1", "State_2"])
  plt.yticks([0.5,1.5, 2.5],["State_0", "State_1", "State_2"])
  plt.title(f"Recovery of  Latent States for: \n Transition Probability= {combinations[ind][0]} | Added BOLD Noise= {combinations[ind][1]}")
  ## Saving classification figures
  plt.savefig(os.path.join(base,"derivatives/recovery",f"recovery_figure_latent-state_c-hat_trans-{combinations[ind][0]}_Noise-{combinations[ind][1]}.tiff"), dpi=400)
  ind += 1
  plt.show()
  plt.close()


## Recovering: Transitions_______________________________________________
ch_tru = []
ch_est = []
for ii in range(len(ss)):
  tru = true_states_v[ii]
  est = ss[ii]
  ch_t = []
  ch_e = []
  for jj in range(1,len(tru)):
    ch_t.append((abs(tru[jj]-tru[jj-1])>0)*1)
    ch_e.append((abs(est[jj]-est[jj-1])>0)*1)
  ch_tru.append(np.array(ch_t))
  ch_est.append(np.array(ch_e))

  ## Transitions - Across all subjects
    # initializing 0'ed out confusion matrix
trans_cm = np.zeros((2,2))
    # making a 0'ed out df to initialize df structure for overall calculation
trans_classify_df = pd.DataFrame(classification_report(ch_tru[0], ch_est[0], output_dict=True, target_names = ["Stay", "Change"]))
trans_classify_df[:] = 0 # specifying 0's for all cells
for ii in range(len(ch_tru)):
  true_change_i = np.append(ch_tru[ii], [0,1])
  predicted_change_i =np.append(ch_est[ii], [0,1])
  if ii < len(ch_tru)-1:
    trans_cm += confusion_matrix(true_change_i, predicted_change_i, normalize='true')
    trans_classify_df += pd.DataFrame(classification_report(true_change_i, predicted_change_i, output_dict=True, target_names = ["Stay", "Change"]))
  else:
    trans_cm += confusion_matrix(true_change_i, predicted_change_i, normalize='true')
    trans_cm_av = (trans_cm/(len(ch_tru)))
    trans_classify_df += pd.DataFrame(classification_report(true_change_i, predicted_change_i, output_dict=True, target_names = ["Stay", "Change"]))
    trans_classify_df_av = trans_classify_df/len(ch_tru)

print("\nAll Simulated Subjects Classification Report - State Transition")
print(trans_classify_df_av)
  ## Saving classification matrix
trans_classify_df_av.to_csv(os.path.join(base,"derivatives/recovery","recovery_df_transition_c-hat_all-subj.csv"),index=True)

  ## Plotting/saving plot
sns.heatmap(trans_cm_av, annot=True, vmin=0, vmax=1, fmt=".3f", cmap="viridis")
plt.xlabel("Predicted State")
plt.ylabel("True State")
plt.xticks([0.5,1.5],["Stay", "Change"])
plt.yticks([0.5,1.5],["Stay", "Change"])
plt.title("All Simulated Subjects Recovery - State Transitions")
plt.savefig(os.path.join(base_fig,"recovery_figs","recovery_figure_transition_c-hat_all-subj.tiff"), dpi=400)
plt.show()
plt.close()



  ## Transitions - By Transition and noise type
ind = 0
for jj in np.arange(0, len(bold_v), per_comb_subj): 
  # initializing 0'ed out confusion matrix
  trans_cm = np.zeros((2,2))
  # making a 0'ed out df to initialize df structure for overall calculation
  trans_classify_df = pd.DataFrame(classification_report(ch_tru[0], ch_est[0], output_dict=True, target_names = ["Stay", "Change"]))
  trans_classify_df[:] = 0 # specifying 0's for all cells
  for ii in range(len(ch_tru[jj:jj+per_comb_subj])):
    true_change_i = np.append(ch_tru[jj:jj+per_comb_subj][ii], [0,1])
    predicted_change_i =np.append(ch_est[jj:jj+per_comb_subj][ii], [0,1]) 
    if ii < len(true_change_i)-1:
      trans_cm += confusion_matrix(true_change_i, predicted_change_i, normalize='true')
      trans_classify_df += pd.DataFrame(classification_report(true_change_i, predicted_change_i, output_dict=True, target_names = ["Stay", "Change"]))
    else:
      trans_cm += confusion_matrix(true_change_i, predicted_change_i, normalize='true')
      trans_cm_av = (trans_cm/(len(ch_tru[jj:jj+per_comb_subj])))
      trans_classify_df += pd.DataFrame(classification_report(true_change_i, predicted_change_i, output_dict=True, target_names = ["Stay", "Change"]))
      trans_classify_df_av = trans_classify_df/len(ch_tru[jj:jj+per_comb_subj])

  print("\nAll Simulated Subjects Classification Report - State Transition")
  print(trans_classify_df_av)
  ## saving classification dfs
  trans_classify_df_av.to_csv(os.path.join(base,"derivatives/recovery",f"recovery_df_transitions_c-hat_trans-{combinations[ind][0]}_Noise-{combinations[ind][1]}.csv"),index=True)

    ## Plotting/saving plot
  sns.heatmap(trans_cm_av, annot=True, vmin=0, vmax=1, fmt=".3f", cmap="viridis")
  plt.xlabel("Predicted State")
  plt.ylabel("True State")
  plt.xticks([0.5,1.5],["Stay", "Change"])
  plt.yticks([0.5,1.5],["Stay", "Change"])
  plt.title(f"Recovery of State Transitions for: \n Transition Probability= {combinations[ind][0]} | Added BOLD Noise= {combinations[ind][1]}")
  plt.savefig(os.path.join(base_fig,"recovery_figs",f"recovery_figure_transitions_c-hat_trans-{combinations[ind][0]}_Noise-{combinations[ind][1]}.tiff"), dpi=400)
  ind += 1
  plt.show()
  plt.close()



# Recovery from posterior------------------------------------------------------------
## Recovering: Latent States_______________________________________________
ss_pp = []
for ii in range(len(posterior_probs_l)):
  sss = []
  sss.append(resim.decode_state_from_posterior(posterior_probs_l[ii], [5]*int(len(posterior_probs_l[ii])/5)))
  ss_pp.append(np.array(sss[0]))

  ## Latent State - Across all subjects 
    # initializing 0'ed out confusion matrix
trans_cm = np.zeros((3,3))
# making a 0'ed out df to initialize df structure for overall calculation
trans_classify_df = pd.DataFrame(classification_report(true_states_v[0], ss_pp[0], output_dict=True, target_names = ["State_0", "State_1", "State_2"],zero_division=1))
trans_classify_df[:] = 0 # specifying 0's for all cells
for ii in range(len(true_states_v)):
  true_state_i = np.append(true_states_v[ii], [0,1,2])
  predicted_state_pp_i =np.append(ss_pp[ii], [0,1,2])
  if ii < len(true_states_v)-1:
    trans_cm += confusion_matrix(true_state_i, predicted_state_pp_i, normalize='true')
    trans_classify_df += pd.DataFrame(classification_report(true_state_i, predicted_state_pp_i, output_dict=True, target_names = ["State_0", "State_1", "State_2"],zero_division=1))
  else:
    trans_cm += confusion_matrix(true_state_i, predicted_state_pp_i, normalize='true')
    trans_cm_av = (trans_cm/(len(true_states_v)))
    trans_classify_df += pd.DataFrame(classification_report(true_state_i, predicted_state_pp_i, output_dict=True, target_names = ["State_0", "State_1", "State_2"],zero_division=1))
    trans_classify_df_av = trans_classify_df/len(true_states_v)

print("\nAll Simulated Subjects Classification Report - Latent Attention State")
print(trans_classify_df_av)

  ## Saving classification matrix
trans_classify_df_av.to_csv(os.path.join(base,"derivatives/recovery","recovery_df_latent-state_posterior_all-subj.csv"),index=True)

  ## Plotting/saving plot
sns.heatmap(trans_cm_av, annot=True, vmin=0, vmax=1, fmt=".3f", cmap="viridis")
plt.xlabel("Predicted State")
plt.ylabel("True State")
plt.xticks([0.5,1.5, 2.5],["State_0", "State_1", "State_2"])
plt.yticks([0.5,1.5, 2.5],["State_0", "State_1", "State_2"])
plt.title("All Simulated Subjects Recovery - Latent Attention States: Posterior")
plt.savefig(os.path.join(base_fig,"recovery_figs","recovery_figure_latent-state_posterior_all-subj.tiff"), dpi=400)
plt.show()
plt.close()



  ## Latent State - By Transition and noise type
ind = 0
for jj in np.arange(0, len(bold_v), per_comb_subj):
    # initializing 0'ed out confusion matrix
  trans_cm = np.zeros((3,3))
    # making a 0'ed out df to initialize df structure for overall calculation
  trans_classify_df = pd.DataFrame(classification_report(true_states_v[0], ss_pp[0], output_dict=True, target_names = ["State_0", "State_1", "State_2"],zero_division=1))
  trans_classify_df[:] = 0 # specifying 0's for all cells
  for ii in range(len(true_states_v[jj:jj+per_comb_subj])):
  # We add 0,1,2 to all subjects
    # becasue there are a few that do not have one of these latent states
    # this is the nature of their dat
    # however - to align to all other subjects we add one of all values
    # this ensure all states are represented but not impact the classification metrics
    true_state_i = np.append(true_states_v[jj:jj+per_comb_subj][ii], [0,1,2])
    predicted_state_pp_i =np.append(ss_pp[jj:jj+per_comb_subj][ii], [0,1,2])
    if ii < len(true_states_v[jj:jj+per_comb_subj])-1:
      trans_cm += confusion_matrix(true_state_i, predicted_state_pp_i, normalize='true')
      trans_classify_df += pd.DataFrame(classification_report(true_state_i, predicted_state_pp_i, output_dict=True, target_names = ["State_0", "State_1", "State_2"],zero_division=1))
    else:
      trans_cm += confusion_matrix(true_state_i, predicted_state_pp_i, normalize='true')
      trans_cm_av = (trans_cm/(len(true_states_v[jj:jj+per_comb_subj])))
      trans_classify_df += pd.DataFrame(classification_report(true_state_i, predicted_state_pp_i, output_dict=True, target_names = ["State_0", "State_1", "State_2"],zero_division=1))
      trans_classify_df_av = trans_classify_df/len(true_states_v[jj:jj+per_comb_subj])

  print("\nAll Simulated Subjects Classification Report - Latent State")
  print(trans_classify_df_av)
  ## saving classification dfs
  trans_classify_df_av.to_csv(os.path.join(base,"derivatives/recovery",f"recovery_df_latent-state_posterior_trans-{combinations[ind][0]}_Noise-{combinations[ind][1]}.csv"),index=True)

    ## Plotting/saving plot
  sns.heatmap(trans_cm_av, annot=True, vmin=0, vmax=1, fmt=".3f", cmap="viridis")
  plt.xlabel("Predicted State")
  plt.ylabel("True State")
  plt.xticks([0.5,1.5, 2.5],["State_0", "State_1", "State_2"])
  plt.yticks([0.5,1.5, 2.5],["State_0", "State_1", "State_2"])
  plt.title(f"Recovery of  Latent States for: \n Transition Probability= {combinations[ind][0]} | Added BOLD Noise= {combinations[ind][1]}")
  ## Saving classification figures
  plt.savefig(os.path.join(base_fig,"recovery_figs",f"recovery_figure_latent-state_posterior_trans-{combinations[ind][0]}_Noise-{combinations[ind][1]}.tiff"), dpi=400)
  ind += 1
  plt.show()
  plt.close()



## Recovering: Transitions_______________________________________________
ch_tru_pp = []
ch_est_pp = []
for ii in range(len(ss_pp)):
  tru = true_states_v[ii]
  est = ss_pp[ii]
  ch_t = []
  ch_e = []
  for jj in range(1,len(tru)):
    ch_t.append((abs(tru[jj]-tru[jj-1])>0)*1)
    ch_e.append((abs(est[jj]-est[jj-1])>0)*1)
  ch_tru_pp.append(np.array(ch_t))
  ch_est_pp.append(np.array(ch_e))


  ## Transitions - across entire sample
    # initializing 0'ed out confusion matrix
trans_cm = np.zeros((2,2))
    # making a 0'ed out df to initialize df structure for overall calculation
trans_classify_df = pd.DataFrame(classification_report(ch_tru_pp[0], ch_est_pp[0], output_dict=True, target_names = ["Stay", "Change"]))
trans_classify_df[:] = 0 # specifying 0's for all cells
for ii in range(len(ch_tru)):
  true_change_i = np.append(ch_tru_pp[ii], [0,1])
  predicted_change_i =np.append(ch_est_pp[ii], [0,1])
  if ii < len(ch_tru)-1:
    trans_cm += confusion_matrix(true_change_i, predicted_change_i, normalize='true')
    trans_classify_df += pd.DataFrame(classification_report(true_change_i, predicted_change_i, output_dict=True, target_names = ["Stay", "Change"]))
  else:
    trans_cm += confusion_matrix(true_change_i, predicted_change_i, normalize='true')
    trans_cm_av = (trans_cm/(len(ch_tru)))
    trans_classify_df += pd.DataFrame(classification_report(true_change_i, predicted_change_i, output_dict=True, target_names = ["Stay", "Change"]))
    trans_classify_df_av = trans_classify_df/len(ch_tru)

print("\nAll Simulated Subjects Classification Report - State Transition")
print(trans_classify_df_av)
  ## Saving classification matrix
trans_classify_df_av.to_csv(os.path.join(base,"derivatives/recovery","recovery_df_transition_posterior_all-subj.csv"),index=True)

  ## Plotting/saving plot
sns.heatmap(trans_cm_av, annot=True, vmin=0, vmax=1, fmt=".3f", cmap="viridis")
plt.xlabel("Predicted State")
plt.ylabel("True State")
plt.xticks([0.5,1.5],["Stay", "Change"])
plt.yticks([0.5,1.5],["Stay", "Change"])
plt.title("All Simulated Subjects Recovery - State Transitions")
plt.savefig(os.path.join(base_fig,"recovery_figs","recovery_figure_transition_posterior_all-subj.tiff"), dpi=400)
plt.show()
plt.close()



  ## Transitions - By Transition and noise type
ind = 0
for jj in np.arange(0, len(bold_v), per_comb_subj): 
  # initializing 0'ed out confusion matrix
  trans_cm = np.zeros((2,2))
  # making a 0'ed out df to initialize df structure for overall calculation
  trans_classify_df = pd.DataFrame(classification_report(ch_tru_pp[0], ch_est_pp[0], output_dict=True, target_names = ["Stay", "Change"]))
  trans_classify_df[:] = 0 # specifying 0's for all cells
  for ii in range(len(ch_tru_pp[jj:jj+per_comb_subj])):
    true_change_i = np.append(ch_tru_pp[jj:jj+per_comb_subj][ii], [0,1])
    predicted_change_i =np.append(ch_est_pp[jj:jj+per_comb_subj][ii], [0,1]) 
    if ii < len(true_change_i)-1:
      trans_cm += confusion_matrix(true_change_i, predicted_change_i, normalize='true')
      trans_classify_df += pd.DataFrame(classification_report(true_change_i, predicted_change_i, output_dict=True, target_names = ["Stay", "Change"]))
    else:
      trans_cm += confusion_matrix(true_change_i, predicted_change_i, normalize='true')
      trans_cm_av = (trans_cm/(len(ch_tru_pp[jj:jj+per_comb_subj])))
      trans_classify_df += pd.DataFrame(classification_report(true_change_i, predicted_change_i, output_dict=True, target_names = ["Stay", "Change"]))
      trans_classify_df_av = trans_classify_df/len(ch_tru_pp[jj:jj+per_comb_subj])

  print("\nAll Simulated Subjects Classification Report - State Transition")
  print(trans_classify_df_av)
  ## saving classification dfs
  trans_classify_df_av.to_csv(os.path.join(base,"derivatives/recovery",f"recovery_df_transitions_posterior_trans-{combinations[ind][0]}_Noise-{combinations[ind][1]}.csv"),index=True)

    ## Plotting/saving plot
  sns.heatmap(trans_cm_av, annot=True, vmin=0, vmax=1, fmt=".3f", cmap="viridis")
  plt.xlabel("Predicted State")
  plt.ylabel("True State")
  plt.xticks([0.5,1.5],["Stay", "Change"])
  plt.yticks([0.5,1.5],["Stay", "Change"])
  plt.title(f"Recovery of State Transitions for: \n Transition Probability= {combinations[ind][0]} | Added BOLD Noise= {combinations[ind][1]}")
  plt.savefig(os.path.join(base_fig,"recovery_figs",f"recovery_figure_transitions_posterior_trans-{combinations[ind][0]}_Noise-{combinations[ind][1]}.tiff"), dpi=400)
  ind += 1
  plt.show()
  plt.close()


