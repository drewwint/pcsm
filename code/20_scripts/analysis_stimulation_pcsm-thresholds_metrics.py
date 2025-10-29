#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
PCSM Metric Thresholds
Dr. Drew E. Winters
Created on 9/16/2025
"""


#---------------------
# Setup
#---------------------

# packages---------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from scipy.special import expit, logit

from projlib import metric_calculation as mc
from projlib import metric_thresholding as mt

# settign save base filename-----------------------------------------------------------
load_dat = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/simulated_participants"
load_mod = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/derivatives/models"
base_out = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/derivatives/thresholds"
fig_out = r"/home/dwinters/ascend-lab/publications/pcsm/code/90_figures/threshold_figs"


#----------------------
# Loading Data
#----------------------

# Loading transition and noise combinations
combinations = np.array(
    pd.read_csv(
        os.path.join(
            load_dat,
            "support_file-combinations_all_possible.csv"
            )
        )
    )


# Loading BOLD data
bold_v = []
for ii in range(15000):
  bb = np.load(os.path.join(load_dat,f"sim_subject-{ii+1}_BOLD_full.npy"))
  bold_v.append(bb)


# Loading GMMHMM posterior values
mean_mat_l = []
posterior_probs_l = []
alpha_1_l = []

for ii in range(15000):
  # posterior_probs_l
  pp = np.load(os.path.join(load_mod, f"sim_subj_model-{ii+1}_posterior_probs.npy"))
  posterior_probs_l.append(pp)
  # mean_mat_l
  mml = np.load(os.path.join(load_mod, f"sim_subj_model-{ii+1}_state_mean_mat.npy"))
  mean_mat_l.append(mml)
  # alpha_1_l
  a1 = np.load(os.path.join(load_mod, f"sim_subj_model-{ii+1}_alpha_1.npy"))
  alpha_1_l.append(a1)



#-----------------------
# Calculating Thresholds
#-----------------------

# rho thresholds-----------------------------------------------------------------------
# rho values___________________________________________
rho_all = []
rho_sub = []
for ii in range(len(posterior_probs_l)):
  rr = mc.compute_rho(posterior_probs_l[ii])
  rho_all.extend(rr)
  rho_sub.append(rr)

## making df for descriptives
rr_all_df = pd.Series(rho_all)
rr_all_df.describe()

# Finding threshold_______________________________________________
tau_rho, info_rho = mt.gmm2_threshold(rho_all)

# Saving rho thresholds and details_______________________________
## Making thresholds df  with 15 decimal precision
rho_threshold = pd.DataFrame(
  {
    "tau":(f"{tau_rho:.15f}")
  }, 
  index = ["Gaussian Split rho: "]
  )

rho_details = pd.DataFrame(
  info_rho, 
  index = ["Low Gaussian", "High Gaussian"]
  )

## Saving
rho_threshold.to_csv(os.path.join(base_out,"rho_threshold.tsv"),sep="\t",index=True)
rho_details.to_csv(os.path.join(base_out,"rho_threshold_details.tsv"),sep="\t",index=True)


# Plotting and saving rho distribution and threshold_____________
sns.kdeplot(rho_all, fill=True)
plt.axvline(tau_rho, linestyle = "--", color="red", label=(r"$\tau$" f" ~ {round(tau_rho,5)}"), linewidth = 2.5)
plt.grid(axis='y', linewidth = 1, alpha = 0.3)
plt.legend(title = r"$\rho$ Threshold", loc="upper center")
## Saving figure
plt.savefig(os.path.join(fig_out,"rho_native_threshold_density.tiff"), dpi=400)
## Displaying
plt.show(), plt.close()

# Descriptives of rho values______________________________________
# By transition probability descriptives
## Calculating descriptives and cutoffs
rho_descriptive = []
for ii in range(len(rho_sub)):
  rho = rho_sub[ii]
  df = (round(pd.DataFrame(rho, columns = [f"sub-{[ii]}"]).describe(),3)).iloc[np.r_[1,2,3,7],:].T
  df["n_strong"] = len(np.where(rho >= tau_rho)[0])
  df["n_weak"] = len(np.where(rho < tau_rho)[0])
  rho_descriptive.append(df)

## Making df for descriptives
rho_df = pd.concat(rho_descriptive)
rho_df.describe()

## Saving rho and threshold #'s
rho_df.to_csv(os.path.join(base_out,"rho_threshold_subject_df.tsv"),sep="\t",index=True)


## Saving overall rho means
rho_df.mean(axis=0)
  # Saving overall rho means
rho_df.mean(axis=0).to_csv(os.path.join(base_out,"rho_threshold_subject_means_all.tsv"),sep="\t",index=True)

# Descriptives by transition probability______________________________________
# Parameters
t_ind = [0,4,8]
ind = 0
per_comb_subj = 1250
n_per_trans = (per_comb_subj*4)

# Descriptive report and saving by transition probability
for ii in np.arange(0, 15000, n_per_trans):
  # saving by transition probability rho means
  rho_df.iloc[
      ii:ii+n_per_trans,:].mean(
          axis=0
          ).to_csv(
              os.path.join(
                  base_out,
                  f"rho_threshold_subject_means_transition_prob_{combinations[t_ind[ind]][0]}.tsv"),
                  sep="\t",
                  index=True
            )
  # Reporting by transition probability rho means
  print(f"Transition Probability {combinations[t_ind[ind]][0]} \n", rho_df.iloc[ii:ii+n_per_trans,:].mean(axis=0))
  print("\n")
  ind += 1


# Pt thresholds-----------------------------------------------------------------------
# Pt value ________________________________________________________
pt_all = []
pt_sub = []
for ii in range(len(posterior_probs_l)):
  pt = mc.compute_pt(posterior_probs_l[ii], alpha_1_l[ii])
  pt_all.extend(pt)
  pt_sub.append(pt)

## making df for descriptives
pt_all_df = pd.Series(pt_all)
pt_all_df.describe()

## two gaussian threshold
_, info_pt = gmm2_threshold(pt_all)
  # investigation shows a poor cutoff due to two eaual distributions on two sides

## threshold by weighting the lower bound
    # Because P_t is equally weigted across two gaussians - we need to weights the lower bound
tau_pt = mt.tau_low_quantile(pt_all)

## Saving thresholds with 15 decimal precision
    # making df
pt_threshold = pd.DataFrame(
  {
    "tau":(f"{tau_pt:.15f}")
  }, 
  index = ["Gaussian Split pt: "]
  )

pt_details = pd.DataFrame(
  info_pt, 
  index = ["Low Gaussian", "High Gaussian"]
  )

## Saving
pt_threshold.to_csv(os.path.join(base_out,"pt_threshold.tsv"),sep="\t",index=True)
pt_details.to_csv(os.path.join(base_out,"pt_threshold_details.tsv"),sep="\t",index=True)

## plotting distribution and threshold
sns.kdeplot(pt_all, fill=True)
plt.axvline(tau_pt, linestyle = "--", color="red", label=(r"$\tau$" f" ~ {round(tau_pt,5)}"), linewidth = 2.5)
plt.grid(axis='y', linewidth = 1, alpha = 0.3)
plt.legend(title = r"$P_{t}$ Threshold", loc="upper center")
## saving figure
plt.savefig(os.path.join(fig_out,"pt_native_threshold_density.tiff"), dpi=400)
## displaying
plt.show() , plt.close()


# Descriptives of Pt values______________________________________
# By transition probability descriptives
## calculating descriptives and cutoffs
pt_descriptive = []
for ii in range(len(pt_sub)):
  pt = pt_sub[ii]
  df = (round(pd.DataFrame(pt, columns = [f"sub-{[ii]}"]).describe(),3)).iloc[np.r_[1,2,3,7],:].T
  df["n_strong"] = len(np.where(pt >= tau_pt)[0])
  df["n_weak"] = len(np.where(pt < tau_pt)[0])
  pt_descriptive.append(df)

## Making df for descriptives
pt_df = pd.concat(pt_descriptive)
pt_df.describe()

## Saving pt and threshold #'s
pt_df.to_csv(os.path.join(base_out,"pt_threshold_subject_df.tsv"),sep="\t",index=True)

## Saving overall pt means
pt_df.mean(axis=0)
  # Saving overall pt means
pt_df.mean(axis=0).to_csv(os.path.join(base_out,"pt_threshold_subject_means_all.tsv"),sep="\t",index=True)


# Descriptives by transition probability______________________________________
# Parameters
t_ind = [0,4,8]
ind = 0
per_comb_subj = 1250
n_per_trans = (per_comb_subj*4)

# Descriptive report and saving by transition probability
for ii in np.arange(0, 15000, n_per_trans):
  # saving by transition probability pt means
  pt_df.iloc[ii:ii+n_per_trans,:].mean(axis=0).to_csv(
      os.path.join(
          base_out,
          f"pt_threshold_subject_means_transition_prob_{combinations[t_ind[ind]][0]}.tsv"),
          sep="\t",
          index=True
          )
  print(f"Transition Probability {combinations[t_ind[ind]][0]} \n", pt_df.iloc[ii:ii+n_per_trans,:].mean(axis=0))
  print("\n")
  ind += 1


# D^maha Thresholds-----------------------------------------------------------------------
# Dmaha values ________________________________________________________
dmaha_all = []
dmaha_sub = []
for ii in range(len(posterior_probs_l)):
  dmaha = mc.compute_D_maha_fir_mahalanobis(posterior_probs_l[ii], mean_mat_l[ii], bold_v[ii])
  dmaha_all.extend(dmaha)
  dmaha_sub.append(dmaha)

## making df for descriptives
dmaha_all_df = pd.Series(dmaha_all)
dmaha_all_df.describe()

## finding threshold
tau_dmaha, info_dmaha = mt.Dmaha_high_threshold(dmaha_all,p_star = 0.9)

## Saving thresholdmaha with 15 decimal precision
  # making df
dmaha_threshold = pd.DataFrame(
  {
    "tau":(f"{tau_dmaha:.15f}")
  }, 
  index = ["Gaussian Split Dmaha: "]
  )

dmaha_details = pd.DataFrame(
  info_dmaha, 
  index = ["Low Gaussian -0 ", "Med Gaussian -1", "High Gaussian-2"]
  )

dmaha_details = dmaha_details.loc[:,['means', 'stds', 'weights', 'sep12']]

  # saving
dmaha_threshold.to_csv(os.path.join(base_out,"dmaha_threshold.tsv"),sep="\t",index=True)
dmaha_details.to_csv(os.path.join(base_out,"dmaha_threshold_details.tsv"),sep="\t",index=True)

## plotting distribution and threshold
sns.kdeplot(dmaha_all, fill=True)
plt.axvline(tau_dmaha, linestyle = "--", color="red", label=(r"$\tau$" f" ~ {round(tau_dmaha,5)}"), linewidth = 2.5)
plt.grid(axis='y', linewidth = 1, alpha = 0.3)
plt.legend(title = r"$D^{maha}$ Threshold", loc="upper center")
## Saving figure
plt.savefig(os.path.join(fig_out,"dmaha_native_threshold_density.tiff"), dpi=400)
## Displaying
plt.show() , plt.close()


  # descriptives by transition probability 
dmaha_descriptive = []
for ii in range(len(dmaha_sub)):
  dmaha = dmaha_sub[ii]
  df = (round(pd.DataFrame(dmaha, columns = [f"sub-{[ii]}"]).describe(),3)).iloc[np.r_[1,2,3,7],:].T
  df["n_flag"] = len(np.where(dmaha > tau_dmaha)[0])
  df["n_good"] = len(np.where(dmaha <= tau_dmaha)[0])
  dmaha_descriptive.append(df)

## Making df for descriptives
dmaha_df = pd.concat(dmaha_descriptive)
dmaha_df.describe()

  # saving dmaha and threshold #'s
dmaha_df.to_csv(os.path.join(base_out,"dmaha_threshold_subject_df.tsv"),sep="\t",index=True)

dmaha_df.mean(axis=0)
  # Saving overall dmaha means
dmaha_df.mean(axis=0).to_csv(os.path.join(base_out,"dmaha_threshold_subject_means_all.tsv"),sep="\t",index=True)

# Descriptives by transition probability______________________________________
# Parameters
t_ind = [0,4,8]
ind = 0
per_comb_subj = 1250
n_per_trans = (per_comb_subj*4)

# Descriptive report and saving by transition probability
for ii in np.arange(0, 15000, n_per_trans):
  # saving by transition probability dmaha means
  dmaha_df.iloc[ii:ii+n_per_trans,:].mean(axis=0).to_csv(
      os.path.join(
          base_out,
          f"dmaha_threshold_subject_means_transition_prob_{combinations[t_ind[ind]][0]}.tsv"),
          sep="\t",
          index=True
          )
  print(f"Transition Probability {combinations[t_ind[ind]][0]} \n", dmaha_df.iloc[ii:ii+n_per_trans,:].mean(axis=0))
  print("\n")
  ind += 1



# B_node -------------------------------------------------------------------------------
# B_node values ________________________________________________________
bs_all = []
bs_sub = []
for ii in range(len(posterior_probs_l)):
  b = mc.compute_B_node(posterior_probs_l[ii], mean_mat_l[ii], bold_v[ii])[0]
  bs_all.extend(b)
  bs_sub.append(b)

# Sensitivity testing of B_node FDR thresholds__________________________
# Averaging B_node values across subjects
ave_b = []
for ii in range(len(bs_sub)):
  ave_b.append(bs_sub[ii][0:660,:])

alphas_fdr = [0.1,0.15,0.2,0.25,0.3]
fdr_summary = []
for aa in alphas_fdr:
  tau_s = 0
  k_s = 0
  fdr_s = 0
  for ii in range(len(bs_sub)):
      tau, k, fdr = mc.bnode_fdr(ave_b[ii],alpha=aa,return_mask=False)
      tau_s += tau
      k_s += k
      fdr_s += fdr
  tau = tau_s/len(bs_sub)
  k = k_s/len(bs_sub)
  fdr = fdr_s/len(bs_sub)
  fdr_summary.append([aa, np.mean(tau), np.median(tau), np.mean(k), np.median(k), np.mean(fdr), np.median(fdr)])

# Making df for sensitivity descriptives
fdr_summary = pd.DataFrame(
    fdr_summary, columns = [
        "alpha", 
        "mean_tau", 
        "median_tau", 
        "mean_k", 
        "median_k", 
        "mean_fdr", 
        "median_fdr"
        ]
    )
fdr_summary 
    #    alpha  mean_tau  median_tau     mean_k   median_k  mean_fdr  median_fdr
    # 0   0.10  0.963796    0.963858   2.268392   2.259333  0.024122    0.024089
    # 1   0.15  0.913771    0.913822   4.057458   4.043833  0.061372    0.061315
    # 2   0.20  0.841058    0.841191   6.790524   6.767633  0.117553    0.117501
    # 3   0.25  0.761990    0.762066  11.087870  11.058733  0.181440    0.181412
    # 4   0.30  0.686319    0.686384  17.962063  17.921733  0.246079    0.246044

  # Choosing alpha = 0.25 as a balance of sensitivity and precision

# Saving fdr sensitivity summary
fdr_summary.to_csv(os.path.join(base_out,"bnode_fdr_summary_all.tsv"), sep = "\t",index=False)



# D_sp thresholds-----------------------------------------------------------------------
# Dsp values ________________________________________________________
Dsp_all = []
Dsp_sub = []
for ii in range(len(posterior_probs_l)):
  b = mc.compute_Dsp(bs_sub[ii])
  Dsp_all.extend(b)
  Dsp_sub.append(b)

# Dsp_all_df = pd.Series(Dsp_all)
# Dsp_all_df.describe()

# Finding thresholds_______________________________________________
## GMM approach for Dsp thresholds
tau_Dsp_lo,tau_Dsp_hi,info_Dsp = mt.multi_gmm_threshold(Dsp_all)

tau_Dsp_lo
    # 0.5426835208729968
tau_Dsp_hi
    # 0.6122026721313432
info_Dsp
    # {
    #    'k': 3, 
    #    'means_z': [-0.40202309521616275, 0.1509409178225851, 0.7192777670140872], 
    #    'stds_z': [0.8202428672293529, 0.1540208781574447, 0.15511520976695842], 
    #    'weights': [0.07783422387499608, 0.33911553284624385, 0.5830502432787601], 
    #    'cuts_z': [0.17115064981406958, 0.45658053900650336], 
    #    'cuts_x': [0.5426835208729968, 0.6122026721313432], 
    #    'separation': [0.9370099442462569, 3.6768362981294422], 
    #    'ok_pairs': [False, True]}

  # While these check out - the lower threshold is low and a borderline separation that does not meet criteria
    # so we will leverage the kde approach for the lower threshold to determine if we can get a more robust threshold
    # the upper threshold is well separated and meets criteria
    # however, the kde approach for the upper threshold is not as good as the gmm
    # thus we have a mixed approach here
    # the gmm approach is more sensitive and the kde is more conservative
  # we will use the values from the gmm approach as starting places for the kde approach



## KDE approach for Dsp low threshold
tau_Dsp_lo_kde, _, Dsp_info_kde = mt.Dsp_thresholds_kde_windows_binned(
    Dsp_all,
    bins=4096,
    low_anchor=0.51,   # “the low point near 0.51” (first minimum ≥ 0.5)
    high_anchor=0.62,  # “just below the high window” (last minimum ≤ 0.62)
    tail="low"  # we only want the low tail for this kde search
    ) # type: ignore

print("τ_low:", tau_Dsp_lo_kde)

## Quality and stability checks of Dsp thresholds
  # Stability check of low threshold
summary, grid = mt.Dsp_tau_sensitivity_binned(
    Dsp_all,
    low_window=(0.5,0.6),
    low_anchor=0.50,
    tail="low",
)

stability_df = pd.DataFrame(summary)
stability_df
    #             low  baseline_bandwidth
    # mean   0.522413            0.004343
    # std    0.000185            0.004343
    # min    0.522217            0.004343
    # max    0.522705            0.004343
    # drift  0.000488            0.004343

  # we can see std is low and drift is low so this is a stable threshold


  # Quality check of both thresholds
qual = mt.Dsp_threshold_quality(Dsp_all, tau_Dsp_lo_kde, tau_Dsp_hi, bins=4096)
qual_df = pd.DataFrame(qual)
qual_df
    #                                  low      high
    # threshold                   0.522339  0.612203
    # mass_left                   0.182608  0.416861
    # mass_right                  0.817392  0.583139
    # sep_robust                  3.724509  3.149628
    # valley_density              1.526555  2.241106
    # peak_left                   3.605181  4.382884
    # peak_right                  4.626065  4.675530
    # prominence_min_over_valley  2.361645  1.955679
    # bandwidth                   0.004343  0.004343
    # bin_width                   0.000244  0.000244

# Given robust separation: We will use the kde low threshold and gmm high threshold
tau_Dsp_lo = tau_Dsp_lo_kde

  # Saving thresholds with 15 decimal precision
    # making df
Dsp_threshold = pd.DataFrame(
  {
    "tau_lo":(f"{tau_Dsp_lo:.15f}"),
    "tau_hi":(f"{tau_Dsp_hi:.15f}")
  }, 
  index = ["Gaussian Split Dsp: "]
  )


  # saving
Dsp_threshold.to_csv(os.path.join(base_out,"dsp_threshold.tsv"),sep="\t",index=True)
qual_df.to_csv(os.path.join(base_out,"dsp_threshold_details.tsv"),sep="\t",index=True)
stability_df.to_csv(os.path.join(base_out,"dsp_low-threshold_stability.tsv"),sep="\t",index=True)

  # plotting distribution and threshold
sns.kdeplot(Dsp_all, fill=True)
plt.axvline(tau_Dsp_hi, linestyle = "--", color="red", label=(r"$\tau$" f" ~ {round(tau_Dsp_hi,5)}"), linewidth = 2.5)
plt.axvline(tau_Dsp_lo, color="red", label=(r"$\tau$" f" ~ {round(tau_Dsp_lo,5)}"), linewidth = 2.5)
plt.grid(axis='y', linewidth = 1, alpha = 0.3)
plt.legend(title = r"$D^{SP}$ Threshold", loc="upper left")
  # saving figure
plt.savefig(os.path.join(fig_out,"dsp_native_threshold_density.tiff"), dpi=400)
  # displaying
plt.show(), plt.close()



  # by transition probability descriptives

# Descriptives of Dsp values______________________________________
# By transition probability descriptives
Dsp_descriptive = []
for ii in range(len(Dsp_sub)):
  Dsp = Dsp_sub[ii]
  df = (round(pd.DataFrame(Dsp, columns = [f"sub-{[ii]}"]).describe(),3)).iloc[np.r_[1,2,3,7],:].T
  df["serial"] = len(np.where(Dsp >= tau_Dsp_hi)[0])
  df["mixed"] = len(np.where((Dsp > tau_Dsp_lo) & (Dsp < tau_Dsp_hi))[0])
  df["parallel"] = len(np.where(Dsp <= tau_Dsp_lo)[0])
  Dsp_descriptive.append(df)

Dsp_df = pd.concat(Dsp_descriptive)
Dsp_df.describe()

  # saving Dsp and threshold #'s
Dsp_df.to_csv(os.path.join(base_out,"dsp_threshold_subject_df.tsv"),sep="\t",index=True)

Dsp_df.mean(axis=0)
  # Saving overall Dsp means
Dsp_df.mean(axis=0).to_csv(os.path.join(base_out,"dsp_threshold_subject_means_all.tsv"),sep="\t",index=True)



# Descriptives by transition probability______________________________________
# Parameters
t_ind = [0,4,8]
ind = 0
per_comb_subj = 1250
n_per_trans = (per_comb_subj*4)

# Descriptive report and saving by transition probability
for ii in np.arange(0, 15000, n_per_trans):
  # saving by transition probability Dsp means
  Dsp_df.iloc[ii:ii+n_per_trans,:].mean(axis=0).to_csv(
      os.path.join(
          base_out,f"dsp_threshold_subject_means_transition_prob_{combinations[t_ind[ind]][0]}.tsv")
          ,sep="\t",
          index=True
          )
  print(f"Transition Probability {combinations[t_ind[ind]][0]} \n", Dsp_df.iloc[ii:ii+n_per_trans,:].mean(axis=0))
  print("\n")
  ind += 1




# Flagging unstable trials-----------------------------------------------------------------------

trial_flags_sub = []
time_flags_sub = []
info_sub = []
for jj in [0.30, 0.5]:
    trial_l = []
    time_l = []
    info_l = []
    for ii in range(len(posterior_probs_l)):
        time_flags, trial_flags, info = mc.flag_trials(
            Dmaha_val = dmaha_sub[ii],
            rho_val = rho_sub[ii],
            pt_val = pt_sub[ii],
            tau_D = tau_dmaha,
            tau_rho = tau_rho,
            tau_Pt = tau_pt,
            fir_bins = 5,
            min_frac = jj
        )
        # appending
            # by subject
        time_l.append(len(time_flags))
        trial_l.append(len(trial_flags))
        info_l.append(info)
    time_flags_sub.append(len(time_l))
    trial_flags_sub.append(len(trial_l))
    info_sub.append(info_l)


# Making df for descriptives
flag_df = pd.DataFrame(info_sub[0])
flag_df.describe()  
        #                  T  fir_bins  min_frac     n_trials  n_trials_flagged  n_timepoints_flagged
        # count  15000.00000   15000.0   15000.0  15000.00000      15000.000000          15000.000000
        # mean     661.97600       5.0       0.3    132.39520          0.000200             22.575400
        # std        2.44455       0.0       0.0      0.48891          0.024495             17.903473
        # min      660.00000       5.0       0.3    132.00000          0.000000              0.000000
        # 25%      660.00000       5.0       0.3    132.00000          0.000000              5.000000
        # 50%      660.00000       5.0       0.3    132.00000          0.000000             21.500000
        # 75%      665.00000       5.0       0.3    133.00000          0.000000             35.000000
        # max      665.00000       5.0       0.3    133.00000          3.000000            186.000000

flag_df = pd.DataFrame(info_sub[1])
flag_df.describe()  
        #                  T  fir_bins  min_frac     n_trials  n_trials_flagged  n_timepoints_flagged
        # count  15000.00000   15000.0   15000.0  15000.00000      15000.000000          15000.000000
        # mean     661.97600       5.0       0.5    132.39520          0.000200             22.575400
        # std        2.44455       0.0       0.0      0.48891          0.024495             17.903473
        # min      660.00000       5.0       0.5    132.00000          0.000000              0.000000
        # 25%      660.00000       5.0       0.5    132.00000          0.000000              5.000000
        # 50%      660.00000       5.0       0.5    132.00000          0.000000             21.500000
        # 75%      665.00000       5.0       0.5    133.00000          0.000000             35.000000
        # max      665.00000       5.0       0.5    133.00000          3.000000            186.000000

## Saving flagging details
    # saving flagging details and #'s by subject
ind=0
for jj in ["0.30", "0.5"]:
    flag_df = pd.DataFrame(info_sub[ind])
    flag_df.to_csv(os.path.join(base_out,f"flagged_trials_subject_df_min-frac-{jj}.tsv"),sep="\t",index=True)
    flag_df.mean(axis=0).to_csv(os.path.join(base_out,f"flagged_trials_subject_means_all_min-frac-{jj}.tsv"),sep="\t",index=True)
    ind += 1


# Descriptives by transition probability______________________________________
# Parameters
t_ind = [0,4,8]
ind = 0
per_comb_subj = 1250
n_per_trans = (per_comb_subj*4)

# Descriptive report and saving by transition probability
for ii in np.arange(0, 15000, n_per_trans):
  # saving by transition probability flagged means
  flag_df.iloc[ii:ii+n_per_trans,:].mean(axis=0).to_csv(
      os.path.join(
          base_out,f"flagged_subject_means_transition_prob_{combinations[t_ind[ind]][0]}.tsv")
          ,sep="\t",
          index=True
          )
  print(f"Transition Probability {combinations[t_ind[ind]][0]} \n", flag_df.iloc[ii:ii+n_per_trans,:].mean(axis=0))
  print("\n")
  ind += 1








