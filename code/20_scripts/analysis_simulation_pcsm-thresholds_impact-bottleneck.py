#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Identifying load/bottleneck Cutpoints for PCSM
Dr. Drew E. Winters
Created on 9/16/2025

Note this was done after scaling factors introduced into calculations (generalizable/comparable)
"""

#------------------
# Setup/loading packages 
# -----------------

# packages -----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats
from joblib import Parallel, delayed # for parallel processing
import projlib # local package for project specific functions
from projlib import metric_calculation as mc # loading the metric calculation functions specifically
from projlib import metric_thresholding as mt

#----------------------------------
# Setting up paths and loading data
#-----------------------------------

# Paths-------------------------------------------------------------------
load_dat = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/simulated_participants"
load_mod = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/derivatives/models"
base_out = r"/home/dwinters/ascend-lab/data/pcsm/simulated_data/derivatives/thresholds"
fig_out = r"/home/dwinters/ascend-lab/publications/pcsm/code/90_figures/threshold_figs"


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


# Loading transition and noise combinations---------------------------------
combinations = np.array(
    pd.read_csv(
        os.path.join(
            load_dat,
            "support_file-combinations_all_possible.csv"
            )
        )
    )




#--------------------------------------
# Aquring Values Across All Simulations
#--------------------------------------

# Loop to extract values -------------------------------------------------------
## paralellelized to improve computing efficiency
bottleneck_idx = []
imp_cumulative = []
imp_duration = []
resource_l = []
demand_l = []

def _process_one(posterior_probs, means, bold):
    bnode, _ = mc.compute_B_node(posterior_probs, means, bold)
    dsp = mc.compute_Dsp(bnode)
    demand, resource_level, impacts,_,_,_, components = mc.compute_transition_load(posterior_probs, dsp)
    bnidx = mc.compute_serial_bottleneck(components['labels']['mode'], demand, verbose=False)
    return (bnidx, impacts["impact_cumulative"], impacts["impact_duration"], demand, resource_level)


results = Parallel(n_jobs=-1, backend="loky")(  # "loky" if BLAS is the bottleneck or "threading" if not
    delayed(_process_one)(posterior_probs_l[ii], mean_mat_l[ii], bold_v[ii])
    for ii in range(len(posterior_probs_l))
)


## mapping to repective lists
bottleneck_idx, imp_cumulative, imp_duration, demand_l, resource_l = map(list, zip(*results))

## test to ensure expected sizes
len(bottleneck_idx) == 15000 #True
len(imp_cumulative) == 15000 #True
len(imp_duration) == 15000 #True
len(demand_l) == 15000 #True
len(resource_l) == 15000 #True


#-----------------------------------------
# Identifying thresholds
#----------------------------------------

# Impact Cumulative---------------------------------------------------------------
impact_cum_tau, impact_cum_info = mt.gmm2_threshold(imp_cumulative)
# >>> impact_cum_tau
    #  0.6489781181179637
# >>> impact_cum_info
    # {'means': [-0.4346497300782442, 4.936564314147948], 
    # 'stds': [0.5981208259395492, 1.3581120568834613], 
    # 'weights': [0.9256809323138616, 0.07431906768613832], 
    # 'separation': 5.118673323892103})

## Saving thresholds with 15 decimal precision
    # making df
imp_cum_threshold = pd.DataFrame(
  {
    "impact_cumulative_threshold_tau":(f"{impact_cum_tau:.15f}")
  }, 
  index = ["Gaussian Split impact cumulative: "]
  )

imp_cum_details = pd.DataFrame(
  impact_cum_info, 
  index = ["Low Gaussian", "High Gaussian"]
  )

## Saving
imp_cum_threshold.to_csv(os.path.join(base_out,"impact-cumulative_threshold.csv"),index=True)
imp_cum_details.to_csv(os.path.join(base_out,"impact-cumulative_threshold_details.csv"),index=True)

## plotting distribution and threshold
sns.kdeplot(imp_cumulative, fill=True)
plt.axvline(x=impact_cum_tau, color="r", linestyle="--", label=("Strong \n" r"$\tau$" f" ~ {round(impact_cum_tau,5)}"))
plt.xlabel("Impact Magnitude")
plt.legend(loc="best", title= "Impact Cumulative")
plt.savefig(os.path.join(fig_out,"impact-cumulative_threshold_density.tiff"), dpi=400)
plt.show();plt.close()



pd.Series(imp_cumulative).describe()

np.sum((np.array(imp_cumulative)>impact_cum_tau)==True)


## Impact Duration ---------------------------------------------------------------------
_, impact_dur_info = mt.gmm2_threshold(imp_duration)
impact_dur_tau = mt.tau_low_quantile(imp_duration)

# >>> impact_dur_tau
    # 13.113130658730196
# >>> impact_dur_info
    # {'means': [2.739796382017775, 126.79725756811118], 
    # 'stds': [6.306539443231846, 0.47584661461080674], 
    # 'weights': [0.9368, 0.0632], 
    # 'separation': 27.740483556714292}

## Saving thresholds with 15 decimal precision
    # making df
imp_dur_threshold = pd.DataFrame(
  {
    "impact_duration_threshold_tau":(f"{impact_dur_tau:.15f}")
  }, 
  index = ["Gaussian Split impact duration: "]
  )

imp_dur_details = pd.DataFrame(
  impact_dur_info, 
  index = ["Low Gaussian", "High Gaussian"]
  )

## Saving
imp_dur_threshold.to_csv(os.path.join(base_out,"impact-duration_threshold.csv"),index=True)
imp_dur_details.to_csv(os.path.join(base_out,"impact-duration_threshold_details.csv"),index=True)

## plotting distribution and threshold
sns.kdeplot(imp_duration, fill=True)
plt.axvline(x=impact_dur_tau, color="r", linestyle="--", label=("Long \n" r"$\tau$" f" ~ {round(impact_dur_tau,5)}"))
plt.xlabel("Duration Magnitude")
plt.legend(loc="best", title= "Impact Duration")
plt.savefig(os.path.join(fig_out,"impact-duration_threshold_density.tiff"), dpi=400)
plt.show();plt.close()


pd.Series(imp_duration).describe()


# Demand -------------------------------------------------------------------------------
# For demand we identify when demands are high and low
# Given  this is a unimodal distribution we define percentiles for thresholding

dmnd_qdist, dmnd_th, dmnd_thci = mt.perc_threshold(demand_l)

demm_tau_high = dmnd_th["tau_high"]
demm_tau_low = dmnd_th["tau_low"]

# Saving thresholds and details_______________________________
## Making thresholds df  with 15 decimal precision
demand_threshold = pd.DataFrame(
  {
    "demand_tau_low":[(f"{demm_tau_low:.15f}"),dmnd_thci["tau_low_ci95"]],
    "demand_tau_high":[(f"{demm_tau_high:.15f}"), dmnd_thci["tau_high_ci95"]]
  }, 
  index = ["Demand Percentile Tau : ", "Low/High CI: "]
  )

## calculating separation index:
##- finding critical t
crit_t = scipy.stats.t.ppf((1-0.05/2),(15000-1))
##- defining sd
sd_low = ((np.diff(dmnd_thci["tau_low_ci95"])/2)* np.sqrt(15000))/crit_t
sd_high = ((np.diff(dmnd_thci["tau_high_ci95"])/2)* np.sqrt(15000))/crit_t
##- calculating sepration d
demand_d = np.diff([demm_tau_low,demm_tau_high])[0] / (np.sqrt((sd_low[0]**2+sd_high[0]**2)/2))
demand_d
  # np.float64(17.40977567129103)


## Saving
demand_threshold.to_csv(os.path.join(base_out,"demand_threshold.tsv"), sep="\t",index=True)


# Building most likely distribution for plotting
## defining sd
demm_aprox_std = (demm_tau_high - demm_tau_low)/1.349
## building distribution
demand_gen_data = np.random.normal(loc=np.median(dmnd_qdist), scale=demm_aprox_std, size=15000)

## plotting distribution and threshold
sns.kdeplot(demand_gen_data, fill=True)
plt.axvline(x=demm_tau_low, color="r", linestyle="--", label=("Low \n" r"$\tau$" f" ~ {round(demm_tau_low,5)}"))
plt.axvline(x=demm_tau_high, color="r", label=("High \n" r"$\tau$" f" ~ {round(demm_tau_high,5)}"))
plt.legend(loc="best", title= "Demand")
plt.savefig(os.path.join(fig_out,"demand_threshold_density.tiff"), dpi=400)
plt.show();plt.close()

# Descriptives by transition probability______________________________________
  # descriptives by transition probability 
dmnd_descriptive = []
for ii in range(len(demand_l)):
  dmnd = demand_l[ii]
  df = (round(pd.DataFrame(dmnd, columns = [f"sub-{[ii]}"]).describe(),3)).iloc[np.r_[1,2,3,7],:].T
  df["n_high"] = len(np.where(dmnd > demm_tau_high)[0])
  df["n_low"] = len(np.where(dmnd < demm_tau_low)[0])
  dmnd_descriptive.append(df)

## Making df for descriptives
dmnd_df = pd.concat(dmnd_descriptive)

dmnd_df.to_csv(os.path.join(base_out,"demand_threshold_subject_df.tsv"),sep="\t",index=True)
mean_demand = dmnd_df.mean(axis=0)
mean_demand.to_csv(os.path.join(base_out,"demand_threshold_subject_means_all.tsv"),sep="\t",index=True)


# Parameters
t_ind = [0,4,8]
ind = 0
per_comb_subj = 1250
n_per_trans = (per_comb_subj*4)

# Descriptive report and saving by transition probability
for ii in np.arange(0, 15000, n_per_trans):
  # saving by transition probability demand means
  dmnd_df.iloc[ii:ii+n_per_trans,:].mean(axis=0).to_csv(
      os.path.join(
          base_out,
          f"demand_threshold_subject_means_transition_prob_{combinations[t_ind[ind]][0]}.tsv"),
          sep="\t",
          index=True
          )
  print(f"Transition Probability {combinations[t_ind[ind]][0]} \n", dmnd_df.iloc[ii:ii+n_per_trans,:].mean(axis=0))
  print("\n")
  ind += 1



# Resources -------------------------------------------------------------------------------
# For resources we identify when resources are high and low
# Given  this is a unimodal distribution we define percentiles for thresholding

resor_qdist, resor_th, resor_thci = mt.perc_threshold(resource_l)

resm_tau_high = resor_th["tau_high"]
resm_tau_low = resor_th["tau_low"]

# Saving thresholds and details_______________________________
## Making thresholds df  with 15 decimal precision
resource_threshold = pd.DataFrame(
  {
    "resource_tau_low":[(f"{resm_tau_low:.15f}"),resor_thci["tau_low_ci95"]],
    "resource_tau_high":[(f"{resm_tau_high:.15f}"), resor_thci["tau_high_ci95"]]
  }, 
  index = ["Resources Percentile Tau : ", "Low/High CI: "]
  )

## calculating separation index:
##- finding critical t
crit_t = scipy.stats.t.ppf((1-0.05/2),(15000-1))
##- defining sd
sd_low = ((np.diff(resor_thci["tau_low_ci95"])/2)* np.sqrt(15000))/crit_t
sd_high = ((np.diff(resor_thci["tau_high_ci95"])/2)* np.sqrt(15000))/crit_t
##- calculating sepration d
resource_d = np.diff([resm_tau_low,resm_tau_high])[0] / (np.sqrt((sd_low[0]**2+sd_high[0]**2)/2))
resource_d
  # np.float64(7.381059857164519)


## Saving
resource_threshold.to_csv(os.path.join(base_out,"resource_threshold.tsv"), sep="\t",index=True)

# Building most likely distribution for plotting
## defining sd
resm_aprox_std = (resm_tau_high - resm_tau_low)/1.349
## building distribution
resource_gen_data = np.random.normal(loc=np.median(resor_qdist), scale=resm_aprox_std, size=15000)

## plotting distribution and threshold
sns.kdeplot(resource_gen_data, fill=True)
plt.axvline(x=resm_tau_low, color="r", linestyle="--", label=("Low \n" r"$\tau$" f" ~ {round(resm_tau_low,5)}"))
plt.axvline(x=resm_tau_high, color="r", label=("High \n" r"$\tau$" f" ~ {round(resm_tau_high,5)}"))
plt.legend(loc="best", title= "Resources")
plt.savefig(os.path.join(fig_out,"resource_threshold_density.tiff"), dpi=400)
plt.show();plt.close()

# Descriptives by transition probability______________________________________
  # descriptives by transition probability 
resor_descriptive = []
for ii in range(len(resource_l)):
  resor = resource_l[ii]
  df = (round(pd.DataFrame(resor, columns = [f"sub-{[ii]}"]).describe(),3)).iloc[np.r_[1,2,3,7],:].T
  df["n_high"] = len(np.where(resor > resm_tau_high)[0])
  df["n_low"] = len(np.where(resor < resm_tau_low)[0])
  resor_descriptive.append(df)

## Making df for descriptives
resor_df = pd.concat(resor_descriptive)

resor_df.to_csv(os.path.join(base_out,"resource_threshold_subject_df.tsv"),sep="\t",index=True)
mean_resource = resor_df.mean(axis=0)
mean_resource.to_csv(os.path.join(base_out,"resource_threshold_subject_means_all.tsv"),sep="\t",index=True)

# Parameters
t_ind = [0,4,8]
ind = 0
per_comb_subj = 1250
n_per_trans = (per_comb_subj*4)

# Descriptive report and saving by transition probability
for ii in np.arange(0, 15000, n_per_trans):
  # saving by transition probability resource means
  resor_df.iloc[ii:ii+n_per_trans,:].mean(axis=0).to_csv(
      os.path.join(
          base_out,
          f"resource_threshold_subject_means_transition_prob_{combinations[t_ind[ind]][0]}.tsv"),
          sep="\t",
          index=True
          )
  print(f"Transition Probability {combinations[t_ind[ind]][0]} \n", resor_df.iloc[ii:ii+n_per_trans,:].mean(axis=0))
  print("\n")
  ind += 1



# Bottleneck---------------------------------------------------------------------
# For bottleneck thresholding we only identify points for a long and extremely long bottleneck
## given this is a tri distribution we do by finding kde valleys


pd.Series(bottleneck_idx).describe()
    # count    15000.000000
    # mean        -0.232427
    # std          0.493544
    # min         -0.587926
    # 25%         -0.581206
    # 50%         -0.523217
    # 75%          0.000000
    # max          1.000000
    # dtype: float64
  ##- we can see here that the mean is >50% indicating a low/normative distribution
  ##- the 50% is much lower
  ##- the 75% is around 0 suggesting we should see a cutpoint about there indicating high
  ##- the max is much higher in magnitide than the < 0 distribution - so we have 3 meaninful distributions.

## While we clearly see three distributions indicating normal, high, very high
### the very high only makes up a small weight of the distribuition - which is expected
#### therefore we set the min_weight parameter for this function to 5% inorder to capture this very high cut. 
#### Given that the separation is well above apiroi criteria - this appears justified and true to the data. 
bottleneck_tau_high,bottleneck_tau_very_high, btn_tau_info = mt.multi_gmm_threshold(bottleneck_idx,min_weight=0.05)

bottleneck_tau_high
  # 1.0042364280061544e-06
bottleneck_tau_very_high
  # 0.9318624587258728
btn_tau_info
  # {'k': 3, 
  # 'means_z': [-13.815509557959176, 0.754018163934115, 8.955694916998375], 
  # 'stds_z': [0.001, 2.3861733919990207, 3.5890426963931628], 
  # 'weights': [0.8286666666663732, 0.11994883595271387, 0.051384497380913045], 
  # 'cuts_z': [-13.811282074118388, 2.6156568991829086], 
  # 'cuts_x': [1.0042364280061544e-06, 0.9318624587258728], 
  # 'separation': [8.634920807793586, 2.6912411616270755], 
  # 'ok_pairs': [True, True]}

  ##- about 17% of the distribution density weights is above the high cutpoint with only about 5% at the very high
  ##- thre were no meaninful separations form normative to low so there is only normative, high, very high

# Saving thresholds and details_______________________________
## Making thresholds df  with 15 decimal precision
bottleneck_threshold = pd.DataFrame(
  {
    "bottleneck_tau_high":(f"{bottleneck_tau_high:.15f}"),
    "bottleneck_tau_very_high":(f"{bottleneck_tau_very_high:.15f}")
  }, 
  index = ["Bottleneck KDE Tau : "]
  )

bottleneck_info_df = pd.DataFrame({k: pd.Series(v) for k, v in btn_tau_info.items()})
bottleneck_info_df.cuts_z = bottleneck_info_df.cuts_z.values[[2,0,1]]
bottleneck_info_df.cuts_x = bottleneck_info_df.cuts_x.values[[2,0,1]]
bottleneck_info_df.separation = bottleneck_info_df.separation.values[[2,0,1]]
bottleneck_info_df.ok_pairs = bottleneck_info_df.ok_pairs.values[[2,0,1]]
bottleneck_info_df.index=["gaussian_low","gaussian_med","gaussian_high"]

## Saving
bottleneck_threshold.to_csv(os.path.join(base_out,"bottleneck_threshold.tsv"), sep="\t",index=True)
bottleneck_info_df.to_csv(os.path.join(base_out,"bottleneck_threshold_info.tsv"), sep="\t",index=True)


## plotting and saving
sns.kdeplot(bottleneck_idx, fill=True)
plt.axvline(x=bottleneck_tau_high, color="r", linestyle="--", label=("Long \n" r"$\tau$" f" ~ {round(bottleneck_tau_high,5)}"))
plt.axvline(x=bottleneck_tau_very_high, color="r", label=("Very Long \n" r"$\tau$" f" ~ {round(bottleneck_tau_very_high,5)}"))
plt.xlabel("Bottleneck Index")
plt.legend(loc="best", title= "Bottleneck")
plt.savefig(os.path.join(fig_out,"bottleneck_threshold_density.tiff"), dpi=400)
plt.show();plt.close()



# Descriptives by transition probability______________________________________
  # descriptives by transition probability 
bottleneck_descriptive_df = pd.Series(bottleneck_idx).describe().iloc[np.r_[1,2,3,7]]
bottleneck_descriptive_df["n_>high_all"]=len(np.where(bottleneck_idx>bottleneck_tau_high)[0])
bottleneck_descriptive_df["n_high_only"]=len(np.where((bottleneck_idx>bottleneck_tau_high)&(bottleneck_idx<bottleneck_tau_very_high))[0])
bottleneck_descriptive_df["n_very_high_only"]=len(np.where(bottleneck_idx>bottleneck_tau_very_high)[0])

bottleneck_descriptive_df.to_csv(os.path.join(base_out,"bottleneck_threshold_subject_means_all.tsv"),sep="\t",index=True)


# Parameters
t_ind = [0,4,8]
ind = 0
per_comb_subj = 1250
n_per_trans = (per_comb_subj*4)

# Descriptive report and saving by transition probability
for ii in np.arange(0, 15000, n_per_trans):
  ddfbn = pd.Series(bottleneck_idx).iloc[ii:ii+n_per_trans].describe().iloc[np.r_[1,2,3,7]]
  ddfbn["n_>high_all"]=len(np.where(pd.Series(bottleneck_idx).iloc[ii:ii+n_per_trans]>bottleneck_tau_high)[0])
  ddfbn["n_high_only"]=len(np.where((pd.Series(bottleneck_idx).iloc[ii:ii+n_per_trans]>bottleneck_tau_high)&(pd.Series(bottleneck_idx).iloc[ii:ii+n_per_trans]<bottleneck_tau_very_high))[0])
  ddfbn["n_very_high_only"]=len(np.where(pd.Series(bottleneck_idx).iloc[ii:ii+n_per_trans]>bottleneck_tau_very_high)[0])
  ddfbn["%_high_all"] = ddfbn["n_>high_all"]/n_per_trans
  # saving by transition probability resource means
  ddfbn.to_csv(
      os.path.join(
          base_out,
          f"bottleneck_threshold_subject_means_transition_prob_{combinations[t_ind[ind]][0]}.tsv"),
          sep="\t",
          index=True
          )
  print(f"Transition Probability {combinations[t_ind[ind]][0]} \n", ddfbn)
  print("\n")
  ind += 1





