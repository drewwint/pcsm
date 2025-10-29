#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for cross-validated selection of GMMHMM latent states and mixtures
Dr. Drew E. Winters
Created on 9/16/2025
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from hmmlearn import hmm

def run_gmmhmm_cv_fast(
    Y,
    lengths,
    state_range=range(2,6),
    n_mixtures=2,
    n_splits=5,
    n_starts=1,
    n_iter=60,
    tol=1e-2,
    covariance_type="diag",
    random_state=5745,
    n_jobs=-1,
    pre_dispatch="all",
    batch_size=1,
):
    """
    Subject-level k-fold CV for a GMM-HMM using lengths. For each K in state_range, fits on train subjects
    and scores held-out subjects, returning per-fold held-out LL/obs and per-K summaries.
    This function does not select K; use select_best_k_strict for selection logic.

    Returns
    -------
    results_df : pd.DataFrame with columns per K:
        'latent_states','mean_LL','SE_LL','delta_LL','perc_folds'
        where delta_LL and perc_folds compare K vs (K-1) foldwise
    details : dict per K with keys:
        'LL_per_obs_folds','mean_LL_per_obs','SE_LL_per_obs','fold_sizes'
    """
    Y=np.asarray(Y,dtype=np.float32)
    lengths=[int(x) for x in lengths]
    assert sum(lengths)==Y.shape[0],"sum(lengths) must equal Y.shape[0]"
    n_sub=len(lengths)
    idx=0
    slices=[]
    for L in lengths:
        slices.append((idx,idx+L))
        idx+=L
    def _concat_subjects_by_index(subj_indices):
        parts=[];lens=[]
        for i in subj_indices:
            s,e=slices[i]
            parts.append(Y[s:e])
            lens.append(lengths[i])
        Yout=np.ascontiguousarray(np.concatenate(parts,axis=0))
        return Yout,lens
    kf=KFold(n_splits=n_splits,shuffle=True,random_state=random_state)
    subj_indices=np.arange(n_sub)
    folds=[(train_idx,test_idx) for train_idx,test_idx in kf.split(subj_indices)]
    def fit_one_fold(k,train_subj,val_subj,seed_base):
        rng=np.random.default_rng(seed_base+k*1009)
        Ytr,ltr=_concat_subjects_by_index(train_subj)
        Yva,lva=_concat_subjects_by_index(val_subj)
        best_val_LL=-np.inf;best_model=None
        for s in range(n_starts):
            seed=int(rng.integers(0,2**31-1))^(s+1)
            model=hmm.GMMHMM(n_components=k,n_mix=n_mixtures,n_iter=n_iter,tol=tol,covariance_type=covariance_type,random_state=seed,verbose=False)
            try:
                model.fit(Ytr,lengths=ltr)
                val_LL=model.score(Yva,lengths=lva)
                if val_LL>best_val_LL:
                    best_val_LL=val_LL;best_model=model
            except Exception:
                continue
        if(best_model is None)or(not np.isfinite(best_val_LL)):
            return np.nan,sum(lva)
        ll_per_obs=best_val_LL/float(sum(lva))
        del best_model
        return ll_per_obs,sum(lva)
    details={}
    rows=[]
    prev_k=None
    prev_ll_folds=None
    rng_master=np.random.default_rng(random_state)
    for k in state_range:
        fold_results=Parallel(n_jobs=n_jobs,backend="loky",prefer="processes",pre_dispatch=pre_dispatch,batch_size=batch_size)(delayed(fit_one_fold)(k,train_subj,val_subj,seed_base=int(rng_master.integers(0,2**31-1)))for(train_subj,val_subj)in folds)
        ll_folds=np.full(len(fold_results),np.nan,dtype=np.float64)
        fold_sizes=np.zeros(len(fold_results),dtype=int)
        for i,(ll_i,n_i) in enumerate(fold_results):
            ll_folds[i]=ll_i;fold_sizes[i]=n_i
        valid=np.isfinite(ll_folds)
        mean_LL=np.nan if not valid.any() else float(np.nanmean(ll_folds[valid]))
        se_LL=np.nan
        if valid.sum()>1:
            se_LL=float(np.nanstd(ll_folds[valid],ddof=1)/np.sqrt(valid.sum()))
        delta=np.nan;wins=np.nan
        if prev_k is not None and prev_ll_folds is not None:
            a=ll_folds; b=prev_ll_folds
            vb=np.isfinite(a)&np.isfinite(b)
            if vb.any():
                delta=float(np.nanmean(a[vb])-np.nanmean(b[vb]))
                wins=float(np.mean(a[vb]>b[vb]))
        rows.append({"latent_states":k,"mean_LL":mean_LL,"SE_LL":se_LL,"delta_LL":delta,"perc_folds":wins})
        details[k]={"LL_per_obs_folds":ll_folds.tolist(),"mean_LL_per_obs":mean_LL,"SE_LL_per_obs":se_LL,"fold_sizes":fold_sizes.tolist()}
        prev_k=k
        prev_ll_folds=ll_folds
    results_df=pd.DataFrame(rows)
    return results_df,details


## selecting latent states
def select_best_k_strict(
    results_df,
    delta_thresh=2.0,
    wins_thresh=0.70,
    se_factor=2.0,
    rel_gain=0.10,
):
    """
    Apply K-selection rules exactly and returns annotated table and K*.
    Rules for K vs K-1: 
        - delta_LL > delta_thresh
        - delta_LL > se_factor*SE_LL
        - perc_folds >= wins_thresh
        - if previous delta>0 then delta_LL >= (1+rel_gain)*previous_delta.
    """
    df=results_df.copy()
    df=df.sort_values("latent_states").reset_index(drop=True)
    df.loc[:,"selected_model"]=False
    Kmin=int(df["latent_states"].min()) if not df.empty else None
    K_star=None
    for i,row in df.iterrows():
        k=int(row["latent_states"])
        if k==Kmin:
            continue
        d=row.get("delta_LL",np.nan); se=row.get("SE_LL",np.nan); wins=row.get("perc_folds",np.nan)
        ok_abs=np.isfinite(d) and np.isfinite(se) and (d>delta_thresh) and (d>se_factor*se)
        ok_wins=(wins is not None) and np.isfinite(wins) and (wins>=wins_thresh)
        ok_rel=True
        prev_row=df.iloc[i-1] if i-1>=0 else None
        if prev_row is not None and np.isfinite(prev_row.get("delta_LL",np.nan)) and prev_row.get("delta_LL",np.nan)>0:
            ok_rel=d>=(1.0+rel_gain)*prev_row["delta_LL"]
        if ok_abs and ok_wins and ok_rel:
            df.at[i,"selected_model"]=True
            K_star=k
            break
    return df,K_star

## selecting latent states and mixtures together
def select_states_and_mixtures_strict_gain(
    Y,
    lengths,
    state_range=range(2,6),
    mixture_range=(1,2,3),
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
):
    """
    Joint CV over mixtures with simple between-mixture rule. For each M, run CV over K and pick K* via select_best_k_strict.
    Then compare consecutive mixtures by mean_LL at K*: row i is 'selected' iff mean_LL[i]-mean_LL[i-1] >= gain_thresh_mixtures.
    Returns a per-mixture summary table with columns: mixtures, K_star, mean_LL, SE_LL, delta (consecutive gain), selected; plus a 'details' dict with per-K tables and a short criteria report for each mixture.
    """
    Y=np.asarray(Y,dtype=np.float32)
    lengths=[int(x) for x in lengths]
    assert sum(lengths)==Y.shape[0],"sum(lengths) must equal Y.shape[0]"
    rng=np.random.default_rng(random_state)
    rows=[]
    details={}
    for M in mixture_range:
        res,det=run_gmmhmm_cv_fast(Y=Y,lengths=lengths,state_range=state_range,n_mixtures=M,n_splits=n_splits,n_starts=n_starts_cv,n_iter=n_iter_cv,tol=tol_cv,covariance_type=covariance_type,random_state=int(rng.integers(0,2**31-1)),n_jobs=n_jobs,pre_dispatch=pre_dispatch,batch_size=batch_size)
        res_sel,Ks=select_best_k_strict(results_df=res,delta_thresh=delta_thresh,wins_thresh=wins_thresh,se_factor=se_factor,rel_gain=rel_gain_states)
        if Ks is None:
            rows.append({"mixtures":M,"K_star":np.nan,"mean_LL":np.nan,"SE_LL":np.nan,"delta":np.nan,"selected":False})
            details[M]={"cv_table":res_sel,"criteria":{"gain_vs_prev":np.nan,"gain_required":gain_thresh_mixtures,"pass_gain":False}}
            continue
        star_row=res_sel.loc[res_sel["latent_states"]==Ks].iloc[0]
        rows.append({"mixtures":M,"K_star":int(Ks),"mean_LL":float(star_row["mean_LL"]) if np.isfinite(star_row["mean_LL"]) else np.nan,"SE_LL":float(star_row["SE_LL"]) if np.isfinite(star_row["SE_LL"]) else np.nan,"delta":np.nan,"selected":False})
        details[M]={"cv_table":res_sel,"criteria":None}
    summary_df=pd.DataFrame(rows).sort_values("mixtures").reset_index(drop=True)
    n=summary_df.shape[0]
    summary_df.loc[:,"delta"]=np.nan
    summary_df.loc[:,"selected"]=False
    pass_flags=[False]*n
    for i in range(n):
        if i==0:
            pass_flags[i]=True
            M=int(summary_df.loc[i,"mixtures"])
            details[M]["criteria"]={
                "gain_vs_prev":np.nan,
                "gain_required":float(gain_thresh_mixtures),
                "se_candidate":float(summary_df.loc[i,"SE_LL"]) if np.isfinite(
                    summary_df.loc[i,"SE_LL"]) else None,
                "rule":"gain >= max(gain_thresh_mixtures, 2*SE_candidate)",
                "pass_gain":True,
                "note":"baseline",
            }
        else:
            prev_mean=summary_df.loc[i-1,"mean_LL"]
            curr_mean=summary_df.loc[i,"mean_LL"]
            cand_se=summary_df.loc[i,"SE_LL"]
            if np.isfinite(curr_mean) and np.isfinite(prev_mean):
                d=float(curr_mean-prev_mean)
                summary_df.loc[i,"delta"]=d
                se_term=2.0*float(cand_se) if np.isfinite(cand_se) else 0.0
                req=float(max(gain_thresh_mixtures, se_term))
                pass_flags[i]=bool(d>=req)
            else:
                d=np.nan
                req=float(gain_thresh_mixtures)
                pass_flags[i]=False
            M=int(summary_df.loc[i,"mixtures"])
            details[M]["criteria"]={
                "gain_vs_prev":None if not np.isfinite(d) else float(d),
                "gain_required":req,
                "se_candidate":None if not np.isfinite(cand_se) else float(cand_se),
                "rule":"gain >= max(gain_thresh_mixtures, 2*SE_candidate)",
                "pass_gain":pass_flags[i],
            }
    selected_idx=0
    for i in range(1,n):
        if pass_flags[i]:
            selected_idx=i
        else:
            break
    summary_df.loc[selected_idx,"selected"]=True
    return summary_df,details

