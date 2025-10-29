#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for cross-validating parameters
Dr. Drew E. Winters
Created on 9/16/2025
"""

#-------------------
# Required Packages
#-------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from hmmlearn import hmm



#-----------
# Functions
# ----------



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
    Joint CV over mixtures with the same rules/outputs as your original trio:
      - Runs CV over K for each mixture M (was: run_gmmhmm_cv_fast)
      - Selects K* via strict rules (was: select_best_k_strict)
      - Compares mixtures by mean_LL at K* with the same gain rule
    Returns
    -------
    summary_df : pd.DataFrame with columns ['mixtures','K_star','mean_LL','SE_LL','delta','selected']
    details    : dict[M] -> {'cv_table': <per-K table with 'selected_model'>, 'criteria': {...}}
    """
    # small helpers (local)
    ## ensuring shape
    Y = np.asarray(Y, dtype=np.float32)
    lengths = [int(x) for x in lengths]
    assert sum(lengths) == Y.shape[0], "sum(lengths) must equal Y.shape[0]"
    ## getting lengths
    n_sub = len(lengths)
    slices = []
    idx = 0
    for L in lengths:
        slices.append((idx, idx + L))
        idx += L
    ## function for concatenating by subject index
    def _concat_subjects_by_index(subj_indices):
        parts, lens = [], []
        for i in subj_indices:
            s, e = slices[i]
            parts.append(Y[s:e])
            lens.append(lengths[i])
        return np.ascontiguousarray(np.concatenate(parts, axis=0)), lens
    ## function for cv folds
    def _folds(n_splits_, seed):
        kf = KFold(n_splits=n_splits_, shuffle=True, random_state=seed)
        subj_indices = np.arange(n_sub)
        return [(tr, va) for tr, va in kf.split(subj_indices)]
    ## function to speed up fold calculations
    def _fit_one_fold(k, M, train_subj, val_subj, base_seed):
        rng = np.random.default_rng(base_seed + k * 1009 + M * 7919)
        Ytr, ltr = _concat_subjects_by_index(train_subj)
        Yva, lva = _concat_subjects_by_index(val_subj)
        best_val_LL = -np.inf
        for s in range(n_starts_cv):
            seed = int(rng.integers(0, 2**31 - 1)) ^ (s + 1)
            model = hmm.GMMHMM(
                n_components=k,
                n_mix=M,
                n_iter=n_iter_cv,
                tol=tol_cv,
                covariance_type=covariance_type,
                random_state=seed,
                verbose=False,
            )
            try:
                model.fit(Ytr, lengths=ltr)
                val_LL = model.score(Yva, lengths=lva)
                if val_LL > best_val_LL:
                    best_val_LL = val_LL
            except Exception:
                continue
        if not np.isfinite(best_val_LL):
            return np.nan, sum(lva)
        return float(best_val_LL) / float(sum(lva)), sum(lva)
    ## function for cross-validation over states
    def _cv_over_states(state_range_, M, master_seed):
        folds = _folds(n_splits, master_seed)
        rows, prev_ll = [], None
        details_k = {}
        for k in state_range_:
            fold_results = Parallel(
                n_jobs=n_jobs, backend="loky", prefer="processes",
                pre_dispatch=pre_dispatch, batch_size=batch_size
            )(
                delayed(_fit_one_fold)(
                    k, M, train_idx, val_idx,
                    base_seed=int(np.random.default_rng(master_seed).integers(0, 2**31 - 1))
                )
                for (train_idx, val_idx) in folds
            )
            ll_folds = np.array([r[0] for r in fold_results], dtype=np.float64)
            fold_sizes = np.array([r[1] for r in fold_results], dtype=int)
            valid = np.isfinite(ll_folds)
            mean_LL = float(np.nanmean(ll_folds[valid])) if valid.any() else np.nan
            SE_LL = (
                float(np.nanstd(ll_folds[valid], ddof=1) / np.sqrt(valid.sum()))
                if valid.sum() > 1 else np.nan
            )
            #- Identifying when folds out perform subsequent folds
            delta = wins = np.nan
            if prev_ll is not None:
                vb = np.isfinite(ll_folds) & np.isfinite(prev_ll)
                if vb.any():
                    delta = float(np.nanmean(ll_folds[vb]) - np.nanmean(prev_ll[vb]))
                    wins = float(np.mean(ll_folds[vb] > prev_ll[vb]))
            #- building cross validation outputs
            rows.append({
                "latent_states": k,
                "mean_LL": mean_LL,
                "SE_LL": SE_LL,
                "delta_LL": delta,
                "perc_folds": wins,
            })
            details_k[k] = {
                "LL_per_obs_folds": ll_folds.tolist(),
                "mean_LL_per_obs": mean_LL,
                "SE_LL_per_obs": SE_LL,
                "fold_sizes": fold_sizes.tolist(),
            }
            prev_ll = ll_folds
        return pd.DataFrame(rows), details_k
    ## function for selecting the best number of states
    def _select_best_k_strict(df_in):
        df = df_in.sort_values("latent_states").reset_index(drop=True).copy()
        df.loc[:, "selected_model"] = False
        if df.empty:
            return df, None
        Kmin = int(df["latent_states"].min())
        K_star = None
        for i, row in df.iterrows():
            k = int(row["latent_states"])
            if k == Kmin:
                continue
            d = row.get("delta_LL", np.nan)
            se = row.get("SE_LL", np.nan)
            wins = row.get("perc_folds", np.nan)
            ok_abs = (np.isfinite(d) and np.isfinite(se) and (d > delta_thresh) and (d > se_factor * se))
            ok_wins = (wins is not None) and np.isfinite(wins) and (wins >= wins_thresh)
            ok_rel = True
            if i - 1 >= 0:
                prev_d = df.iloc[i - 1].get("delta_LL", np.nan)
                if np.isfinite(prev_d) and prev_d > 0:
                    ok_rel = d >= (1.0 + rel_gain_states) * prev_d
            if ok_abs and ok_wins and ok_rel:
                df.at[i, "selected_model"] = True
                K_star = k
                break
        return df, K_star
    # Computations
    ## cross validating
    rng = np.random.default_rng(random_state)
    summary_rows, details = [], {}
    for M in mixture_range:
        res_df, det_k = _cv_over_states(state_range, M, int(rng.integers(0, 2**31 - 1)))
        res_sel, K_star = _select_best_k_strict(res_df)
        if K_star is None:
            summary_rows.append({
                "mixtures": M, "K_star": np.nan, "mean_LL": np.nan,
                "SE_LL": np.nan, "delta": np.nan, "selected": False
            })
            details[M] = {
                "cv_table": res_sel,
                "criteria": {"gain_vs_prev": np.nan, "gain_required": gain_thresh_mixtures, "pass_gain": False},
            }
            continue
        # selecting metrics for latent states (k)
        star_row = res_sel.loc[res_sel["latent_states"] == K_star].iloc[0]
        summary_rows.append({
            "mixtures": M,
            "K_star": int(K_star),
            "mean_LL": float(star_row["mean_LL"]) if np.isfinite(star_row["mean_LL"]) else np.nan,
            "SE_LL": float(star_row["SE_LL"]) if np.isfinite(star_row["SE_LL"]) else np.nan,
            "delta": np.nan,
            "selected": False,
        })
        details[M] = {"cv_table": res_sel, "criteria": None}
    # Building summary
    summary_df = pd.DataFrame(summary_rows).sort_values("mixtures").reset_index(drop=True)
    n = summary_df.shape[0]
    summary_df.loc[:, "delta"] = np.nan
    summary_df.loc[:, "selected"] = False
    # Building Flags for selection
    pass_flags = [False] * n
    for i in range(n):
        if i == 0:
            pass_flags[i] = True
            M = int(summary_df.loc[i, "mixtures"])
            details[M]["criteria"] = {
                "gain_vs_prev": np.nan,
                "gain_required": float(gain_thresh_mixtures),
                "se_candidate": float(summary_df.loc[i, "SE_LL"]) if np.isfinite(summary_df.loc[i, "SE_LL"]) else None,
                "rule": "gain >= max(gain_thresh_mixtures, 2*SE_candidate)",
                "pass_gain": True,
                "note": "baseline",
            }
        else:
            prev_mean = summary_df.loc[i - 1, "mean_LL"]
            curr_mean = summary_df.loc[i, "mean_LL"]
            cand_se = summary_df.loc[i, "SE_LL"]
            if np.isfinite(curr_mean) and np.isfinite(prev_mean):
                d = float(curr_mean - prev_mean)
                summary_df.loc[i, "delta"] = d
                se_term = 2.0 * float(cand_se) if np.isfinite(cand_se) else 0.0
                req = float(max(gain_thresh_mixtures, se_term))
                pass_flags[i] = bool(d >= req)
            else:
                d = np.nan
                req = float(gain_thresh_mixtures)
                pass_flags[i] = False
            M = int(summary_df.loc[i, "mixtures"])
            details[M]["criteria"] = {
                "gain_vs_prev": None if not np.isfinite(d) else float(d),
                "gain_required": req,
                "se_candidate": None if not np.isfinite(cand_se) else float(cand_se),
                "rule": "gain >= max(gain_thresh_mixtures, 2*SE_candidate)",
                "pass_gain": pass_flags[i],
            }
    # making final selection for model before returning
    selected_idx = 0
    for i in range(1, n):
        if pass_flags[i]:
            selected_idx = i
        else:
            break
    summary_df.loc[selected_idx, "selected"] = True
    return summary_df, details


