#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for aligning GMMHMM latent states and mixtures across subjects
Dr. Drew E. Winters
Created on 9/16/2025
"""

import numpy as np

# Alligining latent states across subjects
def align_model_to_expected(subj_model, global_model, wt_var=True):
    '''
    This function aligns to the preset expected state profiles learned from the global model.
    Specifically, the product of the sample model specified bold intensity for each state and probability of responding to the current trial by latent state to sort subject models to match the global model.
    This provides extra assurance that the identified state is consistent across participants.
    '''
    if wt_var == False:
        # If weights are fixed to the global model when it is only necessary to align to the global states
        order = np.argsort(global_model.means_.mean(axis=(1,2)) * global_model.weights_[:,0])[::-1]
    elif wt_var == True:
        # If weights vary by subject we need to align them to the subject adjusted alphas and global model means
        order = np.argsort(global_model.means_.mean(axis=(1,2)) * (subj_model.weights_ @ subj_model.weights_.mean(axis=0)))[::-1]
    else:
        raise ValueError("Invalid wt_var")
    subj_model.startprob_ = subj_model.startprob_[order]
    subj_model.transmat_ = subj_model.transmat_[order][:, order]
    subj_model.weights_ = subj_model.weights_[order]
    subj_model.means_ = subj_model.means_[order]
    subj_model.covars_ = subj_model.covars_[order]
    return subj_model, order

# Function for detecting the most likely 'respond' mixture per state, with ambiguity detection & fallback
def detect_respond_mixture(
    means, variances=None, baseline=0.0,
    rule="snr",              # 'snr' | 'absmean' | 'signed'
    respond_sign=None,       # +1 or -1 for 'signed' rule
    mode="per_state",        # 'per_state' or 'global'
    min_separation=0.35,     # mark ambiguous if sep < this
    near_tie_abs=0.02,       # mark ambiguous if |s1-s2| < this (SNR/score units)
    near_tie_rel=0.02,       # or if s_max/s_min - 1 < this
    fallback="global",       # 'global' | 'fixed' | 'keep'
    prefer_m=1,              # used when fallback='fixed'
    return_details=False,
    eps=1e-12
):
    """
    Identifying the most likely 'respond' mixture per state, with ambiguity detection & fallback.
    This is recomended to not prespecify because GMMHMM training may swap mixture indices.
    means: (C, M, N);
    variances (optional): (C, M, N)
    Returns idx: (C,) mixture index per state (or scalar broadcast if mode='global').
    """
    C, M, N = means.shape
    if variances is None:
        variances = np.ones_like(means)
    if rule == "snr":
        score = np.median(np.abs(means - baseline) / np.sqrt(np.maximum(variances, eps)), axis=2)  # (C,M)
    elif rule == "absmean":
        score = np.median(np.abs(means - baseline), axis=2)                                       # (C,M)
    elif rule == "signed":
        if respond_sign is None:
            raise ValueError("Set respond_sign=+1 or -1 for rule='signed'.")
        score = np.median(respond_sign * (means - baseline), axis=2)
    else:
        raise ValueError("rule must be 'snr' | 'absmean' | 'signed'")
    # primary choice
    if mode == "per_state":
        idx = np.argmax(score, axis=1)  # (C,)
    elif mode == "global":
        m_star = int(np.argmax(np.median(score, axis=0)))
        idx = np.full(C, m_star, dtype=int)
    else:
        raise ValueError("mode must be 'per_state' or 'global'")
    # separation (only meaningful when M=2)
    sep = None
    if M == 2:
        num = np.abs(means[:,1,:] - means[:,0,:])
        den = np.sqrt(0.5 * (variances[:,1,:] + variances[:,0,:]) + eps)
        sep = np.median(num / den, axis=1)  # (C,)
    # near-tie & ambiguity mask (M assumed 2; generalizes by looking at max/min)
    s_max = score.max(axis=1)
    s_min = score.min(axis=1)
    delta = s_max - s_min
    rel   = s_max / (s_min + eps) - 1.0
    ambiguous = (delta < near_tie_abs) | (rel < near_tie_rel)
    if sep is not None:
        ambiguous |= (sep < min_separation)
    if (ambiguous.any()) and (mode == "per_state"):
        if fallback == "global":
            m_star = int(np.argmax(np.median(score, axis=0)))
            idx = np.where(ambiguous, m_star, idx)
        elif fallback == "fixed":
            idx = np.where(ambiguous, int(prefer_m), idx)
        elif fallback == "keep":
            pass
        else:
            raise ValueError("fallback must be 'global' | 'fixed' | 'keep'")
    details = {
        "score": score, "sep": sep, "delta": delta, "rel": rel,
        "ambiguous": ambiguous, "fallback": fallback, "rule": rule
    }
    return (idx, details) if return_details else idx

