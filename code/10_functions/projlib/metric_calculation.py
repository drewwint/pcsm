#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for metric calculations for PCSM
Dr. Drew E. Winters
Created on 9/16/2025

Note - to reload a function after modifying
>>> import importlib
>>> importlib.reload(mc) # where mc = metric_calculation
"""

import numpy as np

# Diagnostic Metrtics-------------------------------------------------------------
## Response Probability at time point t (Pt)_______________________
def compute_pt(pi_t, alpha, clip_alpha=True, renorm_pi=False, eps=1e-12):
    """
    Compute the per-timepoint probability of a task-evoked response.
        P_t = sum_s pi_t[t, s] * alpha[s]

    Inputs
    -------
    pi_t : array-like, shape (T, S)
        Posterior over latent state (s) across all states (S) at each timepoint.
    alpha : array-like, shape (S,)
        State-conditional probability of responding given state s. Must be in [0, 1].
    
    Parameters
    ----------
    clip_alpha : bool, default True
        If True, clip alpha to [0, 1].
    renorm_pi : bool, default False
        If True, renormalize each row of `pi_t` to sum to 1 (with `eps` floor).
        Not typically needed but less stable posterior values may require it
    eps : float, default 1e-12
        Numerical floor for stabilization.

    Returns
    -------
    Pt : ndarray, shape (T,)
        Probability the brain is responding at each timepoint.

    Raises
    ------
    ValueError
        If shapes are incompatible.
    """
    # Validating inputs
    import numpy as np
    pi_t = np.asarray(pi_t, dtype=float)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)  # ensure (S,)
    if pi_t.ndim != 2:
        raise ValueError("pi_t must be 2D (T, S).")
    T, S = pi_t.shape
    if alpha.shape[0] != S:
        raise ValueError(f"alpha must have length S={S}; got {alpha.shape}.")
    # Applying parameter selections
    if renorm_pi:
        row_sums = np.clip(pi_t.sum(axis=1, keepdims=True), eps, None)
        pi_t = pi_t / row_sums
    if clip_alpha:
        alpha = np.clip(alpha, 0.0, 1.0)
    # Calculating Pt
    Pt = pi_t @ alpha  # (T, S) @ (S,) = (T,)
    return Pt



## Temporal Stability at timepoint t (rho)_________________________
## adaptive smoothing
def soften_pi_ema_adaptive(pi_t, lam_low=0.15, lam_high=0.85, eps=1e-12):
    """
    Adaptive EMA smoother for posterior probabilities.

    For each step t, compute a volatility score from the Jensen-Shannon
    divergence between pi_{t-1} and pi_t. Map (low volatility --> high lambda)
    so the EMA is *smoother* in stable periods and *more responsive* in
    volatile periods:
        out[t] = lambda_t * out[t-1] + (1 - lambda_t) * pi_t

    where lambda_t is an element of [lam_low, lam_high] 
          and larger lambda means heavier smoothing.

    Inputs
    ------
    pi_t : array-like, shape (T, K)
        Posteriors per timepoint (rows).

    Parameters
    ----------
    lam_low, lam_high : float
        Bounds for EMA weight (0 ≤ lam_low ≤ lam_high ≤ 1).
        lam_high is used in stable periods (more inertia), lam_low in volatile ones.
    eps : float
        Numerical floor for logs/divisions.

    Returns
    -------
    out : ndarray, shape (T, K)
        Smoothed posteriors.
    """
    # Validating inputs
    pi_t = np.asarray(pi_t, dtype=float)
    if pi_t.ndim != 2:
        raise ValueError("pi_t must be 2D (T, K).")
    # Setting up parameters
    T, K = pi_t.shape
    out = np.empty_like(pi_t)
    out[0] = pi_t[0]
    # Volatility via Jensen-Shannon divergence between t-1 and t
    p, q = pi_t[1:], pi_t[:-1]
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)), axis=1)
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)), axis=1)
    js = 0.5 * (kl_pm + kl_qm)
    vol = js / np.log(2.0)            # in [0,1]
    rho = 1.0 - vol                   # stability in [0,1]
    # map stability to EMA weight
    lam_t = lam_low + (lam_high - lam_low) * rho  # element in [lam_low, lam_high]
    for t in range(1, T):
        lam = lam_t[t-1]              # use stability from last step
        out[t] = lam * out[t-1] + (1.0 - lam) * pi_t[t]
    return out

## Rho 
def compute_rho(pi_t, smooth=True, lam_low=0.15, lam_high=0.85):
    """
    Temporal stability of posteriors between consecutive timepoints.

    Stability is defined as:
        rho[t] = 1 - TV(pi_{t-1}, pi_t)  for t >= 1,
    where TV is total variation distance: TV(p,q) = 0.5 * ||p - q||_1.
    By construction, rho is an element of [0,1] if each row sums to 1.

    An optional adaptive EMA smoother is applied first to improve SNR.

    Inputs
    ----------
    pi_t : array-like, shape (T, K)
        Posteriors per timepoint (rows).

    Parameters
    ----------
    smooth : bool, default True
        If True, smooth with adaptive EMA before computing stability.
    lam_low, lam_high : float
        Bounds for EMA smoothing (passed through).
    eps : float
        Numerical floor for safety.

    Returns
    -------
    rho : ndarray, shape (T,)
        Temporal stability, with rho[0] copied from rho[1] (or set to 1.0 if T=1).
    """
    # Validating inputs
    pi_t = np.asarray(pi_t, dtype=float)
    if pi_t.ndim != 2:
        raise ValueError("pi_t must be 2D (T, K).")
    # Smoothing
    if smooth:
        pi_t = soften_pi_ema_adaptive(pi_t, lam_low=lam_low, lam_high=lam_high)
    else:
        pi_t = pi_t
    # Distance
    d = 0.5 * np.sum(np.abs(pi_t[1:] - pi_t[:-1]), axis=1)
    # Rho from distance
    rho = 1.0 - d
    ## padding first point
    rho = np.r_[rho[0], rho]
    return rho


## Distance as an indicator of certainty (D^maha)__________________
## Helper to vectorize computations
def fast_mahalanobis_batch(X, Y, VI):
    """
    Vectorized Mahalanobis distance between X and Y.

    Inputs
    ------
    X, Y: array-like, (T, N)
        Two time-aligned sequences of N-dimensional observations.
    VI: array-like, (N, N)
        Inverse covariance (precision) matrix for Mahalanonis metric.
    
    Returns
    ------- 
    dists: ndarray, (T,) 
        Vector of Mahalanobis distances sqrt((x - y)^T VI (x - y)) per timepoint.
    """
    # Validating inputs
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    VI = np.asarray(VI, dtype=float)
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have identical shape; got {X.shape} vs {Y.shape}.")
    if VI.shape[0] != VI.shape[1] or VI.shape[0] != X.shape[1]:
        raise ValueError("VI must be (N,N) with N == X.shape[1].")
    # Calculating distance
    delta = X - Y  # (T, N)
    left = np.dot(delta, VI)      # (T, N)
    ## element-wise dot product for each row
    dists = np.einsum("td,td->t", left, delta) #(T,)
    return np.sqrt(dists)

## D^maha
def compute_D_maha_fir_mahalanobis(posterior_probs, means, bold):
    """
    Mahalanobis-based D^maha between expected and observed BOLD.

    Concept
    -------
    For each timepoint t, form the **expected** BOLD across nodes by mixing
    state-specific means with the posterior over states, then measure how far
    the **observed** BOLD is from that expectation under the (node) covariance
    estimated from the observed BOLD. Larger distance --> larger deviation from
    the model's expectation (potentially higher cognitive load / mismatch).

    Inputs
    ------
    posterior_probs : ndarray, shape (T, S)
        Posteriors over S latent states at each timepoint. 
    means : ndarray, shape (S, M, N)
        State x mixture x node mean patterns. Mixtures are averaged across M
        after mixing by posteriors to produce expected node activity.
    bold : ndarray, shape (E, D, N)
        Observed BOLD arranged as events x FIR-delays x nodes. Assumes the
        flattened order (E*D, :) aligns with time T in posterior_probs.

    Returns
    -------
    D_maha : ndarray, shape (T,)
        Distance scores scaled to [0,1] after robust scaling.

    Notes
    -----
    * Enforces T == E*D (time alignment) and N == N across inputs.
    * Uses pseudo-inverse of a ridge-regularized covariance for stability.
    * Robust scaling: median/IQR z, winsorize extreme values [1 and 99th percetiles], 
      then min-max map to [0,1] using those cutpoints.
    """
    # Validating inputs
    posterior_probs = np.asarray(posterior_probs, dtype=float)
    means = np.asarray(means, dtype=float)
    bold = np.asarray(bold, dtype=float)
    if posterior_probs.ndim != 2:
        raise ValueError("posterior_probs must be 2D (T, S).")
    if means.ndim != 3:
        raise ValueError("means must be 3D (S, M, N).")
    if bold.ndim != 3:
        raise ValueError("bold must be 3D (E, D, N).")
    T, S = posterior_probs.shape
    S_m, M, N_m = means.shape
    E, D_fir, N = bold.shape
    if S_m != S:
        raise ValueError(f"means first dim (S={S_m}) must match posterior S={S}.")
    if N != N_m:
        raise ValueError(f"bold N={N} must match means N={N_m}.")
    if T != E * D_fir:
        raise ValueError(f"Time mismatch: posterior T={T} must equal E*D={E*D_fir}.")
    # Compute expected BOLD
    expected_bold = np.einsum("ts,snv->tnv", posterior_probs, means)  # (T, M, N)
    # Flatten spatial dimensions
    bold_flat = bold.reshape(E * D_fir, N)         # (T,N)
    expected_flat = expected_bold.mean(axis=1)     # (T,N)
    # Covariance and inverse
    cov = np.cov(bold_flat, rowvar=False)
    inv_cov = np.linalg.pinv(cov)  # Pseudo-inverse for stability
    # Mahalanobis distances
    D = fast_mahalanobis_batch(expected_flat, bold_flat, inv_cov)
    # Robust scale and normalize
    D_z = (D - np.median(D)) / ((np.quantile(D, 0.75) - np.quantile(D, 0.25)) + 1e-6)
    D_z_max = np.quantile(D_z,0.99)
    D_z_min = np.quantile(D_z,0.01)
    D_z = np.array(D_z).clip(D_z_min,D_z_max)
    D_maha = (D_z - D_z_min) / (D_z_max - D_z_min)
    return D_maha


# Function for diagnostic flagging of trials__________________________
def flag_trials(
    Dmaha_val,                 # data values for an individual subject
    rho_val, 
    pt_val,           
    fir_bins=5,
    min_frac=0.50             # majority rule; e.g., 0.50 of fir bins indicate trial removal
    ):
    """
    Flag potentially unstable timepoints and identify trials (consecutive FIR blocks)
    with less reliable decoding.

    The time series is split into non-overlapping blocks of length `fir_bins`.
    A timepoint is flagged if:
      * D^maha < tau_D  AND  rho < tau_rho  AND  Pt < tau_Pt,
      * OR D^maha is NaN.

    A block (trial) is flagged if the fraction of its bins that are flagged
    exceeds `min_frac`. (Note: this implementation uses integer division in
    the per-block fraction, consistent with the current behavior). Flagged timeperiods
    are identified using simulation-based thresholds that are hardcoded into the function.

    Inputs
    ----------
    Dmaha_val : array_like, shape (T,)
        D^maha values per timepoint.
    rho_val   : array_like, shape (T,)
        Temporal stability metric per timepoint (typically in [0, 1]).
    pt_val    : array_like, shape (T,)
        P_t metric per timepoint (typically in [0, 1]).

    Parameters
    ----------
    fir_bins : int
        Block size (number of timepoints per trial).
        Must exactly divide T; otherwise an AssertionError is raised.
    min_frac : float in [0, 1]
        Minimum fraction of flagged bins within a block for that block to be flagged.

    Returns
    -------
    time_flags : list[int]
        Indices of timepoints to remove (flagged).
    trial_flags : list[int]
        Indices of blocks/trials to remove (flagged), numbered 0..n_trials-1.
    info : dict
        Summary information:
          {
            "T": int,                # number of timepoints
            "fir_bins": int,
            "min_frac": float,
            "n_trials": int,
            "n_trials_flagged": int,
            "n_timepoints_flagged": int
          }

    Notes
    -----
    * This routine assumes the series length T is an exact multiple of `fir_bins`.
      An assertion ensures this is the case. Given that FIR bins are the basis of timeweries,
      if this assumption does not hold there is an error in FIR estimation or subject alighnment.
    * NaN in D^maha is always treated as a timepoint failure (flagged).
    """
    # Validating inputs
    Dmaha = np.array(Dmaha_val)
    rho   = np.array(rho_val)
    Pt    = np.array(pt_val)
    assert Dmaha.shape == rho.shape == Pt.shape, "All inputs must have same length"
    T = Dmaha.size
    assert T % fir_bins == 0, f"T={T} must be a multiple of fir_bins={fir_bins}"
    n_trials = T//fir_bins
    # thresholds for metrics tau (simulation-based)
    tau_D=0.7409035409035409; tau_rho=0.8604972640007917; tau_Pt=0.1524348304585251   
    # Timepoint level flags
    time_flags = []
    for ii in range(T):
      if np.isnan(Dmaha[ii]) or (Dmaha[ii] < tau_D and (rho[ii] < tau_rho and Pt[ii] < tau_Pt)):
        time_flags.append(ii)
    # Block/Trial level flags
    r=np.arange(0,T,fir_bins)
    perc_fl = []
    for ii in r:
      block_idx = np.r_[ii:ii + fir_bins]
      # Count how many of this block’s indices are in time_flags
      n_in_block = np.count_nonzero(np.intersect1d(block_idx, time_flags))
      # Integer division by fir_bins: yields 0 unless all bins are flagged (kept as-is)
      perc_fl.append(n_in_block // fir_bins)
    trial_flags = []
    for jj in range(len(perc_fl)):
      if perc_fl[jj]>min_frac:
        trial_flags.append(jj)
    # Summary
    info = {
        "T": int(T),
        "fir_bins": int(fir_bins),
        "min_frac": float(min_frac),
        "n_trials": int(n_trials),
        "n_trials_flagged": int(np.count_nonzero(trial_flags)),
        "n_timepoints_flagged": int(np.count_nonzero(time_flags))
    }
    return time_flags, trial_flags, info


# Decoding Functions-----------------------------------------------------------------
# Node-Level
## Bold response probability per node (B^node)_________________________
## B^node helper: detecting responding mixture per state
def detect_respond_mixture(
    means, 
    variances=None, 
    baseline=0.0,
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

    Why this exists
    ---------------
    In GMM-HMMs, mixture indices can permute between trainings. This helper
    re-identifies, per state, which mixture most plausibly corresponds to a
    “respond” component using node-wise evidence aggregated across nodes.

    Inputs
    ------
    means : array, shape (C, M, N)
        Per-state (C) / mixture (M) / node (N) means.

    Parameters
    ----------
    variances : array or None, shape (C, M, N)
        Per-state/mixture/node variances. If None, ones are used.
    baseline : float
        Baseline against which the mean departures are scored.
    rule : {'snr','absmean','signed'}
        How to score the “respond” mixture within a state across nodes:
          - 'snr'     : median over nodes of |mean - baseline| / sqrt(var)
          - 'absmean' : median over nodes of |mean - baseline|
          - 'signed'  : median over nodes of respond_sign * (mean - baseline)
        Larger scores --> more likely “respond”.
    respond_sign : {+1,-1} or None
        Required when rule='signed'; ignored otherwise.
    mode : {'per_state','global'}
        'per_state': choose a possibly different mixture per state.
        'global'   : choose a single mixture index (argmax of the state-median scores)
                     then broadcast to all states.
    min_separation : float
        For M==2 only: flag ambiguity if the SNR-like separation between the two
        mixtures (median over nodes) is < min_separation.
    near_tie_abs : float
        Flag ambiguity if max(score) - min(score) < near_tie_abs (per state).
    near_tie_rel : float
        Flag ambiguity if max(score)/min(score) - 1 < near_tie_rel (per state).
    fallback : {'global','fixed','keep'}
        How to resolve ambiguous states when mode='per_state':
          - 'global': replace ambiguous states with the “global” winner.
          - 'fixed' : replace ambiguous with mixture index prefer_m.
          - 'keep'  : keep the original per-state argmax even if ambiguous.
    prefer_m : int
        Mixture index to use when fallback='fixed'.
    return_details : bool
        If True, also return diagnostic details.
    eps : float
        Numerical floor for stability in divisions.

    Returns
    -------
    idx : ndarray of shape (C,)
        Responding mixture index per state. (If mode='global', all entries identical.)
    details : dict (only if return_details=True)
        {
          'score': (C,M) array used for selection,
          'sep'  : (C,) separation for M==2 else None,
          'delta': (C,) s_max - s_min,
          'rel'  : (C,) s_max/s_min - 1,
          'ambiguous': (C,) bool mask,
          'fallback' : str,
          'rule'     : str
        }
    """
    # Validating inputs
    means = np.asarray(means, float)
    if means.ndim != 3:
        raise ValueError("means must have shape (C, M, N).")
    C, M, N = means.shape
    if M < 1:
        raise ValueError("Mixture dimension M must be >= 1.")
    if variances is None:
        variances = np.ones_like(means)
    if rule == "snr":
        score = np.median(np.abs(means - baseline) / np.sqrt(np.maximum(variances, eps)), axis=2) # (C,M)
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
    # Separation (only meaningful when M=2)
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


## B^node
def compute_B_node(
    pi_t,           # (T, C)
    means,          # (C, M, N)
    bold,           # (T, N) or (E, D, N) with T=E*D
    variances=None, # (C, M, N) or None
    mix_weights=None, # (C, M) or None (equal within state)
    respond_m=None,   # None -> auto-detect per state; or int; or (C,)
    baseline=0.0,
    respond_rule="snr",
    eps=1e-12
    ):
    """
    Node-level “responding” probability q_{n,t} by mixing HMM posteriors with
    mixture responsibilities for the “respond” component per state.

    Definition
    ----------
    For each time t and node n:
      1) Within each state c, compute responsibilities r_{c,m}(n|t) over mixtures:
         r_{c,m}(n|t) ∝ w_{c,m} · N(y_{t,n} | mu_{c,m,n}, σ²_{c,m,n}),
         normalized over m.
      2) Select the responding mixture index m* per state (either provided via
         `respond_m` or automatically detected by `detect_respond_mixture`).
      3) Combine across states with the HMM posteriors:
         q_{n,t} = Σ_c  π_t(c) · r_{c,m*}(n|t).

    Inputs
    ------
    pi_t : array, shape (T, C)
        HMM posteriors per timepoint (rows need not be perfectly normalized; they
        are used as given).
    means : array, shape (C, M, N)
        Per-state/mixture/node means for the emission model.
    bold : array, shape (T, N) or (E, D, N) with E*D == T
        Observations per timepoint (flattened automatically if 3D).
    variances : array or None, shape (C, M, N)
        Per-state/mixture/node variances for the emission model.
        If None, ones are used.
    mix_weights : array or None, shape (C, M)
        Mixture weights within each state. If None, set to uniform within state.
        Each state's weights are renormalized to sum to 1.
    respond_m : None | int | array shape (C,)
        If None: infer per-state responding mixture via `respond_rule`.
        If int: use the same mixture index for all states.
        If array: must have length C.

    Parameters
    ----------
    baseline : float
        Baseline passed to `detect_respond_mixture` when respond_m is None.
    respond_rule : {'snr','absmean','signed'}
        Rule passed to `detect_respond_mixture` when respond_m is None.
    eps : float
        Numerical floor for stability in PDF/normalizations.

    Returns
    -------
    B_node : array, shape (T, N)
        Responding probability per node/timepoint.
        q_{n,t} = sum_c pi_{t,c} * P(M=respond | y_{t,n}, C=c)
    respond_idx : array, shape (C,)
        The responding mixture index used for each state.
    """
    # Helper: 1D Gaussian PDF
    def _gauss_pdf_1d(y, mu, var, eps=1e-12):
        var = np.maximum(var, eps)
        return np.exp(-0.5 * (y - mu)**2 / var) / np.sqrt(2.0 * np.pi * var)
    # Validating inputs
    pi = np.asarray(pi_t, float); T, C = pi.shape
    Y = np.asarray(bold, float)
    if Y.ndim == 3:
        E, D, N = Y.shape; assert E*D == T; Y = Y.reshape(T, N)
    else:
        T_b, N = Y.shape; assert T_b == T
    means = np.asarray(means, float); C2, M, N2 = means.shape
    assert (C2, N2) == (C, N)
    if variances is None:
        variances = np.ones_like(means)
    else:
        variances = np.asarray(variances, float); assert variances.shape == means.shape
    if mix_weights is None:
        mix_weights = np.full((C, M), 1.0/M, float)
    else:
        mix_weights = np.asarray(mix_weights, float)
        mix_weights /= (mix_weights.sum(axis=1, keepdims=True) + eps)
    # detect responding mixture per state if not provided
    if respond_m is None:
        respond_idx = detect_respond_mixture(means, variances, baseline, respond_rule)
    else:
        respond_idx = np.asarray(respond_m, int)
        if respond_idx.ndim == 0: respond_idx = np.full(C, int(respond_idx))
        assert respond_idx.size == C
    # Calculating Bnode
    B_node = np.empty((T, N), float)
    for t in range(T):
        y = Y[t]  # (N,)
        # responsibilities r[c,n,m] = P(M=m | y_{t,n}, C=c)
        like = np.empty((C, M, N), float)
        for c in range(C):
            # unnormalized mixture evidence per node
            for m in range(M):
                like[c, m] = mix_weights[c, m] * _gauss_pdf_1d(y, means[c, m], variances[c, m], eps)
            # normalize over mixtures for each node
            like[c] /= (like[c].sum(axis=0, keepdims=True) + eps)
        # pick the responding mixture responsibility per state
        r_resp = like[np.arange(C), respond_idx]           # (C, N)
        # mix across states with pi_t
        q = (pi[t, :, None] * r_resp).sum(axis=0)          # (N,)
        B_node[t] = np.clip(q, 0.0, 1.0)
    return B_node, respond_idx

## B^node FDR correction                                                                                                                                                                                                                                                                                                                                                                                              
def bnode_fdr(B_node, alpha=0.25, return_mask=False):
    """
    Row-wise adaptive FDR gate for node-level response probabilities.

    Purpose
    -------
    Given a (T, N) matrix Q of per-timepoint (rows) node probabilities in [0,1],
    this selects, for each time t, the largest prefix of nodes (after sorting
    Q[t] descending) whose *expected* false discovery rate (FDR) is ≤ alpha.
    Expected false discoveries for a prefix of size k are
        sum_{i=1..k} (1 - q_(i))
    where q_(i) are the sorted probabilities; expected FDR is that sum / k.

    Inputs
    ------
    B_node : array-like, shape (T, N)
        Probabilities in [0,1]; NaNs are treated as 0 (never selected).
    alpha : float in (0,1)
        Target expected FDR per timepoint.
    return_mask : bool
        If True, also return a boolean selection mask of shape (T, N).

    Returns
    -------
    tau : (T,) float
        Per-timepoint threshold: the k*-th largest probability when k*>0; 1.0 if no selection.
    k    : (T,) int
        Number of selected nodes at each timepoint (may be 0).
    fdr  : (T,) float
        Realized expected FDR for the chosen prefix (0 when k=0).
    mask : (T, N) bool   [only if return_mask=True]
        Selection mask Q >= tau per row (ties included). All False when k=0.
    """
    # Validating inputs
    Q = np.asarray(B_node, float)
    if Q.ndim != 2:
        raise ValueError("B_node must be a 2D array of shape (T, N).")
    T, N = Q.shape
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    Q = np.clip(Q, 0.0, 1.0)
    # sort each row in descending order
    idx = np.argsort(-Q, axis=1)
    row = np.arange(T)[:, None]
    qs = Q[row, idx]                          # (T, N) sorted per row
    # cumulative expected false discoveries and FDR per prefix
    k_vec = np.arange(1, N+1, dtype=float)[None, :]
    cum_false = np.cumsum(1.0 - qs, axis=1)
    fdr = cum_false / k_vec
    # largest k with FDR <= alpha
    ok = fdr <= alpha
    ## For rows with at least one True, take the last True index + 1; else 0
    k_star = ok.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1)
    ## argmax on reversed gives first True from the right; convert back to forward index
    has_any = ok.any(axis=1)
    k_star = np.where(has_any, k_star + 1, 0).astype(int)
    # thresholds from sorted qs (1.0 if none)
    tau = np.full(T, 1.0, float)
    sel = k_star > 0
    tau[sel] = qs[sel, k_star[sel]-1]
    # realized expected FDR
    if return_mask:
        mask = Q >= tau[:, None]
        num_false = np.sum((1.0 - Q) * mask, axis=1)
    else:
        # compute numerator using the sorted prefix (no mask needed)
        num_false = np.zeros(T, float)
        num_false[sel] = cum_false[sel, k_star[sel]-1]
    fdr_hat = np.zeros(T, float)
    fdr_hat[sel] = num_false[sel] / k_star[sel]
    if return_mask:
        return tau, k_star, fdr_hat, mask
    return tau, k_star, fdr_hat



# Whole brain
## Deviation from perfect uncertainty (D^sp)_________________________
## helper to choose pmf computation method
def _choose_pmf(q, mode="fft"):
    """
    Poisson-binomial PMF for independent Bernoulli probs q.

    Parameters
    ----------
    q : array-like, shape (N,)
        Success probabilities, expected in [0, 1]. NaNs are treated as 0.
    mode : {"fft","dp"}, default "fft"
        "fft": fast generating-function method using roots of unity.
        "dp" : dynamic programming (O(N^2)), numerically straightforward.

    Returns
    -------
    pmf : np.ndarray, shape (N+1,)
        P(K = k) for k = 0..N where K = sum Bernoulli(q_i). Sums to ~1.

    Notes
    -----
    * Falls back to DP if the FFT-produced PMF fails to normalize (rare).
    """
    def _pb_pmf_dp(q):
        q = np.asarray(q, dtype=float).ravel()
        N = q.size
        pmf = np.zeros(N+1, dtype=float); pmf[0] = 1.0
        for p in q:
            pmf[1:] = pmf[1:]*(1-p) + pmf[:-1]*p
            pmf[0] *= (1-p)
        return pmf
    def _pb_pmf_fft(q):
        q = np.asarray(q, dtype=float).ravel()
        N = q.size
        L = 1
        while L < N+1:
            L <<= 1
        k = np.arange(L)
        omega = np.exp(2j*np.pi*k/L)                # complex eval points
        poly = np.ones(L, dtype=complex)
        for p in q:
            poly *= (1.0 - p) + p*omega
        pmf = np.real(np.fft.ifft(poly))[:N+1]
        pmf = np.maximum(pmf, 0.0)
        s = pmf.sum()
        return pmf/s if s > 0 else _pb_pmf_dp(q)
    return _pb_pmf_dp(q) if mode == "dp" else _pb_pmf_fft(q)

## D^sp
def compute_Dsp(B_node, pmf_mode="fft", verbose = False):
    """
    Finite-sample deviation (D_sp) from a (T, N) matrix of node probabilities.
    Adapted for human cognitive task-based fMRI BOLD data from: 
        Li K, et al. (2020) doi: 10.1098/rsos.191553.

    Purpose
    -------
    For each time t, nodes n=1..N fire independently with probabilities q_{t,n}.
    Let K_t be the random count of active nodes. D_sp(t) is the expected
    *normalized absolute deviation* of K_t from a balanced split N/2:
        D_sp(t) = E[|K_t - N/2|]/(N/2) an element of [0, 1]

    Intuition: 0 --> activity mass is concentrated near N/2 (“balanced/mixed”);
               1 --> mass near the extremes 0 or N (“strongly serial/parallel”).

    Inputs
    -------
    B_node : array-like, shape (T, N)
        Per-timepoint node probabilities in [0,1]. NaNs are treated as 0.

    Parameters
    ----------
    pmf_mode : {"fft","dp"}, default "fft"
        Backend for the Poisson-binomial PMF computation (see _choose_pmf).
    verbose : bool, default False
        If True, also return a dictionary with serial/mixed/parallel timepoint
        indices and counts using hardcoded simulation-derived thresholds.

    Returns
    -------
    Dsp : np.ndarray, shape (T,)
        Finite-sample deviation per timepoint, clipped to [0, 1].
    If verbose is True:
        Dsp : np.ndarray, shape (T,)
        Dsp_info : dict
            {
              # Timepoint indices by regime (note: key names reflect current code)
              "timepoints": {
                  "serial_indx":   np.ndarray[int],
                  "mixed_indx":    np.ndarray[int],
                  "parallel_indx":  np.ndarray[int]   # (indices despite the name)
              },
              # Counts per regime (note the key name for parallel count)
              "n_timepoints": {
                  "serial_n_timepoints":             int,
                  "mixed_n_timepoints":              int,
                  "parallel_n_timepoints":  int
              }
            }

    Notes
    -----
    * Thresholds used when verbose=True:
        tau_dsp_lo = 0.52233887, tau_dsp_hi = 0.61220267
        serial   : D_sp > tau_dsp_hi
        mixed    : tau_dsp_lo < D_sp ≤ tau_dsp_hi
        parallel : D_sp < tau_dsp_lo
    * D_sp reduces to the exact expectation under the Poisson-binomial PMF of K_t.
    * Values are clipped to [0, 1] for numerical safety.
    """
    B_node = np.asarray(B_node, dtype=float)
    T, N = B_node.shape
    halfN = N/2.0
    z = np.arange(N+1, dtype=float)
    D = np.empty(T, dtype=float)
    for t in range(T):
        pmf = _choose_pmf(B_node[t], pmf_mode)
        D[t] = np.dot(np.abs(z - halfN), pmf) / (halfN if halfN > 0 else 1.0)
    Dsp = np.clip(D, 0.0, 1.0)
    if verbose:
        # Simulation derived thresholds (serial/parallel)
        low_th=0.52233887; high_th=0.61220267
        serial_indx = np.where(np.array(Dsp)>high_th)[0]
        mixed_indx = np.where((np.array(Dsp)<=high_th) & (np.array(Dsp)>low_th))[0]
        parallel_indx = np.where(np.array(Dsp)<low_th)[0]
        serial_len = len(serial_indx)
        mixed_len = len(mixed_indx)
        parallel_len = len(parallel_indx)
        Dsp_info={
            "timepoints":{
                "serial_indx":serial_indx,
                "mixed_indx":mixed_indx,
                "parallel_indx":parallel_indx
            },
            "n_timepoints":{
                "serial_n_timepoints":serial_len,
                "mixed_n_timepoints":mixed_len,
                "parallel_n_timepoints":parallel_len
            }
        }
        return Dsp, Dsp_info
    if not verbose:
        return Dsp


# Transition, Effort, and Load Function ----------------------------------------------
## Computing demand, resource levels, and load_________________________ 
##- main and impact outputs are comparable across all subjects/studies
def compute_transition_load(
        pi_t,
        d_sp,
        normalize='robust',             # {'rank','robust','zscore','none'}
        eps=1e-12,
        verbose=True
    ):
    """
    Compute a pair of time series that summarize control “load” dynamics from
    HMM posteriors and a serial-parallel deviation signal D_SP:
      * demand(t): an instantaneous, signed “flow” 
            + = recovery/settling (lower load)
            - = demand/strain (higher load)
      * resource_level(t): a cumulative, leaky “stock” (relative resources over time)
            Higher = more resources available
            lower = fewer resources available 
            A sustained negative mean demand produces a downward trend (no forced drift).

    The method mixes three stepwise changes:
      1) delta_U_proc(t): change in processing potential from d_SP
      2) delta_H(t): change in posterior entropy of y
      3) JSD(t): Jensen-Shannon divergence between y_{t-1} and y_t

    A non-parametric “synergy” term up-weights moments where both the D_SP
    mode transition and the HMM distributional change are large. The stock is a
    leaky integrator with a data-driven leak (half-life from mode dwell length)
    and a variance-matched gain.

    Inputs
    ------
    pi_t : array, shape (T, C)
        HMM posteriors y_t(c). Internally clipped to [eps, 1] and renormalized
        per row before entropy/JSD calculations.
    d_sp : array, shape (T,)
        Serial-parallel deviation per time point. Mode labels are defined via
        fixed thresholds (simulation-derived, hardcoded in the function):
            parallel: d_sp < tau_dsp_lo  (tau_dsp_lo = 0.52233887)
            mixed   : tau_dsp_lo ≤ d_sp ≤ tau_dsp_hi
            serial  : d_sp > tau_dsp_hi  (tau_dsp_hi = 0.61220267)

    Parameters
    ----------
    normalize : {'rank','robust','zscore','none'}, default 'robust'
        Per-component scaling before mixing:
          'rank'   : ECDF rank centered to [-1,1], non-parametric (no magnitude).
          'robust' : median/MAD with IQR fallback (Gaussian consistency factors 1.4826, 1.349).
          'zscore' : mean/SD z-score (not robust to outliers).
          'none'   : raw deltas (units differ).
    eps : float
        Numerical floor for logs/divisions.
    verbose : bool
        If True, also return diagnostic components.

    Method (summary)
    ----------------
    * Processing potential: for each t, define mode-specific center mu and width w
         using {min(d_sp), tau_dsp_lo, tau_dsp_hi, max(d_sp)}.
        U_proc(t) = ((d_sp(t) - mu_mode(t)) / w_mode(t))^2
        delta_U_proc(t) = U_proc(t) - U_proc(t-1)  (delta at t=0 set to 0)
    * HMM entropy & delta:
        H(t) = -sum_c y_t(c) log y_t(c) / log(C)  element of [0,1]
        delta_H(t) = H(t) - H(t-1)
    * Distributional change (JSD):
        JSD(t) = 1/2 KL(y_{t-1} || m) + 1/2 KL(y_t || m), m = 1/2(y_{t-1}+y_t)
        With median-center JSD before normalization.
    * Normalization:
        P  := normalized delta_U_proc
        E  := normalized delta_H
        Jn := normalized centered JSD
        Choice per `normalize` as above.
    * Data-driven synergy (non-parametric):
        mode_switch := ECDF-rank of boundary-crossing depth (0..1)
        hmm_switch  := ECDF-rank of JSD (0..1)
        synergy_raw = sqrt(mode_switch * hmm_switch)
        synergy     = synergy_raw - median(synergy_raw)
        synergy_weight = mean(abs(demand_core))
    * Instantaneous demand (flow):
        - demand_core(t) = -(P(t) + E(t) + Jn(t)) / 3
        - demand_raw(t)      = demand_core(t) - synergy_weight * synergy(t)
        - demand_norm(t) = robust-z(demand_raw) # cross-subject comparabilty 
        - demand_bounded = tahn(demand_norm(t))
        -- interpretation: ( + = recovery / settling; - = demand / strain )
    * Resource level (stock):
        - half-life from median mode dwell: half_life_steps = max(3, 2*D_med)
            lambda = 1 - 0.5**(1/half_life_steps)
        - variance-matched gain a so Var(level) ~= 1:
            a = sqrt( (1 - (1 - lambda)**2) / Var(demand) )
        - level[t] = (1 - lambda)*level[t-1] + a*demand[t]
        - robust between-subject scaling of `level`: robust z (MAD/IQR) then tanh to [-1,1]
            (only for `resource_level`, not for `demand`)

    Returns
    -------
    demand : array, shape (T,)
        Instantaneous signed flow of demand/recovery robust scaled (unbouded). 
        Positive = recovery; negative = demand.
    resource_level : array, shape (T,)
        Leaky integrated resource level (relative). 
        Higher = more resources, Lower = less resources, Downward trend reflects sustained demand.
    impact: dict represending load impact on subject
        {
          'impact_cumulative'    : change by unit step in demand (high = more impact, lower = less)
          'impact_duration'      : constant exponential decay time (in steps)  
        }
        If normalize='robust', these are standardized with simulation-derived
        center/scale; otherwise raw values are returned.
    Only if verbose=True then additional dicts"
        impact_severity : dict
            Bucketed interpretations of impact magnitude/duration (standardized only for 'robust').
        demand_info : dict
            Indices, means, counts, and proportions for high/low demand epochs and demand trend.
        resource_info : dict
            Indices, means, counts, and proportions for high/low resource epochs and resource trend.
        components : dict 
            {
            # raw primitives
            'proc_delta'         : delta_U_proc(t),
            'hmm_entropy_d'      : delta_H(t),
            'jsd_gamma'          : JSD(t),
            # switch & synergy diagnostics
            'mode_switch'        : ECDF-rank of d_SP crossing depth (0..1),
            'hmm_switch'         : ECDF-rank of JSD (0..1),
            'synergy'            : geometric-mean switch strength (median-centered),
            'synergy_weight'     : scalar weight applied to synergy,
            # flow composition
            'parts'              : {'P': P, 'E': E, 'Jn': Jn, 'U_proc': U_proc, 'H': H},
            'flows'              : {'demand_core': demand_core, 'demand': demand},
            # integrator settings (data-driven)
            'integrator'         : {'half_life_steps': half_life_steps,
                                    'lambda': lam,
                                    'alpha': a},
            # raw impact (scaled based on simulated scaling factors for stability/comparability)
            'impact_raw'         : {
                'impact_cumulative_raw'    : raw change by unit step in demand
                'impact_duration_raw'      : raw constant exponential decay time (in steps)}
            # labels & bookkeeping
            'labels'             : {'mode': mode, 'hmm_argmax': argmax y},
            'demand_level_trend' : mean(demand),   # sign & magnitude of drift across timeseries
            'normalization'      : normalization chosen,
            'warnings'           : any warnings
            }

    Notes
    -----
    * Thresholds tau_dsp_lo=0.52233887 and tau_dsp_hi=0.61220267 are
      simulation-derived and hardcoded to ensure consistency across runs.
    * Drift in `resource_level` is emergent rather than forced (no added drift).
    * 'rank' normalization is fully non-parametric but discards magnitude;
      'robust' (prefered) retains magnitude while being robust to outliers.
    * Robust scaling uses MAD with IQR fallback (Gaussian consistency),
      and tanh maps resource_level to [-1, 1] for comparability.
    """
    # validate & align
    pi_t=np.asarray(pi_t); d_sp=np.asarray(d_sp)
    if pi_t.ndim!=2: 
        raise ValueError("pi_t must be 2D (T,C).")
    T,C=pi_t.shape
    if d_sp.shape[0]!=T: 
        raise ValueError("d_sp length must equal pi_t.shape[0].")
    if np.any(~np.isfinite(d_sp)):
        ok=np.isfinite(d_sp); pi_t=pi_t[ok]; d_sp=d_sp[ok]; T=pi_t.shape[0]
        if T==0: 
            raise ValueError("No finite timepoints after filtering d_sp.")
    # mode labels from thresholds
    ## hardcoded tau for serialal/parallel thresholds (simulation based)
    tau_dsp_hi=0.61220267; tau_dsp_lo=0.52233887
    ## identifying labels
    mode=np.full(T,'mixed',dtype=object)
    mode[d_sp>tau_dsp_hi]='serial'
    mode[d_sp<tau_dsp_lo]='parallel'
    # Processing potential & delta:
    ## Defining each processing mode center (mu) and width (w) for each cut point (serial/mixed/parallel)
    ##-- finding min/max for placing along with attractor points
    dmin=float(np.nanmin(d_sp)); dmax=float(np.nanmax(d_sp))
    ##-- finding mu mid point from min/max and cut-points
    mu_par=0.5*(dmin+tau_dsp_lo); mu_mix=0.5*(tau_dsp_lo+tau_dsp_hi); mu_ser=0.5*(tau_dsp_hi+dmax)
    ##-- finding width parameters from min/max cutpoints
    w_par=max(tau_dsp_lo-dmin,eps); w_mix=max(0.5*(tau_dsp_hi-tau_dsp_lo),eps); w_ser=max(dmax-tau_dsp_hi,eps)
    ##-- defining mu and w by processing mode: if parallel use mu_par, elif mixed use mu_mix, else use mu serial
    mu=np.where(mode=='parallel',mu_par,np.where(mode=='mixed',mu_mix,mu_ser))
    ww=np.where(mode=='parallel',w_par,np.where(mode=='mixed',w_mix,w_ser))
    # Distance: 
    ## Quantitating the normalized squared distance from the mode specific points
    ##-- subtracting Dsp from mu divided by width of modes distribution squared
    U_proc=((d_sp-mu)/ww)**2
    ## Degree to which the current timepoint stepped away (positive) or toward (negative) attractor point
    ##-- initalizing empty list
    proc_delta=np.zeros(T)
    ##-- subtracting U_proc [1:] from [:-1] for a vectorized version of t - (t-1)
    ##-- given nothing to subtract at t0 this starts appending at [1:]
    proc_delta[1:]=U_proc[1:]-U_proc[:-1]
    # HMM entropy & delta
    ## Deriving latent state uncertainty: uncertainty = cognitive demands on process completion
    ##-- normalizing - safety step to ensure pi_t stays within expected bounds
    g_safe=np.clip(pi_t,eps,1.0); g_safe/=g_safe.sum(1,keepdims=True)
    ##-- normalized entropy of the posterior probabilities
    H=-np.sum(g_safe*np.log(g_safe),axis=1)/np.log(C+eps)
    ##-- change in uncertainty (positive = more uncertainty/demand, negative = less uncertainty, recovery)
    hmm_entropy_d=np.zeros(T); hmm_entropy_d[1:]=H[1:]-H[:-1]
    # Jensen-Shannon Divergence (JSD) between y_{t-1} and y_t
    ## Deriving distributional change (how different) for HMM posteriors at each step (mode/state independent)
    ## jsd vectorized
    p_kl = g_safe[1:]; q_kl = g_safe[:-1]
    m = 0.5*(p_kl + q_kl)
    J = np.zeros(T, float)
    # 0.5 * [ KL(p_kl||m) + KL(q_kl||m) ] with eps added stability for logs/divisions ~0
    J[1:] = 0.5*np.sum(p_kl*np.log((p_kl/(m+eps)) + eps), axis=1) + \
            0.5*np.sum(q_kl*np.log((q_kl/(m+eps)) + eps), axis=1)
    # Normalization helpers: scaling for robust cross-subject comparisons with different approaches
    ## robust z (median/MAD): used to keep relative magnitudes but dampen outliers (robust to outliers)
    def _robust_z(x, eps=eps):
        m = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - m))
        ## determining scaling for robust normalization
        ##-- First MAD * gaussian consistency factor (1.4826)
        if mad > eps:
            scale = 1.4826 * mad
        else:
        ##-- Fallback: IQR-based * IQR-gaussian consistency factor (1.349)
            q75, q25 = np.nanpercentile(x, 75), np.nanpercentile(x, 25)
            iqr = q75 - q25
            scale = (iqr / 1.349) if iqr > eps else 0.0 
        ## scaling
        rob_z = (x - m) / (scale + eps) if scale > 0 else (x - m)
        return rob_z
    ## z-score (mean/sd) normalization: assumes no outlier impact 
    def _zscore(x, eps=eps):
        mu=float(np.nanmean(x)); sd=float(np.nanstd(x))+eps
        z = (x-mu)/sd
        return z
    ## rank non-parametric normalization - model free and without magnitude information
    ##-- makes a rank of values to center and scale - only rank matters (no magnitude)
    def _ecdf_center_pm1(x):
        n=x.size
        if n==0: return x
        r=np.argsort(np.argsort(x,kind='stable'),kind='stable').astype(float)
        u=(r+0.5)/n 
        return 2.0*(u-0.5)
    # normalize components
    if normalize=='robust':
        P=_robust_z(proc_delta)
        E=_robust_z(hmm_entropy_d)
        Jn=_robust_z(J-float(np.nanmedian(J)))
    elif normalize=='zscore':
        P=_zscore(proc_delta)
        E=_zscore(hmm_entropy_d)
        Jn=_zscore(J-float(np.nanmedian(J)))
    elif normalize=='rank':
        P=_ecdf_center_pm1(proc_delta)
        E=_ecdf_center_pm1(hmm_entropy_d)
        Jc=J-float(np.nanmedian(J))
        Jn=_ecdf_center_pm1(Jc)
    elif normalize=='none':
        P=proc_delta
        E=hmm_entropy_d
        Jn=J-float(np.nanmedian(J))
    else:
        raise ValueError("normalize must be one of {'rank','robust','zscore','none'}.")
    # Data-driven synergy (non-parametric)
    ## Functions: 
    ##-- emperical cumulative distribution (ECDF) ranking
    def _ecdf01(x):
        n=x.size
        if n==0: return x
        r=np.argsort(           # rank of each element (stable links to origional order)
            np.argsort(x,kind='stable'),kind='stable').astype(float)
        return (r+0.5)/n           # converts ranks to ECDF probabilities at the midpoint
    ##-- boundary-crossing depth at t-1->t, scaled by mode widths
        ## quantifies (1) crosses in mode space (2) degree (depth) of cross
    def _cross_depth(d_prev,d_cur):
        # mode spaces: P: d<tau_lo ; M: tau_lo<d<tau_hi ; S: d>tau_hi
        def _width(d):
            if d<tau_dsp_lo: return max(tau_dsp_lo-dmin,eps)   # parallel
            if d>tau_dsp_hi: return max(dmax-tau_dsp_hi,eps)   # serial
            else: return max(0.5*(tau_dsp_hi-tau_dsp_lo),eps)  # mixed
        # identifying which and if a boundary was crossed
        crossed=False; depth=0.0
        if (d_prev<tau_dsp_lo and d_cur>=tau_dsp_lo) or (d_prev>=tau_dsp_lo and d_cur<tau_dsp_lo):
            b=tau_dsp_lo; crossed=True
            # penetration into both sides (min of normalized depths) is conservative
            depth=min(abs(d_prev-b)/_width(d_prev), abs(d_cur-b)/_width(d_cur))
        if (d_prev>tau_dsp_hi and d_cur<=tau_dsp_hi) or (d_prev<=tau_dsp_hi and d_cur>tau_dsp_hi):
            b=tau_dsp_hi; crossed=True
            depth=max(depth, min(abs(d_prev-b)/_width(d_prev), abs(d_cur-b)/_width(d_cur)))
        return depth if crossed else 0.0
    ## vectorize crossing depth
    cross_mag=np.zeros(T,float)
    for t in range(1,T):
        cross_mag[t]=_cross_depth(d_sp[t-1],d_sp[t])
    ## rank-scale both magnitudes to [0,1]
    mode_switch = _ecdf01(cross_mag)           # mode-switch strength (0..1)
    hmm_switch = _ecdf01(J)                   # HMM change strength via JSD (0..1)
    ## geometric-mean "AND" high only when both are high
    synergy_raw = np.sqrt(mode_switch*hmm_switch)
    synergy = synergy_raw - float(np.nanmedian(synergy_raw))  # zero-mean
    # resource level calculation
    ## demand base
    demand_core = -(P + E + Jn)/3.0
    ## include data-driven synergy with a small data-driven weight
    synergy_weight = np.mean(abs(demand_core))
    demand_raw = demand_core - synergy_weight*synergy
    demand = _robust_z(demand_raw)
    # Resource level (stock): leaky, model-free and data-driven
    ## data-driven half-life (from mode dwells) for time of impact on resources
    def _median_dwell(labels):
        runs=[]; r=1
        for t in range(1, labels.size):
            if labels[t]==labels[t-1]: r+=1
            else: runs.append(r); r=1
        runs.append(r)
        return max(1, int(np.median(runs))) if runs else 1
    D_med = _median_dwell(mode)
    half_life_steps = max(3, 2*D_med)            # memory spans ~2 dwells
    lam = 1.0 - (0.5**(1.0/half_life_steps))     # leak per step
    ## data-driven gain (variance match)
    var_e = float(np.nanvar(demand)) + eps
    target_var = 1.0                              # desired Var(level)
    alpha = ((target_var * (1 - (1 - lam)**2)) / var_e)**0.5
    ## integrate to derive level of resource
    level = np.zeros(T, float)
    for t in range(1, T):
        level[t] = (1.0 - lam)*level[t-1] + alpha*demand[t]
    # robust scaling level for an across subject comparable resource_level
    med = float(np.nanmedian(level))
    mad = float(np.nanmedian(np.abs(level - med))) + eps
    z = (level - med) / (1.4826 * mad)     # robust z (gaussian consistency)
    resource_level = np.tanh(z)              # in [-1, 1]
    # Impact summaries
    ## simulation derived scaling factors (only for 'robust')
    impact_c_center = 7.072609196481455
    impact_c_scale = 6.551053442838975
    impact_dur_center = 14.932726172912965
    impact_dur_scale = 14.952620075738565
    ## simulation derived impact thresholds (only for 'robust')
    cumulative_impact_th = 0.648978118117964
    duration_impact_th = 13.113130658730196
    ## simulation derived demand thresholds (only for 'robust')
    demand_high_th = 0.675287330352096
    demand_low_th = -0.749974752766904
    resource_high_th = 0.595688706474272
    resource_low_th = -0.700797686780789
    ## metrics
    imp_cumulative_raw = alpha/lam
    imp_duration_raw = 1.0/lam
    if normalize=='robust':
        imp_cumulative_std = (((imp_cumulative_raw) - impact_c_center)/impact_c_scale)
        imp_duration_std = (((imp_duration_raw) - impact_dur_center)/ impact_dur_scale)
        # impact calculations
        imp_cumulative_high = imp_cumulative_std > cumulative_impact_th
        imp_duration_high = imp_duration_std > duration_impact_th
        if imp_cumulative_high:
            imp_cumulative_prob = "Strong Problematic Impact"
            imp_cumulative_severity = 1
        if not imp_cumulative_high:
            imp_cumulative_prob = "Normative Impact"
            imp_cumulative_severity = 0
        if imp_duration_high:
            imp_duration_prob = "Long Problematic Duration"
            imp_duration_severity = 1
        if not imp_duration_high:
            imp_duration_prob = "Normative Duration"
            imp_duration_severity = 0
        impact = {
            "impact_cumulative":imp_cumulative_std,
            "impact_duration":imp_duration_std
        }
        # demand calculations
        ## high
        demand_high_idx = np.where(demand > demand_high_th)[0]
        demand_high_ave = np.mean(demand[demand_high_idx])
        demand_high_len = len(demand_high_idx)
        demand_high_proportion = demand_high_len/len(demand)
        ## low
        demand_low_idx = np.where(demand < demand_low_th)[0]
        demand_low_ave = np.mean(demand[demand_low_idx])
        demand_low_len = len(demand_low_idx)
        demand_low_proportion = demand_low_len/len(demand)
        # resource calculations
        ## high
        resource_high_idx = np.where(resource_level > resource_high_th)[0]
        resource_high_ave = np.mean(resource_level[resource_high_idx])
        resource_high_len = len(resource_high_idx)
        resource_high_proportion = resource_high_len/len(resource_level)
        ## low
        resource_low_idx = np.where(resource_level < resource_low_th)[0]
        resource_low_ave = np.mean(resource_level[resource_low_idx])
        resource_low_len = len(resource_low_idx)
        resource_low_proportion = resource_low_len/len(resource_level)
        warnings=[]
    else:
        # impact calculations
        imp_cumulative_prob = "Unknowing Impact Strength - Not Standardized"
        imp_cumulative_severity = np.nan
        imp_duration_prob = "Unknowing Impact Duration - Not Standardized"
        imp_duration_severity = np.nan
        impact = {
            "impact_cumulative":imp_cumulative_raw,
            "impact_duration":imp_duration_raw
        }
        # demand calculations
        ## high
        demand_high_idx = np.nan
        demand_high_ave = np.nan
        demand_high_len = np.nan
        demand_high_proportion = np.nan
        ## low
        demand_low_idx = np.nan
        demand_low_ave = np.nan
        demand_low_len = np.nan
        demand_low_proportion = np.nan
        # resource calculations
        ## high
        resource_high_idx = np.nan
        resource_high_ave = np.nan
        resource_high_len = np.nan
        resource_high_proportion = np.nan
        ## low
        resource_low_idx = np.nan
        resource_low_ave = np.nan
        resource_low_len = np.nan
        resource_low_proportion = np.nan
        warnings = ["Impact values are raw; for cross-subject comparison, standardize across the cohort."]
    # Building additional outputs
    if verbose:
        impact_severity = {
            "impact_strength_severity_group": imp_cumulative_severity,
            "impact_strength_problem_explain": imp_cumulative_prob,
            "impact_duration_severity_group": imp_duration_severity,
            "impact_duration_problem_explain": imp_duration_prob
        }
        demand_info = {
            "high_demand":{"demand_high_idx":demand_high_idx,
                           "demand_high_ave":demand_high_ave,
                            "demand_high_len":demand_high_len,
                            "demand_high_proportion":demand_high_proportion},
            "low_demand":{"demand_low_idx":demand_low_idx,
                          "demand_low_ave":demand_low_ave,
                          "demand_low_len":demand_low_len,
                          "demand_low_proportion":demand_low_proportion},
            'demand_trend':np.nanmean(demand)
        }
        resource_info = {
            "high_resource":{"resource_high_idx":resource_high_idx,
                           "resource_high_ave":resource_high_ave,
                            "resource_high_len":resource_high_len,
                            "resource_high_proportion":resource_high_proportion},
            "low_resource":{"resource_low_idx":resource_low_idx,
                          "resource_low_ave":resource_low_ave,
                          "resource_low_len":resource_low_len,
                          "resource_low_proportion":resource_low_proportion},
            'resource_level_trend':np.nanmean(resource_level),
        }
        components={'proc_delta':proc_delta,
                    'hmm_entropy_d':hmm_entropy_d,
                    'jsd_gamma':J,
                    'mode_switch':mode_switch,
                    'hmm_switch':hmm_switch,
                    'synergy':synergy,
                    'synergy_weight':synergy_weight,
                    'parts': {'P': P, 'E': E, 'Jn': Jn, 'U_proc': U_proc, 'H': H},
                    'integrator': {'half_life_steps': half_life_steps, 'lambda': lam, 'alpha': alpha},
                    'impact_raw':{"impact_cumulative_raw":imp_cumulative_raw, "impact_duration_raw":imp_duration_raw},
                    'flows': {
                        'demand_core': demand_core, 
                        'demand_raw':demand_raw, 
                        'demand_norm': demand, 
                        'demand_bounded': np.tanh(demand)
                        },
                    'labels':{'mode':mode,'hmm_argmax':np.argmax(g_safe,axis=1)},
                    'normalization':normalize,
                    'warnings':warnings}
        return demand, resource_level, impact, impact_severity, demand_info, resource_info, components
    else:
        return demand, resource_level, impact


## Cohort standardization of impact (not necesary)___________________________
## Standardizes by sample 
##- Uses sample to place all subjects on a common metric
##- this is not recomended but may be useful in QA or specific cases
def impact_cohort_norm(
    impacts,
    normalize="robust",   # {'z','robust','rank01','rank_pm1','rank'}
    eps=1e-12,
    verbose=True
):
    """
    Cohort-standardize impact metrics (impact_cumulative, impact_duration).

    Inputs
    ------
    impacts : list[dict] | dict[str, dict] | tuple(np.ndarray, np.ndarray)
        - If list/dict: each item must have keys:
              'impact_cumulative', 'impact_duration'
          Example list: [{'impact_cumulative': 2.3, 'impact_duration': 14.2}, ...]
          Example dict : {'S01': {...}, 'S02': {...}}
        - If tuple: (cum, dur) arrays of shape (N,).

    normalize : {'z','robust','rank01','rank_pm1','rank'}, default 'robust'
        'z'        : mean/SD z-score (sensitive to outliers).
        'robust'   : median/MAD with IQR fallback (MAD*1.4826; IQR/1.349).
        'rank01'   : ECDF mid-rank in (0,1].
        'rank_pm1' : centered ECDF in [-1,1].
        'rank'     : alias for 'rank_pm1'.

    eps : float, default 1e-12
        Numerical floor to avoid divide-by-zero.

    verbose : bool, default True
        If True, return (out_vals, info). If False, return out_vals only.

    Returns
    -------
    out_vals : dict
        {
          'cumulative': np.ndarray (N,), standardized values per `normalize`,
          'duration'  : np.ndarray (N,)
        }

    info : dict (returned if verbose)
        {
          'cumulative': {
              'z'       : np.ndarray (N,),
              'robust'  : np.ndarray (N,),
              'rank01'  : np.ndarray (N,),
              'rank_pm1': np.ndarray (N,),
              'stats'   : {'mean','std','median','mad','iqr','n'}
          },
          'duration': { ... same keys ... }
        }

    Notes
    -----
    * NaNs in inputs are preserved in outputs; stats ignore NaNs.
    * Robust scaling uses MAD with IQR fallback (Gaussian consistency factors 1.4826, 1.349).
    * Use 'rank_pm1' (or 'rank') for magnitude-free, fully non-parametric comparison.
    """
    # normalize input form to two arrays: cum, dur
    if isinstance(impacts, tuple) or isinstance(impacts, list) and len(impacts) == 2 and all(
        isinstance(a, (list, np.ndarray)) for a in impacts
    ):
        cum = np.asarray(impacts[0], dtype=float)
        dur = np.asarray(impacts[1], dtype=float)
        keys = None
    else:
        is_dict_input = isinstance(impacts, dict)
        keys = list(impacts.keys()) if is_dict_input else None
        items = ([impacts[k] for k in keys] if is_dict_input else list(impacts))
        cum = np.array([d["impact_cumulative"] for d in items], dtype=float)
        dur = np.array([d["impact_duration"]   for d in items], dtype=float)
    def _stats(x):
        m = np.isfinite(x)
        xv = x[m]
        n = int(m.sum())
        if n == 0:
            return dict(mean=np.nan, std=np.nan, median=np.nan, mad=np.nan, iqr=np.nan, n=0)
        mean   = float(np.nanmean(xv))
        std    = float(np.nanstd(xv))
        median = float(np.nanmedian(xv))
        mad    = float(np.nanmedian(np.abs(xv - median)))
        q75, q25 = np.nanpercentile(xv, 75), np.nanpercentile(xv, 25)
        iqr    = float(q75 - q25)
        return dict(mean=mean, std=std, median=median, mad=mad, iqr=iqr, n=n)
    def _z(x):
        m = np.isfinite(x)
        out = np.full_like(x, np.nan, dtype=float)
        if m.any():
            mu  = float(np.nanmean(x[m]))
            sd  = float(np.nanstd(x[m]))
            out[m] = (x[m] - mu) / (sd + eps)
        return out
    def _robust(x):
        m = np.isfinite(x)
        out = np.full_like(x, np.nan, dtype=float)
        if m.any():
            xv = x[m]
            med = float(np.nanmedian(xv))
            mad = float(np.nanmedian(np.abs(xv - med)))
            if mad > eps:
                scale = 1.4826 * mad
            else:
                q75, q25 = np.nanpercentile(xv, 75), np.nanpercentile(xv, 25)
                iqr = float(q75 - q25)
                scale = (iqr / 1.349) if iqr > eps else 0.0
            out[m] = (xv - med) / (scale + eps) if scale > 0 else (xv - med)
        return out
    def _rank01_and_pm1(x):
        m = np.isfinite(x)
        r01  = np.full_like(x, np.nan, dtype=float)
        rpm1 = np.full_like(x, np.nan, dtype=float)
        if m.any():
            xv = x[m]
            # stable double-argsort mid-ranks
            r = np.argsort(np.argsort(xv, kind='stable'), kind='stable').astype(float)
            u = (r + 0.5) / r.size
            r01[m]  = u
            rpm1[m] = 2.0*u - 1.0
        return r01, rpm1
    # compute all normalizations for both metrics
    cum_info = {}
    dur_info = {}
    ## stats
    cum_info['stats']    = _stats(cum)
    dur_info['stats']    = _stats(dur)
    ## z
    cum_info['z']        = _z(cum)
    dur_info['z']        = _z(dur)
    ## robust
    cum_info['robust']   = _robust(cum)
    dur_info['robust']   = _robust(dur)
    ## rank
    cum_r01, cum_rpm1    = _rank01_and_pm1(cum)
    dur_r01, dur_rpm1    = _rank01_and_pm1(dur)
    ## building info output
    cum_info['rank01']   = cum_r01
    cum_info['rank_pm1'] = cum_rpm1
    dur_info['rank01']   = dur_r01
    dur_info['rank_pm1'] = dur_rpm1
    # select the requested normalization
    norm_key = normalize
    if normalize == "rank":
        norm_key = "rank_pm1"
    if norm_key not in ("z","robust","rank01","rank_pm1"):
        raise ValueError("normalize must be one of {'z','robust','rank01','rank_pm1','rank'}.")
    out_vals = {
        f'impact_cumulative_{normalize}': cum_info[norm_key],
        f'impact_duration_{normalize}'  : dur_info[norm_key]
    }
    if verbose:
        info = {'cumulative': cum_info, 'duration': dur_info}
        return out_vals, info
    else:
        return out_vals

# Serial Bottelneck-----------------------------------------------------------------------------
## Computing serial bottleneck score_________________________________ 
##- outputs are scaled from simulation to be comparable across all subjects/studies
def compute_serial_bottleneck(mode, 
                              demand, 
                              verbose=True, 
                              input_normalized="robust", 
                              exit_window=3
    ):
    """
    Quantify a 'serial bottleneck' from mode labels and an instantaneous demand signal.

    Inputs
    ------
    mode   : array-like of length T
        Mode label per timepoint. Expected values include 'serial', 'mixed', 'parallel'
        (strings) but any labeling is accepted as long as 'serial' is identifiable.
    demand : array-like of length T
        Instantaneous signed flow from your pipeline (+ = recovery/settling; - = demand/strain).
    
    Parameters
    ----------
    verbose : bool, default True
        If True, return only the scalar bottleneck_index. If False, also return a metrics dict.
    input_normalized: {'robust', 'zscore', 'rank', 'none'}:
        How the input demand was normalized. if 'robust, the index is standardized with 
        simulation-derived denter/scale for cross-subject comparability; otherwise, the raw
        product is rturned (requires post-hoc cohort-standardization)
    exit_window : int, default 3
        Number of time bins after each serial bout used to compute the *exit demand*
        as the median of the window (clamped to nonnegative cost via max(0, -median)).

    Method (data-driven, no knobs)
   ----------------------
    1) Find all serial periods (maximal consecutive runs with mode=='serial').
    2) For each bout, compute:
       - dwell length (TRs) = number of samples in the bout.
       - exit demand = max(0, -demand[t_exit+1]) where t_exit is the last index of the bout.
         (Uses your sign convention: negative demand = cost. We flip and clamp.)
       If t_exit is the last sample, that bout has no exit demand.
    3) Summaries:
       - median/mean/p90 dwell lengths; fraction of time in serial; median/mean exit demand
    4) Primary metric:
       - bottleneck_index_raw = median_dwell * median_exit_demand
         (High if subject stayed longer in serial *and* it's costly to leave)

    Standardization & Severity (when input_normalized == 'robust')
    --------------------------------------------------------------
    z-like index (hardcoded simulation-based scaling values):
        bottleneck_index = (bottleneck_index_raw --> 17.138951571421238) / 25.410209599789127
    buckets (hardoced simulation-based threshold values):
        * Very-Long Problematic Bottleneck  : bottleneck_index > 0.9318624587258728
        * Long Problematic Bottleneck       : bottleneck_index > 0.0000010042364280
        * Normal Low Concern Bottleneck     : otherwise

    Returns
    -------
    If verbose is False:
        bottleneck_index : float
            Standardized if input_normalized == 'robust', else raw product.
    If verbose is True:
        bottleneck_index : float
            Standardized if input_normalized == 'robust', else raw product.
        severity : dict
            {"bottleneck_severity_group": {0,1,2 or nan},
             "bottleneck_problem_explain": str}
        metrics : dict
            {
              'bottleneck_index_raw'   : float,
              'runs'                   : np.ndarray[int],  # per-bout dwell lengths
              'median_dwell'           : float,
              'mean_dwell'             : float,
              'p90_dwell'              : float,
              'n_bouts'                : int,
              'fraction_time_serial'   : float,            # time in serial / T
              'n_exits'                : int,              # bouts with post-exit data
              'low_evidence'           : bool,             # True if n_exits < exit_window
              'exit_demand_per_bout'   : np.ndarray[float],
              'exit_demand_median'     : float,
              'exit_demand_mean'       : float,
              'warnings'               : list[str]
            }

    Notes
    -----
    * Scale-aware: if `demand` is robust/z-scored, the index is in “(TRs x z-units)”.
    * Cross-subject comparisons are appropriate when `demand` is robust-normalized; otherwise
      cohort standardization of the index is recommended.
    * If there are no serial samples, the index is 0.0 and arrays are empty.
    """
    # simulation derived scaling factors (for stability and cross-subject/study comparability)
    bottleneck_center = 17.138951571421238
    bottleneck_scale = 25.410209599789127
    # simulation derived cutoff values for long and very long bottleneck
    bottleneck_long = 0.0000010042364280
    bottleneck_very_long = 0.9318624587258728
    # verifying inputs
    mode = np.asarray(mode)
    demand = np.asarray(demand, dtype=float)
    T = mode.shape[0]
    if demand.shape[0] != T:
        raise ValueError("mode and demand must have the same length.")
    # Indices where we are in serial mode
    serial_idx = np.where(mode == 'serial')[0]
    if serial_idx.size == 0:
        idx = 0.0
        metrics = {
            'bottleneck_index_raw': 0.0,
            'runs': np.zeros(0, dtype=int),
            'median_dwell': 0.0, 'mean_dwell': 0.0, 'p90_dwell': 0.0,
            'n_bouts': 0,
            'fraction_time_serial': 0.0,
            'n_exits': 0,
            'low_evidence': True,
            'exit_demand_per_bout': np.zeros(0, dtype=float),
            'exit_demand_median': 0.0,
            'exit_demand_mean': 0.0,
            'warnings': ["No serial bouts found; index set to 0."]
        }
        severity = {
            "bottleneck_severity_group": np.nan if input_normalized != "robust" else 0,
            "bottleneck_problem_explain": "No serial bouts found"
        }
        return idx if not verbose else (idx, severity, metrics)
    # Split into maximal consecutive runs
    splits = np.split(serial_idx, np.where(np.diff(serial_idx) != 1)[0] + 1)
    bouts = [b for b in splits if b.size > 0]
    dwell = np.array([b.size for b in bouts], dtype=int)
    # Exit demand per bout = demand cost at the first step immediately after leaving serial
    exit_demands = []
    for b in bouts:
        t_exit = int(b[-1])
        if t_exit + 1 < T:
            j1 = t_exit + 1
            j2 = min(T, t_exit + 1 + exit_window)
            post = demand[j1:j2]
            post = post[np.isfinite(post)]  # ignore NaN/inf if present
            if post.size:
                exit_demands.append(max(0.0, -float(np.nanmedian(post))))
    exit_demands = np.array(exit_demands, dtype=float) if len(exit_demands) else np.zeros(0, dtype=float)
    n_exits = int(exit_demands.size)
    low_evidence = (n_exits < exit_window)
    # Safe summaries
    def _med(x): return float(np.nanmedian(x)) if x.size else 0.0
    def _mean(x): return float(np.nanmean(x)) if x.size else 0.0
    def _p90(x): return float(np.nanpercentile(x, 90)) if x.size else 0.0
    median_dwell = _med(dwell)
    exit_demand_median = _med(exit_demands)
    bottleneck_index_raw = median_dwell * exit_demand_median
    if input_normalized == "robust":
        bottleneck_index = np.tanh((bottleneck_index_raw - bottleneck_center)/bottleneck_scale)
        long = bottleneck_index > bottleneck_long
        very_long = bottleneck_index > bottleneck_very_long
        if very_long:
            bottleneck_prob = "Very-Long Problematic Bottleneck"
            bottleneck_severity = 2
        elif long:
            bottleneck_prob = "Long Problematic Bottleneck"
            bottleneck_severity = 1
        else:
            bottleneck_prob = "Normal Low Concern Bottleneck"
            bottleneck_severity = 0
        warnings=[]
    else: 
        bottleneck_index = np.tanh(bottleneck_index_raw)
        bottleneck_prob = "Unknown Scale Bottleneck Problem - Not Standardized"
        bottleneck_severity = np.nan
        warnings = ["Input not 'robust: for cross-subject comparisons, cohort-standardize this index."]
    if low_evidence:
        warnings.append(f"Low evidence: only {n_exits} exits with post-exit data (recommend >={exit_window}).")
    # building otuputs
    if not verbose:
        return bottleneck_index
    elif verbose:
        severity = {
            "bottleneck_severity_group": bottleneck_severity,
            "bottleneck_problem_explain": bottleneck_prob
        }
        metrics = {
            'bottleneck_index_raw': bottleneck_index_raw,
            'runs': dwell,
            'median_dwell': median_dwell,
            'mean_dwell': _mean(dwell),
            'p90_dwell': _p90(dwell),
            'n_bouts': int(len(dwell)),
            'fraction_time_serial': float(serial_idx.size) / float(T),
            'n_exits': n_exits,
            'low_evidence': low_evidence,
            'exit_demand_per_bout': exit_demands,
            'exit_demand_median': exit_demand_median,
            'exit_demand_mean': _mean(exit_demands),
            'warnings': warnings
        }
        return bottleneck_index, severity, metrics

## Cohort standardization of impact (not necesary)___________________________
## Standardizes by sample 
##- Uses sample to place all subjects on a common metric
##- this is not recomended but may be useful in QA or specific cases
def bottleneck_cohort_norm(
    idx,
    normalize= "robust",
    eps=1e-12,
    verbose = True
    ):
    """
    Cohort-standardize a 1D array of subject-level indices (e.g., serial bottleneck index).

    Parameters
   --
    idx : array-like, shape (N,)
        Cohort values to standardize (one per subject). NaNs are allowed and ignored in
        summary stats; corresponding outputs will be NaN in standardized arrays.
    methods : tuple of {'z','robust','rank'}, default ('z','robust','rank')
        - 'z'      : mean/SD z-score (sensitive to outliers).
        - 'robust' : median/MAD with IQR fallback (Gaussian consistency factors 1.4826, 1.349).
        - 'rank'   : ECDF mid-ranks in (0,1] and a centered version in [-1,1].
    eps : float, default 1e-12
        Numerical floor to avoid division by ~0.

    Returns
    -------
    bottleneck_vals: array, shape (N,) selected nromalization in normalize
    info : dict
        {
          'z'        : np.ndarray (N,)   # if requested
          'robust'   : np.ndarray (N,)   # if requested
          'rank01'   : np.ndarray (N,)   # if requested, ECDF mid-rank in (0,1]
          'rank_pm1' : np.ndarray (N,)   # if requested, 2*rank01-1 in [-1,1]
          'stats'    : {
              'mean'   : float,
              'std'    : float,
              'median' : float,
              'mad'    : float,   # raw MAD
              'iqr'    : float,   # 75th-25th
              'n'      : int,     # non-NaN count
          }
        }
    """
    x = np.asarray(idx, dtype=float)
    info = {}
    mask = np.isfinite(x)
    x_valid = x[mask]
    n = int(mask.sum())
    # summary stats (NaN-safe)
    mu   = float(np.nanmean(x_valid)) if n else np.nan
    sd   = float(np.nanstd(x_valid))  if n else np.nan
    med  = float(np.nanmedian(x_valid)) if n else np.nan
    mad  = float(np.nanmedian(np.abs(x_valid - med))) if n else np.nan
    q75, q25 = (np.nanpercentile(x_valid, 75), np.nanpercentile(x_valid, 25)) if n else (np.nan, np.nan)
    iqr  = float(q75 - q25) if n else np.nan
    info['stats'] = {'mean': mu, 'std': sd, 'median': med, 'mad': mad, 'iqr': iqr, 'n': n}
    # Prepare outputs with NaNs where input is NaN
    def _empty():
        arr = np.full_like(x, np.nan, dtype=float)
        return arr
    # z-score
    z = _empty()
    if n:
        z[mask] = (x_valid - mu) / (sd + eps)
    info['z'] = z
    # robust z (MAD with IQR fallback; gaussian consistency factors)
    r = _empty()
    if n:
        if (mad is not None) and (mad > eps):
            scale = 1.4826 * mad
        elif (iqr is not None) and (iqr > eps):
            scale = iqr / 1.349
        else:
            scale = 0.0
        r[mask] = (x_valid - med) / (scale + eps) if scale > 0 else (x_valid - med)
    info['robust'] = r
    # ECDF ranks
    rank01 = _empty()
    rank_pm1 = _empty()
    if n:
        # stable double argsort
        r = np.argsort(np.argsort(x_valid, kind='stable'), kind='stable').astype(float)
        u = (r + 0.5) / n  # mid-rank in (0,1]
        rank01[mask]  = u
        rank_pm1[mask] = 2.0*u - 1.0  # [-1,1]
    info['rank01'] = rank01
    info['rank_pm1'] = rank_pm1
    bottleneck_vals = info[normalize]
    if verbose == True:
        return bottleneck_vals, info
    else:
        return bottleneck_vals


