#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for metric calculations for PCSM
Dr. Drew E. Winters
Created on 9/16/2025
"""

#-------------------
# Loading packages
#-------------------
import numpy as np
import pandas as pd
import projlib


#-----------------
# Functions
#------------------
# Identifying nodes and states--------------------------------------------------------------
def pcsm_nodes_and_states(
        b_node,
        d_sp,
        gamma,
        min_t_per_mode=1,
        fdr_alpha=0.25,
        fdr_mask=None,
        selection_mode="scored",     # {"fdr","scored"}
        use_active_only=True,
        use_bnode_weight=False,
        min_evidence=1e-9,
        state_names=None,            # names of hmm states
        trial_labels=None,           # shape (T,) per-TR labels OR shape (n_onsets,) per-trial labels
        trial_types=None,            # subset to report; default = unique from trial_labels
        fir_bins=5,                  # FIR window length per onset
        trial_onsets=None,           # explicit onset indices (preferred if available)
        trial_strategy="onset",      # {"onset","all"} windowing strategy
        constrain_trial=True,        # constrain trial nodes to global state/transition nodes
        bnode_fdr_fn=None,           # Optional: inject FDR function to avoid import coupling
        verbose=True
    ):
    """
    Build processing-mode node sets (unique “state” vs. shared “transition”)
    from FDR activity and map each selected node to its most-probable HMM state
    within each processing mode (parallel / mixed / serial). Optionally, derive
    **per-trial-type** node sets and HMM mappings that respect FIR-aligned windows.

    Overview
    --------
    1) Modes from D_SP thresholds (finite rows only):
         parallel: d_sp < tau_dsp_lo
         mixed   : tau_dsp_lo < d_sp < tau_dsp_hi
         serial  : d_sp > tau_dsp_hi
    2) FDR over b_node marks activity (timexnode). Per mode, nodes are flagged:
         • unique   (active in the mode; inactive in the other modes)
         • shared   (active in the mode and ≥ 1 other mode)
    3) Selection:
         selection_mode="fdr"    → raw FDR sets (sensitive).
         selection_mode="scored" → conservative subsets using active-TR counts (never adds nodes).
    4) HMM mapping (mode-level):
         For each selected nodexmode, aggregate posteriors over that mode's TRs
         (optionally restricted to FDR-active TRs and/or weighted by b_node) to
         compute per-state probabilities, best state, and stability.
    5) (Optional) Trial-conditioned mapping and sets (FIR-aware):
         If trial_labels are given, for each trial type u:
           • Build trial windows of length `fir_bins` per onset (strategy="onset"; ideal for 132x5=660),
             or use all TRs carrying that label (strategy="all").
           • Intersect each mode's TRs with the trial's window TRs.
           • Compute **unique (state)** vs **shared (transition)** across modes *within that trial window*,
             then (for states) assign each node to at most one trial type per mode (winner-take-all).
         This yields **trial-conditioned state/transition node sets** and their most-probable HMM states.

    Parameters
    ----------
    b_node : np.ndarray, shape (T, N)
        Node-level response probabilities in [0,1].
    d_sp : np.ndarray, shape (T,)
        Serial-parallel deviation series (aligned to b_node).
    gamma : np.ndarray, shape (T, C)
        HMM posteriors y_t(c) (aligned to b_node).
    min_t_per_mode : int, default 1
        Minimum TRs required for a mode to be summarized; otherwise empty.
    fdr_alpha : float, default 0.25
        Alpha for FDR when computing the activity mask (if fdr_mask is None).
    fdr_mask : np.ndarray or None, shape (T, N), default None
        Optional precomputed boolean activity mask. If None, computed via bnode_fdr_fn(..., alpha).
    selection_mode : {"fdr","scored"}, default "scored"
        Node-set selection policy (see Overview).
    use_active_only : bool, default True
        Accumulate HMM evidence only on TRs where the node is FDR-active in the mode.
    use_bnode_weight : bool, default False
        Weight y_t(c) by b_node[t, n] when summing evidence per node.
    min_evidence : float, default 1e-9
        If total evidence ≤ min_evidence, fall back to uniform state probs.
    state_names : list[str] or None
        Optional names for HMM states (length C). Defaults to ["s0","s1",...,].

    Trial conditioning
    ------------------
    trial_labels : array_like or None
        Either per-TR labels with length T, or per-trial labels with length == len(trial_onsets).
    trial_types : list/array or None, default None
        Subset of trial types to report; default = unique values from trial_labels.
    fir_bins : int, default 5
        FIR window length per onset. For your 660 TRs and 132 onsets, use 5 (132x5=660).
    trial_onsets : array_like or None
        Explicit onset indices (TR indices). Required when trial_labels are per-trial (len != T).
        If trial_labels are per-TR and onsets not provided, they are inferred from label changes.
    trial_strategy : {"onset","all"}, default "onset"
        "onset": windows [t0, t0+fir_bins) per onset (FIR-style). Ideal for non-overlapping 132x5 layouts.
        "all"  : use all TRs whose label equals the trial type (per-TR labels), or
                 the union of windows for all onsets of that type (per-trial labels).
    bnode_fdr_fn : callable or None
        Optional function with signature like: bnode_fdr_fn(b_node, alpha, return_mask=True) → (..., mask).

    Returns
    -------
    [unchanged; see original]
    """
    tau_dsp_hi = 0.61220267
    tau_dsp_lo = 0.52233887
    #  Validate & coerce 
    b_node = np.asarray(b_node, float)
    d_sp   = np.asarray(d_sp,   float)
    gamma  = np.asarray(gamma,  float)
    if b_node.ndim != 2: raise ValueError("b_node must be 2D (T,N).")
    if gamma.ndim  != 2: raise ValueError("gamma must be 2D (T,C).")
    T, N = b_node.shape
    Tg, C = gamma.shape
    if d_sp.shape[0] != T:  raise ValueError("d_sp length must equal T.")
    if Tg != T:             raise ValueError("gamma rows must match b_node rows.")
    # FDR mask
    if fdr_mask is None:
        if bnode_fdr_fn is None:
            try:
                from projlib.metric_calculation import bnode_fdr as _bnode_fdr
                bnode_fdr_fn = _bnode_fdr
            except Exception as e:
                raise RuntimeError("Provide fdr_mask or bnode_fdr_fn; auto-import failed.") from e
        *_unused, msk = bnode_fdr_fn(b_node, alpha=fdr_alpha, return_mask=True)
    else:
        msk = np.asarray(fdr_mask, bool)
    if msk.shape != (T, N):
        raise ValueError("fdr_mask must match (T,N).")
    # Filter non-finite d_sp (rare)
    finite_t = np.isfinite(d_sp)
    if not finite_t.all():
        b_node = b_node[finite_t]
        gamma  = gamma[finite_t]
        d_use  = d_sp[finite_t]
        msk    = msk[finite_t]
    else:
        d_use  = d_sp
    T_use = d_use.shape[0]
    if T_use == 0:
        return "error - d_sp input invalid"
    #  Mode indices from thresholds 
    idx_serial   = np.where(d_use > tau_dsp_hi)[0]
    idx_parallel = np.where(d_use < tau_dsp_lo)[0]
    idx_mixed    = np.where((d_use > tau_dsp_lo) & (d_use < tau_dsp_hi))[0]
    if idx_serial.size   < min_t_per_mode:   idx_serial   = np.array([], int)
    if idx_parallel.size < min_t_per_mode:   idx_parallel = np.array([], int)
    if idx_mixed.size    < min_t_per_mode:   idx_mixed    = np.array([], int)
    n_serial, n_parallel, n_mixed = map(int, (idx_serial.size, idx_parallel.size, idx_mixed.size))
    #  Helpers 
    def _msked(t_idx, mask=msk):
        """Return boolean vector over nodes: active at least once in t_idx."""
        if t_idx.size == 0: return np.zeros(N, bool)
        return mask[t_idx].any(0)
    def _counts(t_idx, mask=msk):
        """Return int counts per node: # of active TRs in t_idx."""
        if t_idx.size == 0: return np.zeros(N, int)
        return mask[t_idx].sum(0).astype(int)
    def _prev(t_idx, mask=msk):
        """Return float prevalence per node: mean active across t_idx."""
        if t_idx.size == 0: return np.zeros(N, float)
        return mask[t_idx].mean(0)
    def _sorted_idx(score_like, mask_bool):
        idx = np.where(mask_bool)[0]
        return np.sort(idx) if idx.size else idx
    def _enforce_and_relax(gate, counts, k):
        m = gate & (counts >= k)
        if (not np.any(m)) and np.any(gate):
            m = gate & (counts >= 1)
        return m
    def _both_side_mask(gate, c_mode, c_other, k):
        m = gate & (c_mode >= k) & (c_other >= k)
        if (not np.any(m)) and np.any(gate):
            m = gate & (c_mode >= 1) & (c_other >= 1)
        return m
    def _per_node_probs(t_idx, nodes_arr):
        """Aggregate HMM evidence for nodes over TRs t_idx."""
        if nodes_arr is None or len(nodes_arr) == 0 or t_idx.size == 0:
            return (np.array([], int), np.array([], int),
                    np.array([], float), np.zeros((0, C), float))
        nodes = np.asarray(nodes_arr, int)
        K = nodes.shape[0]
        probs = np.zeros((K, C), float)
        for ii, n in enumerate(nodes):
            if use_active_only:
                active_t = msk[t_idx, n]
            else:
                active_t = np.ones(t_idx.size, bool)
            if not active_t.any():
                probs[ii, :] = 1.0 / C
                continue
            tt = t_idx[active_t]
            G = gamma[tt]
            if use_bnode_weight:
                w = np.clip(b_node[tt, n].astype(float), 0.0, None)
                if w.ndim == 1: w = w[:, None]
                e = (G * w).sum(0)
            else:
                e = G.sum(0)
            s = e.sum()
            probs[ii, :] = (e / s) if s > min_evidence else (1.0 / C)
        best = np.argmax(probs, 1).astype(int)
        stab = probs[np.arange(probs.shape[0]), best]
        return nodes, best, stab, probs
    #  Build FDR unique/shared sets (mode-level) 
    serial_nodes   = _msked(idx_serial)
    mixed_nodes    = _msked(idx_mixed)
    parallel_nodes = _msked(idx_parallel)
    pn_o = np.where(parallel_nodes & (~serial_nodes) & (~mixed_nodes))[0]
    sn_o = np.where(serial_nodes   & (~parallel_nodes) & (~mixed_nodes))[0]
    mn_o = np.where(mixed_nodes    & (~serial_nodes)   & (~parallel_nodes))[0]
    pn_s = np.where(parallel_nodes & (serial_nodes | mixed_nodes))[0]
    sn_s = np.where(serial_nodes   & (parallel_nodes | mixed_nodes))[0]
    mn_s = np.where(mixed_nodes    & (serial_nodes   | parallel_nodes))[0]
    pn_o, sn_o, mn_o = map(np.sort, (pn_o, sn_o, mn_o))
    pn_s, sn_s, mn_s = map(np.sort, (pn_s, sn_s, mn_s))
    # General transition nodes (present in all three transition sets)
    gen_s = np.intersect1d(np.intersect1d(pn_s, sn_s, assume_unique=True), mn_s, assume_unique=True)
    # Prevalence and bias indices
    serial_prev   = _prev(idx_serial)
    mixed_prev    = _prev(idx_mixed)
    parallel_prev = _prev(idx_parallel)
    prevalence = {
        'p_parallel': parallel_prev,
        'p_mixed'   : mixed_prev,
        'p_serial'  : serial_prev,
        'n_parallel': n_parallel,
        'n_mixed'   : n_mixed,
        'n_serial'  : n_serial
    }
    U_parallel = parallel_prev - np.maximum(serial_prev,  mixed_prev)
    U_serial   = serial_prev   - np.maximum(parallel_prev, mixed_prev)
    U_mixed    = mixed_prev    - np.maximum(parallel_prev, serial_prev)
    # Selection policy
    if selection_mode == "fdr":
        state_sets = {'parallel': pn_o, 'serial': sn_o, 'mixed': mn_o}
        trans_sets = {'parallel': pn_s, 'serial': sn_s, 'mixed': mn_s}
    elif selection_mode == "scored":
        c_par = _counts(idx_parallel)
        c_ser = _counts(idx_serial)
        c_mix = _counts(idx_mixed)
        gate_state_par = np.zeros(N, bool); gate_state_par[pn_o] = True
        gate_state_ser = np.zeros(N, bool); gate_state_ser[sn_o] = True
        gate_state_mix = np.zeros(N, bool); gate_state_mix[mn_o] = True
        gate_trans_par = np.zeros(N, bool); gate_trans_par[pn_s] = True
        gate_trans_ser = np.zeros(N, bool); gate_trans_ser[sn_s] = True
        gate_trans_mix = np.zeros(N, bool); gate_trans_mix[mn_s] = True
        k_min_state = 2
        k_min_trans = 2
        elig_state_par = _enforce_and_relax(gate_state_par, c_par, k_min_state)
        elig_state_ser = _enforce_and_relax(gate_state_ser, c_ser, k_min_state)
        elig_state_mix = _enforce_and_relax(gate_state_mix, c_mix, k_min_state)
        best_other_cnt_par = np.maximum(c_ser, c_mix)
        best_other_cnt_ser = np.maximum(c_par, c_mix)
        best_other_cnt_mix = np.maximum(c_par, c_ser)
        elig_trans_par = _both_side_mask(gate_trans_par, c_par, best_other_cnt_par, k_min_trans)
        elig_trans_ser = _both_side_mask(gate_trans_ser, c_ser, best_other_cnt_ser, k_min_trans)
        elig_trans_mix = _both_side_mask(gate_trans_mix, c_mix, best_other_cnt_mix, k_min_trans)
        state_sets = {
            'parallel': _sorted_idx(c_par, elig_state_par),
            'serial'  : _sorted_idx(c_ser, elig_state_ser),
            'mixed'   : _sorted_idx(c_mix, elig_state_mix)
        }
        trans_sets = {
            'parallel': _sorted_idx(np.minimum(c_par, best_other_cnt_par), elig_trans_par),
            'serial'  : _sorted_idx(np.minimum(c_ser, best_other_cnt_ser), elig_trans_ser),
            'mixed'   : _sorted_idx(np.minimum(c_mix, best_other_cnt_mix), elig_trans_mix)
        }
        gen_s = np.intersect1d(
            np.intersect1d(trans_sets['parallel'], trans_sets['serial'], assume_unique=True),
            trans_sets['mixed'], assume_unique=True
        )
    else:
        raise ValueError("selection_mode must be 'fdr' or 'scored'.")
    # HMM mapping (mode-level)
    states = state_names or [f"s{i}" for i in range(C)]
    modes  = ['serial', 'mixed', 'parallel']
    idx_by_mode = {'serial': idx_serial, 'mixed': idx_mixed, 'parallel': idx_parallel}
    per_mode_best_t = {
        m: (np.array([], int) if idx_by_mode[m].size == 0
            else np.argmax(gamma[idx_by_mode[m]], 1).astype(int))
        for m in modes
    }
    global_best_t = np.argmax(gamma, 1).astype(int)
    global_vals = {'labels': {'p_modes': modes, 'states': states},
                   'timepoint_state': {'global_mostprob_hmmstate': global_best_t}}
    node_idx_pmode = {}
    node_pmode_mostprob_hmmstate = {}
    node_detail = {}
    mode_detail = {}
    for cat, sets in [('proc_state', state_sets), ('proc_transition', trans_sets)]:
        node_idx_pmode[cat] = {}
        node_pmode_mostprob_hmmstate[cat] = {}
        node_detail[cat] = {}
        mode_detail[cat] = {}
        for m in modes:
            t_idx = idx_by_mode[m]
            nodes, best, stab, probs = _per_node_probs(t_idx, sets.get(m, None))
            node_idx_pmode[cat][m] = nodes
            node_pmode_mostprob_hmmstate[cat][m] = best
            node_detail[cat][m] = {'node_stability': stab, 'node_probs_hmmstate_all': probs}
            mode_detail[cat][m] = {'p_mode_t': t_idx, 'p_mode_t_mostprob_hmmstate': per_mode_best_t[m]}
    node_idx_pmode['proc_transition']['general'] = gen_s
    scores = {'selection_criteria': selection_mode,
              'prevalence': prevalence,
              'U_prevalence': {'U_parallel': U_parallel, 'U_mixed': U_mixed, 'U_serial': U_serial}}
    #  Trial conditioning (optional) 
    trial_node_idx_pmode = None
    trial_node_pmode_mostprob_hmmstate = None
    trial_node_detail = None
    trial_info = None
    if trial_labels is not None:
        L = np.asarray(trial_labels)
        # detect label granularity
        labels_per_TR = (L.shape[0] == T)
        if trial_onsets is None:
            if labels_per_TR:
                # infer onsets where label changes (and first valid)
                onsets = []
                prev = object()
                for t in range(T):
                    lt = L[t]
                    if lt is None or (isinstance(lt, float) and not np.isfinite(lt)):
                        prev = object()
                        continue
                    if lt != prev:
                        onsets.append(t)
                        prev = lt
                onsets = np.array(onsets, int)
            else:
                raise ValueError("trial_onsets required when trial_labels are per-trial (len != T).")
        else:
            onsets = np.asarray(trial_onsets, int)
        labels_per_trial = (L.shape[0] == onsets.shape[0]) and (not labels_per_TR)
        # Trial types to report
        if trial_types is None:
            if labels_per_TR:
                valid_mask = (L != None) if L.dtype == object else np.isfinite(L)
                uniq = np.unique(L[valid_mask])
                trials = list(uniq)
            elif labels_per_trial:
                trials = list(np.unique(L))
            else:
                trials = []
        else:
            trials = list(trial_types)
        # Build windows for each trial type
        def _windows_for_type(typ):
            if trial_strategy == "all":
                if labels_per_TR:
                    return np.where(L == typ)[0]
                else:
                    # union of windows for all onsets with this trial label
                    t_idx = []
                    for i, t0 in enumerate(onsets):
                        if t0 >= T: continue
                        if labels_per_trial and L[i] == typ:
                            t1 = min(T, t0 + fir_bins)
                            t_idx.extend(range(t0, t1))
                    if len(t_idx) == 0:
                        return np.zeros((0,), int)
                    return np.unique(np.asarray(t_idx, int))
            # "onset": grab fir_bins TRs starting at each onset whose label == typ
            t_idx = []
            if labels_per_TR:
                for t0 in onsets:
                    if t0 >= T: continue
                    if L[t0] == typ:
                        t1 = min(T, t0 + fir_bins)
                        t_idx.extend(range(t0, t1))
            else:
                for i, t0 in enumerate(onsets):
                    if t0 >= T: continue
                    if labels_per_trial and L[i] == typ:
                        t1 = min(T, t0 + fir_bins)
                        t_idx.extend(range(t0, t1))
            if len(t_idx) == 0:
                return np.zeros((0,), int)
            return np.unique(np.asarray(t_idx, int))
        # precompute trial windows
        trial_windows = {typ: _windows_for_type(typ) for typ in trials}
        # Validate the ideal layout (non-blocking)
        warnings = []
        if trial_strategy == "onset":
            cover = np.zeros(T, int)
            for t0 in onsets:
                t1 = min(T, t0 + fir_bins)
                if t0 < T:
                    cover[t0:t1] += 1
            overlap = np.any(cover > 1)
            uncovered = np.any(cover == 0)
            if overlap:
                warnings.append("Trial windows overlap; FIR windows not strictly non-overlapping.")
            if uncovered:
                warnings.append("Some timepoints are not covered by any trial window.")
            if (fir_bins * len(onsets) != T):
                warnings.append(f"fir_bins * n_onsets != T ({fir_bins} * {len(onsets)} != {T}).")
        # Per-(mode, trial_type) FDR sets and HMM mapping (with exclusivity)
        trial_node_idx_pmode = {'proc_state': {}, 'proc_transition': {}}
        trial_node_pmode_mostprob_hmmstate = {'proc_state': {}, 'proc_transition': {}}
        trial_node_detail = {'proc_state': {}, 'proc_transition': {}}
        idx_by_mode_full = {'serial': idx_serial, 'mixed': idx_mixed, 'parallel': idx_parallel}
        # winner trial type per node, per mode (exclusivity across trial types for STATE nodes)
        winner_typ_by_node = {}
        for m in modes:
            Cmat = np.zeros((N, len(trials)), dtype=int)
            for j, typ in enumerate(trials):
                t_trial = trial_windows[typ]
                t_idx_mt = np.intersect1d(idx_by_mode_full[m], t_trial, assume_unique=False)
                Cmat[:, j] = _counts(t_idx_mt)
            argmax_j = np.argmax(Cmat, axis=1)
            max_counts = Cmat[np.arange(N), argmax_j]
            winner = np.array([trials[j] for j in argmax_j], dtype=object)
            winner[max_counts == 0] = None
            winner_typ_by_node[m] = winner
        # build per-trial sets
        for typ in trials:
            t_trial = trial_windows[typ]
            # per-mode TR indices restricted to this trial window
            t_ser = np.intersect1d(idx_serial,   t_trial, assume_unique=False)
            t_mix = np.intersect1d(idx_mixed,    t_trial, assume_unique=False)
            t_par = np.intersect1d(idx_parallel, t_trial, assume_unique=False)
            # activity masks per mode within this trial window
            m_ser = _msked(t_ser)
            m_mix = _msked(t_mix)
            m_par = _msked(t_par)
            # unique/shared across modes within this trial
            sn_o_t = np.where(m_ser & (~m_par) & (~m_mix))[0]
            mn_o_t = np.where(m_mix & (~m_par) & (~m_ser))[0]
            pn_o_t = np.where(m_par & (~m_ser) & (~m_mix))[0]
            sn_s_t = np.where(m_ser & (m_par | m_mix))[0]
            mn_s_t = np.where(m_mix & (m_par | m_ser))[0]
            pn_s_t = np.where(m_par & (m_ser | m_mix))[0]
            sn_o_t, mn_o_t, pn_o_t = map(np.sort, (sn_o_t, mn_o_t, pn_o_t))
            sn_s_t, mn_s_t, pn_s_t = map(np.sort, (sn_s_t, mn_s_t, pn_s_t))
            # selection policy within this trial
            if selection_mode == "fdr":
                trial_states = {'serial': sn_o_t, 'mixed': mn_o_t, 'parallel': pn_o_t}
                trial_trans  = {'serial': sn_s_t, 'mixed': mn_s_t, 'parallel': pn_s_t}
            else:
                c_ser = _counts(t_ser)
                c_mix = _counts(t_mix)
                c_par = _counts(t_par)
                gate_state_ser = np.zeros(N, bool); gate_state_ser[sn_o_t] = True
                gate_state_mix = np.zeros(N, bool); gate_state_mix[mn_o_t] = True
                gate_state_par = np.zeros(N, bool); gate_state_par[pn_o_t] = True
                gate_trans_ser = np.zeros(N, bool); gate_trans_ser[sn_s_t] = True
                gate_trans_mix = np.zeros(N, bool); gate_trans_mix[mn_s_t] = True
                gate_trans_par = np.zeros(N, bool); gate_trans_par[pn_s_t] = True
                k_min_state = 2
                k_min_trans = 2
                elig_state_ser = _enforce_and_relax(gate_state_ser, c_ser, k_min_state)
                elig_state_mix = _enforce_and_relax(gate_state_mix, c_mix, k_min_state)
                elig_state_par = _enforce_and_relax(gate_state_par, c_par, k_min_state)
                best_other_cnt_ser = np.maximum(c_par, c_mix)
                best_other_cnt_mix = np.maximum(c_par, c_ser)
                best_other_cnt_par = np.maximum(c_ser, c_mix)
                elig_trans_ser = _both_side_mask(gate_trans_ser, c_ser, best_other_cnt_ser, k_min_trans)
                elig_trans_mix = _both_side_mask(gate_trans_mix, c_mix, best_other_cnt_mix, k_min_trans)
                elig_trans_par = _both_side_mask(gate_trans_par, c_par, best_other_cnt_par, k_min_trans)
                trial_states = {
                    'serial'  : _sorted_idx(c_ser, elig_state_ser),
                    'mixed'   : _sorted_idx(c_mix, elig_state_mix),
                    'parallel': _sorted_idx(c_par, elig_state_par)
                }
                trial_trans = {
                    'serial'  : _sorted_idx(np.minimum(c_ser, best_other_cnt_ser), elig_trans_ser),
                    'mixed'   : _sorted_idx(np.minimum(c_mix, best_other_cnt_mix), elig_trans_mix),
                    'parallel': _sorted_idx(np.minimum(c_par, best_other_cnt_par), elig_trans_par)
                }
            # constrain trial sets to global sets (hierarchical consistency)
            if constrain_trial:
                trial_states = {
                    'serial'  : np.intersect1d(trial_states['serial'],   state_sets['serial'],   assume_unique=True),
                    'mixed'   : np.intersect1d(trial_states['mixed'],    state_sets['mixed'],    assume_unique=True),
                    'parallel': np.intersect1d(trial_states['parallel'], state_sets['parallel'], assume_unique=True)
                }
                trial_trans = {
                    'serial'  : np.intersect1d(trial_trans['serial'],   trans_sets['serial'],   assume_unique=True),
                    'mixed'   : np.intersect1d(trial_trans['mixed'],    trans_sets['mixed'],    assume_unique=True),
                    'parallel': np.intersect1d(trial_trans['parallel'], trans_sets['parallel'], assume_unique=True)
                }
            # general transitions within this trial
            gen_nodes_trial = np.intersect1d(
                np.intersect1d(trial_trans['parallel'], trial_trans['serial'], assume_unique=True),
                trial_trans['mixed'], assume_unique=True
            )
            # fill dicts (state nodes get trial-type exclusivity)
            for cat, sets in [('proc_state', trial_states), ('proc_transition', trial_trans)]:
                if typ not in trial_node_idx_pmode.get(cat, {}):
                    trial_node_idx_pmode[cat][typ] = {}
                    trial_node_pmode_mostprob_hmmstate[cat][typ] = {}
                    trial_node_detail[cat][typ] = {}
                for m in modes:
                    if cat == 'proc_state':
                        node_set = sets[m]
                        if node_set is not None and node_set.size:
                            w = winner_typ_by_node[m]
                            keep = np.array([w[n] == typ for n in node_set], dtype=bool)
                            node_set = node_set[keep]
                    else:
                        node_set = sets[m]
                    t_idx_mt = np.intersect1d(idx_by_mode_full[m], t_trial, assume_unique=False)
                    nodes, best, stab, probs = _per_node_probs(t_idx_mt, node_set)
                    trial_node_idx_pmode[cat][typ][m] = nodes
                    trial_node_pmode_mostprob_hmmstate[cat][typ][m] = best
                    trial_node_detail[cat][typ][m] = {
                        'node_stability': stab,
                        'node_probs_hmmstate_all': probs
                    }
                if cat == 'proc_transition':
                    t_union = np.unique(np.concatenate([idx_by_mode_full[m] for m in modes]))
                    t_idx_mt = np.intersect1d(t_union, t_trial, assume_unique=False)
                    nodes, best, stab, probs = _per_node_probs(t_idx_mt, gen_nodes_trial)
                    trial_node_idx_pmode[cat][typ]['general'] = nodes
                    trial_node_pmode_mostprob_hmmstate[cat][typ]['general'] = best
                    trial_node_detail[cat][typ]['general'] = {
                        'node_stability': stab,
                        'node_probs_hmmstate_all': probs
                    }
        trial_info = {
            'onsets': np.asarray(onsets, int),
            'fir_bins': int(fir_bins),
            'strategy': trial_strategy,
            'warnings': warnings
        }
    # Returns 
    if not verbose and trial_labels is not None:
        return node_idx_pmode, node_pmode_mostprob_hmmstate, trial_node_idx_pmode, trial_node_pmode_mostprob_hmmstate
    if not verbose and trial_labels is None:
        return (node_idx_pmode, node_pmode_mostprob_hmmstate,
                node_detail, mode_detail, global_vals, scores)
    else:
        return (node_idx_pmode, node_pmode_mostprob_hmmstate,
                trial_node_idx_pmode, trial_node_pmode_mostprob_hmmstate,
                node_detail, mode_detail, trial_node_detail, trial_info,
                global_vals, scores)



# Helper describing states, nodes, trials ect-------------------------------------------
def profile_hmm_states(
    pi_t,
    d_sp,
    trial_onsets,
    trial_type,
    accuracy,
    rt,
    windows=None,
    decision_window_name="fir",
    min_trials_per_type=1,
    # optional network composition inputs
    bnode = None, 
    node_network_labels=None,    # shape (N_nodes,)
    # optional demand/“load” time series aligned to pi_t
    demand=None,                 # shape (T,)
    # names / formatting
    state_names=None,
    return_pandas=False,
    ):
    """
    One-stop probabilistic profiling of HMM states.

    For each state, returns:
      * Occupancy by window x trial_type (mean y).
      * Occupancy-weighted performance per trial_type: P(correct|state,trial_type) and mean RT.
      * Mode probabilities from d_SP labels (serial/mixed/parallel), overall and by trial_type.
      * (Optional) Network composition per state given node→network labels and state weights.
      * (Optional) Demand/load summaries aligned to trials, modes and states.
      * A concise per-state summary.

    Notes
    -----
    * Occupancy-weighted performance:
        p_correct = Sigma_i occ_i * acc_i / Sigma_i occ_i,   mean_rt = Sigma_i occ_i * rt_i / Sigma_i occ_i
      where occ_i is the mean y over the chosen decision window on trial i.
    * Mode probabilities use time-pooling of y within d_SP-defined mode buckets.
    * Network composition uses absolute node weights per network, normalized to sum to 1.
    * A NumPy-based rank fallback is used for Spearman correlations when pandas is unavailable.
    """
    tau_dsp_hi=0.61220267; tau_dsp_lo=0.52233887
    # basic checks
    pi_t = np.asarray(pi_t)
    d_sp = np.asarray(d_sp)
    on = np.asarray(trial_onsets)
    T, C = pi_t.shape
    if d_sp.shape[0] != T:
        raise ValueError("d_sp must have length T == pi_t.shape[0].")
    if on.ndim != 1 or (on < 0).any() or (on >= T).any():
        raise ValueError("trial_onsets must be 1D ints within [0, T).")
    trial_type = np.asarray(trial_type)
    acc = np.asarray(accuracy).astype(float)
    rt = np.asarray(rt).astype(float)
    if not (len(on) == len(trial_type) == len(acc) == len(rt)):
        raise ValueError("trial arrays (onsets, trial_type, accuracy, rt) must have same length.")
    if windows is None:
        windows = {decision_window_name: (0, 6)}
    if decision_window_name not in windows:
        raise ValueError("decision_window_name must be a key in `windows`.")
    state_names = state_names or [f"s{i}" for i in range(C)]
    # demand validation
    if demand is not None:
        demand = np.asarray(demand, float)
        if demand.shape[0] != T:
            raise ValueError("demand must have shape (T,) aligned to pi_t.")
    # helpers
    def _slice_rel(t0, w):
        a, b = w
        a0 = max(0, int(t0 + a))
        b0 = min(T, int(t0 + b))
        return None if b0 <= a0 else slice(a0, b0)
    # rank helper (Spearman) with NumPy fallback
    def _rankdata(x):
        if pd is not None:
            return pd.Series(x).rank(method='average').to_numpy()
        x = np.asarray(x)
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x), dtype=float)
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        csum = np.cumsum(counts)
        starts = csum - counts
        mean_ranks = (starts + csum - 1) / 2.0
        return mean_ranks[inv]
    # mode labels at each t
    mode_lbl = np.full(T, "mixed", dtype=object)
    mode_lbl[d_sp < tau_dsp_lo] = "parallel"
    mode_lbl[d_sp > tau_dsp_hi] = "serial"
    modes = ("serial", "mixed", "parallel")
    # occupancy by window x trial_type
    win_names = list(windows.keys())
    trial_types = np.unique(trial_type)
    occ = {wname: np.full((len(on), C), np.nan, float) for wname in win_names}
    for i, t0 in enumerate(on):
        for wname, w in windows.items():
            sl = _slice_rel(t0, w)
            if sl is not None:
                occ[wname][i] = np.nanmean(pi_t[sl], axis=0)
    # per-trial demand per window (if provided)
    dem_win = {wname: np.full(len(on), np.nan, float) for wname in win_names} if (demand is not None) else None
    if demand is not None:
        for i, t0 in enumerate(on):
            for wname, w in windows.items():
                sl = _slice_rel(t0, w)
                if sl is not None:
                    dem_win[wname][i] = float(np.nanmean(demand[sl]))
    # performance (occupancy-weighted) in decision window
    wdec = decision_window_name
    occ_dec = occ[wdec]  # (N_trials, C)
    p_correct = {tt: np.full(C, np.nan) for tt in trial_types}
    mean_rt = {tt: np.full(C, np.nan) for tt in trial_types}
    for tt in trial_types:
        idx = np.where(trial_type == tt)[0]
        if idx.size < min_trials_per_type:
            continue
        for c in range(C):
            w = occ_dec[idx, c]
            ok = np.isfinite(w) & np.isfinite(acc[idx])
            if np.any(ok):
                p_correct[tt][c] = np.sum(w[ok] * acc[idx][ok]) / (np.sum(w[ok]) + 1e-12)
            ok_rt = np.isfinite(w) & np.isfinite(rt[idx])
            if np.any(ok_rt):
                mean_rt[tt][c] = np.sum(w[ok_rt] * rt[idx][ok_rt]) / (np.sum(w[ok_rt]) + 1e-12)
    # occupancy means by window x trial_type
    occ_by_w_t = {wname: {tt: np.full(C, np.nan) for tt in trial_types} for wname in win_names}
    for wname in win_names:
        M = occ[wname]
        for tt in trial_types:
            idx = np.where(trial_type == tt)[0]
            if idx.size < min_trials_per_type:
                continue
            occ_by_w_t[wname][tt] = np.nanmean(M[idx], axis=0)
    # mode probabilities per state (overall)
    mode_idx = {m: np.where(mode_lbl == m)[0] for m in modes}
    mode_mass = np.zeros((C, 3), float)
    for j, m in enumerate(modes):
        idx = mode_idx[m]
        if idx.size:
            mode_mass[:, j] = np.nansum(pi_t[idx, :], axis=0)
    mode_probs = mode_mass / (np.sum(mode_mass, axis=1, keepdims=True) + 1e-12)
    # mode probabilities per state x trial_type using decision window occupancy
    mode_probs_by_trial = {tt: np.full((C, 3), np.nan, float) for tt in trial_types}
    # per-trial mode fractions inside the decision window
    trial_mode_frac = np.full((len(on), 3), np.nan, float)
    for i, t0 in enumerate(on):
        sl = _slice_rel(t0, windows[wdec])
        if sl is None:
            continue
        lbls = mode_lbl[sl]
        if lbls.size == 0:
            continue
        trial_mode_frac[i] = np.array([
            np.mean(lbls == "serial"),
            np.mean(lbls == "mixed"),
            np.mean(lbls == "parallel")
        ], dtype=float)
    for tt in trial_types:
        idx = np.where(trial_type == tt)[0]
        if idx.size < min_trials_per_type:
            continue
        for c in range(C):
            w = occ_dec[idx, c]
            valid = np.isfinite(w) & np.all(np.isfinite(trial_mode_frac[idx]), axis=1)
            if not np.any(valid):
                continue
            num = np.sum(w[valid, None] * trial_mode_frac[idx][valid], axis=0)
            den = np.sum(w[valid]) + 1e-12
            mode_probs_by_trial[tt][c] = num / den
    # demand summaries (optional)
    demand_by_w_t = None
    demand_by_mode = None
    demand_by_state_mode = None
    demand_perf = None
    if demand is not None:
        # demand by state x window x trial_type (occupancy-weighted)
        demand_by_w_t = {wname: {tt: np.full(C, np.nan, float) for tt in trial_types}
                         for wname in win_names}
        for wname in win_names:
            for tt in trial_types:
                idx = np.where(trial_type == tt)[0]
                if idx.size < min_trials_per_type:
                    continue
                dbar = dem_win[wname][idx]
                for c in range(C):
                    w = occ[wname][idx, c]
                    ok = np.isfinite(w) & np.isfinite(dbar)
                    if np.any(ok):
                        demand_by_w_t[wname][tt][c] = float(np.sum(w[ok] * dbar[ok]) / (np.sum(w[ok]) + 1e-12))
        # demand by mode (global) and by statexmode (occupancy-weighted)
        demand_by_mode = {}
        demand_by_state_mode = {m: np.full(C, np.nan, float) for m in modes}
        for m in modes:
            idx = mode_idx[m]
            demand_by_mode[m] = (float(np.nanmean(demand[idx])) if idx.size else np.nan)
        for j, m in enumerate(modes):
            idx = mode_idx[m]
            if idx.size == 0:
                continue
            for c in range(C):
                w = pi_t[idx, c]
                x = demand[idx]
                ok = np.isfinite(w) & np.isfinite(x)
                if np.any(ok) and np.nansum(w[ok]) > 0:
                    demand_by_state_mode[m][c] = float(np.sum(w[ok] * x[ok]) / (np.sum(w[ok]) + 1e-12))
        # performance coupling with demand (decision window)
        demand_perf = {tt: {c: {'mean_demand_correct': np.nan,
                                'mean_demand_error':   np.nan,
                                'delta_correct_minus_error': np.nan,
                                'rt_spearman_vs_demand': np.nan,
                                'rt_pearson_vs_demand':  np.nan}
                             for c in range(C)}
                       for tt in trial_types}
        dbar_dec = dem_win[wdec]
        for tt in trial_types:
            idx = np.where(trial_type == tt)[0]
            if idx.size < min_trials_per_type:
                continue
            a_tt = acc[idx]
            r_tt = rt[idx]
            d_tt = dbar_dec[idx]
            for c in range(C):
                w = occ_dec[idx, c]
                # mean demand on correct vs error trials (occupancy-weighted)
                okc = np.isfinite(w) & np.isfinite(d_tt) & (a_tt == 1)
                oke = np.isfinite(w) & np.isfinite(d_tt) & (a_tt == 0)
                m_c = (np.sum(w[okc] * d_tt[okc]) / (np.sum(w[okc]) + 1e-12)) if np.any(okc) else np.nan
                m_e = (np.sum(w[oke] * d_tt[oke]) / (np.sum(w[oke]) + 1e-12)) if np.any(oke) else np.nan
                demand_perf[tt][c]['mean_demand_correct'] = (float(m_c) if np.isfinite(m_c) else np.nan)
                demand_perf[tt][c]['mean_demand_error']   = (float(m_e) if np.isfinite(m_e) else np.nan)
                if np.isfinite(m_c) and np.isfinite(m_e):
                    demand_perf[tt][c]['delta_correct_minus_error'] = float(m_c - m_e)
                # weighted Pearson & Spearman with RT
                okrt = np.isfinite(w) & np.isfinite(d_tt) & np.isfinite(r_tt)
                if np.any(okrt):
                    ww = w[okrt]; x = d_tt[okrt]; y = r_tt[okrt]
                    xw = np.sum(ww * x) / (np.sum(ww) + 1e-12)
                    yw = np.sum(ww * y) / (np.sum(ww) + 1e-12)
                    cov = np.sum(ww * (x - xw) * (y - yw)) / (np.sum(ww) + 1e-12)
                    sx = np.sqrt(np.sum(ww * (x - xw) ** 2) / (np.sum(ww) + 1e-12))
                    sy = np.sqrt(np.sum(ww * (y - yw) ** 2) / (np.sum(ww) + 1e-12))
                    demand_perf[tt][c]['rt_pearson_vs_demand'] = float(cov / (sx * sy + 1e-12))
                    rx = _rankdata(x); ry = _rankdata(y)
                    rxw = np.sum(ww * rx) / (np.sum(ww) + 1e-12)
                    ryw = np.sum(ww * ry) / (np.sum(ww) + 1e-12)
                    covr = np.sum(ww * (rx - rxw) * (ry - ryw)) / (np.sum(ww) + 1e-12)
                    srx = np.sqrt(np.sum(ww * (rx - rxw) ** 2) / (np.sum(ww) + 1e-12))
                    sry = np.sqrt(np.sum(ww * (ry - ryw) ** 2) / (np.sum(ww) + 1e-12))
                    demand_perf[tt][c]['rt_spearman_vs_demand'] = float(covr / (srx * sry + 1e-12))
    # optional network composition
    net_comp = None
    if bnode is not None and node_network_labels is not None:
        def _derive_state_node_weights_from_mask(gamma, fdr_mask, min_evidence=1e-9):
            """
            gamma or pi_t: (T, C)
            fdr_mask: (T, N) boolean, True when node n is active at TR t
            Returns W: (N, C)
            """
            import numpy as np
            T, C = gamma.shape
            Tm, N = fdr_mask.shape
            if Tm != T: 
                raise ValueError("fdr_mask must have same T as gamma.")
            W = np.zeros((N, C), float)
            for n in range(N):
                idx = np.where(fdr_mask[:, n])[0]
                if idx.size == 0:
                    continue
                e = gamma[idx].sum(axis=0)
                s = e.sum()
                W[n] = (e / s) if s > min_evidence else (np.ones(C) / C)
            return W
        from projlib.metric_calculation import bnode_fdr as _bnode_fdr
        mask_bn = _bnode_fdr(bnode, return_mask=True)[3]
        state_node_weights = _derive_state_node_weights_from_mask(
            gamma=pi_t,
            fdr_mask = mask_bn
        )
        W = np.asarray(state_node_weights)  # (N_nodes, C)
        if W.ndim != 2 or W.shape[1] != C:
            raise ValueError("state_node_weights must be (N_nodes, C).")
        labs = np.asarray(node_network_labels)
        if labs.shape[0] != W.shape[0]:
            raise ValueError("node_network_labels must have length N_nodes.")
        nets = np.unique(labs)
        net_comp = {c: {} for c in range(C)}
        for c in range(C):
            weights_c = W[:, c]
            # absolute mass per network, then normalize to 1
            totals = np.array([np.nansum(np.abs(weights_c[labs == nw])) for nw in nets], float)
            shares = totals / (np.sum(totals) + 1e-12)
            net_comp[c] = {str(nw): float(sh) for nw, sh in zip(nets, shares)}
    # assemble per-state outputs
    states = {}
    for c in range(C):
        name = state_names[c]
        # occupancy dictionary per window x trial_type
        occ_dict = {w: {str(tt): (None if not np.any(np.isfinite(occ_by_w_t[w][tt]))
                                  else float(occ_by_w_t[w][tt][c]))
                        for tt in trial_types}
                    for w in win_names}
        # performance dict per trial_type
        perf_dict = {str(tt): {
                        'p_correct': (None if not np.isfinite(p_correct[tt][c]) else float(p_correct[tt][c])),
                        'mean_rt' : (None if not np.isfinite(mean_rt[tt][c]) else float(mean_rt[tt][c]))
                     } for tt in trial_types}
        # mode probs
        mp = {'serial': float(mode_probs[c, 0]),
              'mixed' : float(mode_probs[c, 1]),
              'parallel': float(mode_probs[c, 2])}
        mp_by_trial = {str(tt): None for tt in trial_types}
        for tt in trial_types:
            arr = mode_probs_by_trial[tt][c]
            if np.any(np.isfinite(arr)):
                mp_by_trial[str(tt)] = {'serial': float(arr[0]),
                                        'mixed' : float(arr[1]),
                                        'parallel': float(arr[2])}
        states[c] = {
            'name': name,
            'occupancy': occ_dict,
            'performance': perf_dict,
            'mode_probs': mp,
            'mode_probs_by_trial': mp_by_trial,
            'network_composition': (net_comp[c] if net_comp is not None else None)
        }
    # concise summary
    summary = {}
    for c in range(C):
        # top trial types by mean occupancy in decision window
        dv = {tt: (None if not np.any(np.isfinite(occ_by_w_t[wdec][tt]))
                   else float(occ_by_w_t[wdec][tt][c]))
              for tt in trial_types}
        tops = [k for k, v in sorted(dv.items(), key=lambda kv: (-1 if kv[1] is None else -kv[1], kv[0]))]
        # overall perf (occupancy-weighted across all trials)
        w = occ_dec[:, c]
        ok_acc_all = np.isfinite(w) & np.isfinite(acc)
        p_all = (np.sum(w[ok_acc_all] * acc[ok_acc_all]) / (np.sum(w[ok_acc_all]) + 1e-12)) if np.any(ok_acc_all) else None
        ok_rt_all = np.isfinite(w) & np.isfinite(rt)
        r_all = (np.sum(w[ok_rt_all] * rt[ok_rt_all]) / (np.sum(w[ok_rt_all]) + 1e-12)) if np.any(ok_rt_all) else None
        # mode bias
        mp = states[c]['mode_probs']
        vals = list(mp.values())
        mb = max(mp, key=mp.get) if (max(vals) - min(vals)) > 0.05 else "flat"
        summary[c] = {
            'name': states[c]['name'],
            'top_trial_types_by_occupancy': tops[:3],
            'mode_bias': mb,
            'p_correct_overall': (None if p_all is None or not np.isfinite(p_all) else float(p_all)),
            'mean_rt_overall': (None if r_all is None or not np.isfinite(r_all) else float(r_all)),
        }
    out = {
        'states': states,
        'summary': summary,
        'meta': {
            'windows': {k: tuple(v) for k, v in windows.items()},
            'tau_dsp': {'lo': float(tau_dsp_lo), 'hi': float(tau_dsp_hi)},
            'decision_window': decision_window_name
        }
    }
    if demand is not None:
        out['demand'] = {
            'by_state_window_trial': {
                c: {w: {str(tt): (None if not np.any(np.isfinite(demand_by_w_t[w][tt]))
                                   else float(demand_by_w_t[w][tt][c]))
                        for tt in trial_types}
                    for w in win_names}
                for c in range(C)
            },
            'by_mode_global': {k: (None if (v is None or not np.isfinite(v)) else float(v))
                               for k, v in (demand_by_mode or {}).items()},
            'by_state_mode': (None if demand_by_state_mode is None else
                              {m: {c: (None if not np.isfinite(demand_by_state_mode[m][c]) else float(demand_by_state_mode[m][c]))
                                   for c in range(C)} for m in modes}),
            'performance': demand_perf
        }
    # optional tidy tables (built ONCE; demand tables included conditionally)
    if return_pandas and pd is not None:
        out_tables = {}
        # engagement
        rows = []
        for w in win_names:
            for tt in trial_types:
                vals = occ_by_w_t[w][tt]
                if not np.any(np.isfinite(vals)):
                    continue
                for c in range(C):
                    rows.append({'state_id': c, 'state': state_names[c], 'window': w,
                                 'trial_type': str(tt),
                                 'engagement_mean_pi_t': float(vals[c])})
        out_tables['engagement'] = pd.DataFrame(rows)
        # performance
        rows = []
        for tt in trial_types:
            for c in range(C):
                rows.append({'state_id': c, 'state': state_names[c], 'trial_type': str(tt),
                             'p_correct': (None if not np.isfinite(p_correct[tt][c]) else float(p_correct[tt][c])),
                             'mean_rt' : (None if not np.isfinite(mean_rt[tt][c]) else float(mean_rt[tt][c]))})
        out_tables['performance'] = pd.DataFrame(rows)
        # mode probs (overall)
        rows = []
        for c in range(C):
            for j, m in enumerate(modes):
                rows.append({'state_id': c, 'state': state_names[c], 'mode': m,
                             'prob': float(mode_probs[c, j])})
        out_tables['mode_probs'] = pd.DataFrame(rows)
        # mode probs by trial type
        rows = []
        for tt in trial_types:
            arr = mode_probs_by_trial[tt]
            if not np.any(np.isfinite(arr)):
                continue
            for c in range(C):
                rows.append({'state_id': c, 'state': state_names[c], 'trial_type': str(tt),
                             'serial': float(arr[c, 0]) if np.isfinite(arr[c, 0]) else np.nan,
                             'mixed' : float(arr[c, 1]) if np.isfinite(arr[c, 1]) else np.nan,
                             'parallel': float(arr[c, 2]) if np.isfinite(arr[c, 2]) else np.nan})
        out_tables['mode_probs_by_trial'] = pd.DataFrame(rows)
        # network composition
        if net_comp is not None:
            rows = []
            all_nets = sorted({lab for cdict in net_comp.values() for lab in cdict.keys()})
            for c in range(C):
                for nw in all_nets:
                    rows.append({'state_id': c, 'state': state_names[c], 'network': nw,
                                 'share': float(net_comp[c].get(nw, 0.0))})
            out_tables['network_composition'] = pd.DataFrame(rows)
        else:
            out_tables['network_composition'] = pd.DataFrame()
        # demand-related tables
        if demand is not None:
            rows = []
            for w in win_names:
                for tt in trial_types:
                    vec = demand_by_w_t[w][tt]
                    if not np.any(np.isfinite(vec)):
                        continue
                    for c in range(C):
                        rows.append({'state_id': c, 'state': state_names[c],
                                     'window': w, 'trial_type': str(tt),
                                     'mean_demand': float(vec[c])})
            out_tables['demand_by_state_window_trial'] = pd.DataFrame(rows)
            rows = []
            for m in modes:
                rows.append({'mode': m,
                             'mean_demand': (None if (demand_by_mode[m] is None or not np.isfinite(demand_by_mode[m]))
                                             else float(demand_by_mode[m]))})
            out_tables['demand_by_mode'] = pd.DataFrame(rows)
            rows = []
            for m in modes:
                for c in range(C):
                    val = demand_by_state_mode[m][c]
                    rows.append({'state_id': c, 'state': state_names[c], 'mode': m,
                                 'mean_demand': (np.nan if not np.isfinite(val) else float(val))})
            out_tables['demand_by_state_mode'] = pd.DataFrame(rows)
            rows = []
            for tt in trial_types:
                for c in range(C):
                    drow = demand_perf[tt][c]
                    rows.append({'state_id': c, 'state': state_names[c], 'trial_type': str(tt),
                                 **{k: (None if (v is None or not np.isfinite(v)) else float(v))
                                    for k, v in drow.items()}})
            out_tables['demand_performance'] = pd.DataFrame(rows)
        out['tables'] = out_tables
    return out



