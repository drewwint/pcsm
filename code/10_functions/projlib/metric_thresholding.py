
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for determining thrsholds for PCSM metrics
Dr. Drew E. Winters
Created on 9/16/2025
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import expit, logit
from scipy.stats import norm


# Functions for determining thrsholds____________________________________
## finding intersection between two Gaussians

def gmm2_threshold(values, random_state=123):
    """
    # Two-Gaussian mixture; return density-intersection threshold and a quality metric.
    """
    v = np.asarray(values).reshape(-1, 1)
    # Fit GMM
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=random_state)
    gmm.fit(v)
    w = gmm.weights_
    m = gmm.means_.ravel()
    s = np.sqrt(np.array([gmm.covariances_[i,0,0] for i in range(2)]))
    # order components by mean
    order = np.argsort(m)
    w, m, s = w[order], m[order], s[order]
    # quality metric: separation (Cohen's d-like)
      ## This metric just indicates how much distance is between distributions
        # Small separation (unstable)    <0.5
        # Moderate separation (reliable) ~1
        # Large separation (trustworthy) > 1.5
    sep = np.abs(m[1] - m[0]) / np.sqrt(0.5*(s[0]**2 + s[1]**2) + 1e-12)
    # Solve for density intersection: w1 N(x|m1,s1) = w2 N(x|m2,s2)
    a = 1/(2*s[1]**2) - 1/(2*s[0]**2)
    b = m[0]/(s[0]**2) - m[1]/(s[1]**2)
    c = (m[1]**2)/(2*s[1]**2) - (m[0]**2)/(2*s[0]**2) + np.log((w[1]*s[0])/(w[0]*s[1]+1e-12) + 1e-12)
    roots = np.roots([a, b, c]) if np.isfinite([a,b,c]).all() else np.array([np.nan])
    roots = np.sort(roots.real[np.isfinite(roots.real)])
    # pick root between means if available; otherwise closest to their midpoint
    if roots.size:
        candidates = roots[(roots > m[0]) & (roots < m[1])]
        tau = candidates[0] if candidates.size else roots[np.argmin(np.abs(roots - 0.5*(m[0]+m[1])))]
    else:
        tau = 0.5*(m[0] + m[1])
    # Clip to data range to be safe
    tau = float(np.clip(tau, np.min(values), np.max(values)))
    info = {"means": m.tolist(), "stds": s.tolist(), "weights": w.tolist(), "separation": float(sep)}
    return tau, info


## To establish bounds for far and equally weighted distributions
def tau_low_quantile(values, q=0.95, random_state=123):
    v = np.asarray(values).ravel()
    lo, hi = float(np.min(v)), float(np.max(v))
    g = GaussianMixture(n_components=2, covariance_type='full',
                        reg_covar=1e-6, random_state=random_state).fit(v.reshape(-1,1))
    m = g.means_.ravel(); s = np.sqrt(g.covariances_.ravel())
    m, s = m[np.argsort(m)], s[np.argsort(m)]   # m[0]=low mode
    tau = m[0] + s[0]*norm.ppf(q)               # q≈0.95 puts τ near top of low blob
    return float(np.clip(tau, lo, hi))


## finding intersection for three Gaussians
def Dmaha_high_threshold(values, p_star=None, sep_min=0.75, w_min=0.02,
                         random_state=123, support=(0.0, 1.0)):
    """
    High-tail cutoff for trimodal D^maha:
      - Fit 3-GMM on values strictly inside (0,1) to avoid edge spikes.
      - If p_star is None: use density intersection (middle vs high).
      - If p_star is set (e.g., 0.9): use a posterior target,
        i.e., P(high | x) >= p_star (more conservative).
      - If the fit is weak, fall back to a high quantile.
    Returns (tau, info)
    """
    v = np.asarray(values).ravel()
    lo, hi = float(np.min(v)), float(np.max(v))
    # drop exact edges (from min-max clipping)
    tol = 1e-8
    v_fit = v[(v > support[0] + tol) & (v < support[1] - tol)]
    if v_fit.size < 100:
        # not enough interior data; quantile fallback
        return float(np.quantile(v, 0.975)), {"method": "quantile_0.975_smallN"}
    # 3-GMM
    gmm = GaussianMixture(n_components=3, covariance_type='full',
                          reg_covar=1e-6, random_state=random_state).fit(v_fit.reshape(-1,1))
    m = gmm.means_.ravel()
    s = np.sqrt(gmm.covariances_.reshape(-1))
    w = gmm.weights_
    order = np.argsort(m)            # 0=low, 1=mid, 2=high
    m, s, w = m[order], s[order], w[order]
    # quality checks
    sep12 = abs(m[2]-m[1]) / np.sqrt(0.5*(s[2]**2 + s[1]**2) + 1e-12)
    if (sep12 < sep_min) or (w[2] < w_min):
        # weak or tiny high component → quantile fallback
        return float(np.quantile(v, 0.975)), {
            "method": "quantile_0.975_fallback",
            "sep12": float(sep12), "w_high": float(w[2])
        }
    # grid for safe in-support solutions
    xg = np.linspace(max(lo, m[1]-6*s[1]), min(hi, m[2]+6*s[2]), 4096)
    def pdf(i):  # weighted component density
        si = max(s[i], 1e-6)
        return w[i] * np.exp(-0.5*((xg - m[i])/si)**2) / (np.sqrt(2*np.pi)*si)
    f_mid, f_high = pdf(1), pdf(2)
    if p_star is None:
        # density intersection: f_mid(x) = f_high(x)
        d = f_mid - f_high
        sign = np.sign(d)
        idx = np.where(sign[:-1]*sign[1:] < 0)[0]
        if idx.size:
            # linearized zero-crossing
            i = idx[np.argmin(np.abs(xg[idx] - 0.5*(m[1]+m[2])))]
            x0,x1 = xg[i], xg[i+1]; y0,y1 = d[i], d[i+1]
            tau = x0 - y0*(x1-x0)/(y1-y0) if (y1-y0)!=0 else 0.5*(x0+x1)
            method = "gmm3_intersection_mid_high"
        else:
            tau = 0.5*(m[1]+m[2]); method = "midpoint_mid_high"
    else:
        # posterior target: P(high|x) >= p_star, comparing only mid vs high masses
        post_high = f_high / (f_high + f_mid + 1e-12)
        mask = post_high >= p_star
        if np.any(mask):
            tau = float(xg[np.argmax(mask)])  # first x meeting the target
            method = f"gmm3_posterior_target_{p_star}"
        else:
            tau = 0.5*(m[1]+m[2]); method = "posterior_target_fallback_midpoint"
    tau = float(np.clip(tau, lo, hi))
    return tau, {
        "method": method,
        "means": m.tolist(), "stds": s.tolist(), "weights": w.tolist(),
        "sep12": float(sep12)
    }


# Function for finding high/low thresholds for multi-gaussian densities (e.g.,Dsp)__________________________
## Fit a k-Gaussian mixture on logit(x) and return low/high cutpoints as
def multi_gmm_threshold(x, k=3, tail="both", random_state=0,
                            min_weight=0.10, min_sep=1, eps=1e-6):
    """
    Fit a k-Gaussian mixture on logit(x) and return low/high cutpoints as
    intersections between adjacent components, mapped back to native scale.

    x            : 1D array of metric for thresholding
    k            : number of components to try (use 2 or 3)
    tail         : 'both' | 'low' | 'high'  (what to return)
    min_weight   : ignore intersections involving a component with weight < this
    min_sep      : require Cohen-d-like separation between the two components
    returns      : (tau_low, tau_high, info) with taus possibly None if not reliable
    """
    x = np.asarray(x, float)
    x = np.clip(x, eps, 1.0 - eps)           # avoid inf on logit
    z = logit(x)[:, None]
    # Pick k by BIC over {2,3,4,5} to keep it simple
    ks = [2,3,4,5] if k not in (2,3,4,5) else [k]
    best = None
    for kk in ks:
        g = GaussianMixture(n_components=kk, covariance_type="full",
                            random_state=random_state).fit(z)
        bic = g.bic(z)
        if best is None or bic < best["bic"]:
            best = {"gmm": g, "bic": bic, "k": kk}
    gmm = best["gmm"]; k = best["k"]
    # Extract and sort components by mean (in logit space)
    m = gmm.means_.ravel()
    s = np.sqrt(gmm.covariances_.reshape(-1))
    w = gmm.weights_.ravel()
    order = np.argsort(m)
    m, s, w = m[order], s[order], w[order]
    def intersect(i, j):
        # Solve w_i N_i(z) = w_j N_j(z) for z
        a = 1/(2*s[j]**2) - 1/(2*s[i]**2)
        b = m[i]/(s[i]**2) - m[j]/(s[j]**2)
        c = (m[j]**2)/(2*s[j]**2) - (m[i]**2)/(2*s[i]**2) + np.log((w[j]*s[i])/(w[i]*s[j]) + eps)
        roots = np.roots([a, b, c]) if np.isfinite([a,b,c]).all() else np.array([])
        roots = np.sort(roots.real[np.isfinite(roots.real)])
        if roots.size:
            # pick root between means if available, else closest to midpoint
            cand = roots[(roots > min(m[i],m[j])) & (roots < max(m[i],m[j]))]
            z_tau = cand[0] if cand.size else roots[np.argmin(np.abs(roots - 0.5*(m[i]+m[j])))]
        else:
            z_tau = 0.5*(m[i] + m[j])
        return float(z_tau)
    # Adjacent intersections (in logit space) → back to [0,1]
    zcuts = [intersect(i, i+1) for i in range(k-1)]
    taus  = expit(np.array(zcuts))
    # Quality checks for each adjacent pair
    sep = np.abs(m[1:] - m[:-1]) / np.sqrt(0.5*(s[:-1]**2 + s[1:]**2) + eps)  # Cohen-d-like
    ok   = (w[:-1] >= min_weight) & (w[1:] >= min_weight) & (sep >= min_sep)
    tau_low  = float(taus[0]) if ok[0] else None
    tau_high = float(taus[-1]) if ok[-1] else None
    info = {
        "k": k, "means_z": m.tolist(), "stds_z": s.tolist(), "weights": w.tolist(),
        "cuts_z": list(map(float, zcuts)), "cuts_x": list(map(float, taus)),
        "separation": list(map(float, sep)), "ok_pairs": ok.tolist()
    }
    if tail == "low":
        return tau_low, None, info
    if tail == "high":
        return None, tau_high, info
    return tau_low, tau_high, info


# KDE-based thresholds on [0,1] with boundary correction_________________________
## Helpers for KDE
def _gaussian_kernel_bins(sigma_bins):
    # cover ±3σ; odd length for nice centering
    khalf = int(np.ceil(3.0 * sigma_bins))
    u = np.arange(-khalf, khalf + 1, dtype=float)
    K = np.exp(-0.5 * (u / sigma_bins)**2)
    K /= K.sum()
    return K

def _smooth_hist_reflect(counts, kernel):
    # reflect counts at both ends, convolve, take center back
    M = counts.size
    ref = np.concatenate([counts[::-1], counts, counts[::-1]])
    sm_ref = np.convolve(ref, kernel, mode="same")
    return sm_ref[M:2*M]

def _minima_indices(y):
    # local minima via sign change of derivative: - → +
    dy = np.diff(y)
    s = np.sign(dy)
    idx = np.where((s[:-1] < 0) & (s[1:] > 0))[0] + 1
    return idx


## Dsp thresholds via KDE
def Dsp_thresholds_kde_windows_binned(
    x,
    bins=4096,                # histogram resolution
    bandwidth=None,           # None → Silverman rule
    low_window=(0.51, 0.59),  # context window for low side
    high_window=(0.61, 0.69), # context window for high side
    low_anchor=None,          # default: low_window[1]
    high_anchor=None,         # default: high_window[0]
    tail="both"               # 'both' | 'low' | 'high'  
):
    """
    Memory-safe KDE valley thresholds on [0,1].
    - Low threshold: first local minimum >= low_anchor (fallback: nearest minimum)
    - High threshold: last local minimum <= high_anchor (fallback: nearest minimum)
    - tail: controls which thresholds are returned without changing computation.
    """
    x = np.asarray(x, float)
    eps = 1e-12
    x = np.clip(x, eps, 1.0 - eps)
    n = x.size
    # histogram on [0,1]
    counts, edges = np.histogram(x, bins=bins, range=(0.0, 1.0))
    grid = 0.5 * (edges[:-1] + edges[1:])  # bin centers
    dx = edges[1] - edges[0]
    # bandwidth: Silverman's rule-of-thumb (on original scale)
    if bandwidth is None:
        std = np.std(x)
        bandwidth = 1.06 * std * n**(-1/5)
        # keep a small floor so kernel isn't a delta
        bandwidth = max(bandwidth, 2.5 * dx)
    sigma_bins = max(bandwidth / dx, 1.0)  # in bin units; floor to 1 bin
    kernel = _gaussian_kernel_bins(sigma_bins)
    # KDE via smoothed histogram with reflection (boundary correction)
    smooth_counts = _smooth_hist_reflect(counts, kernel)
    # convert to density
    fx = smooth_counts / (n * dx)  # integrates ≈ 1 over [0,1]
    # all minima across [0,1]
    mins = _minima_indices(fx)
    # default anchors from windows
    if low_anchor  is None: low_anchor  = float(low_window[1])
    if high_anchor is None: high_anchor = float(high_window[0])
    # helper pickers
    def pick_low():
        cand = mins[grid[mins] >= low_anchor]
        if cand.size:
            return int(cand[0])
        return int(mins[np.argmin(np.abs(grid[mins] - low_anchor))]) if mins.size else None
    def pick_high():
        cand = mins[grid[mins] <= high_anchor]
        if cand.size:
            return int(cand[-1])
        return int(mins[np.argmin(np.abs(grid[mins] - high_anchor))]) if mins.size else None
    j_low  = pick_low()
    j_high = pick_high()
    tau_low  = float(grid[j_low])  if j_low  is not None else None
    tau_high = float(grid[j_high]) if j_high is not None else None
    # enforce ordering if both exist
    if (tau_low is not None) and (tau_high is not None) and (tau_low >= tau_high):
        tau_low = float(max(eps, tau_high - dx))  # nudge inside
    # unchanged computation; only the return shape depends on 'tail'
    if tail == "low":
      info = {
          "bins": int(bins),
          "bandwidth": float(bandwidth),
          "sigma_bins": float(sigma_bins),
          "low_window": list(map(float, low_window)),
          "low_anchor": float(low_anchor),
          "minima_x": grid[mins].tolist(),
          "minima_fx": fx[mins].tolist(),
          "chosen_low_idx": j_low,
          "tau_low": tau_low
          }
      return tau_low, None, info
    if tail == "high":
      info = {
          "bins": int(bins),
          "bandwidth": float(bandwidth),
          "sigma_bins": float(sigma_bins),
          "high_window": list(map(float, high_window)),
          "high_anchor": float(high_anchor),
          "minima_x": grid[mins].tolist(),
          "minima_fx": fx[mins].tolist(),
          "chosen_high_idx": j_high,
          "tau_high": tau_high
          }
      return None, tau_high, info
    if tail == "both":
      info = {
          "bins": int(bins),
          "bandwidth": float(bandwidth),
          "sigma_bins": float(sigma_bins),
          "low_window": list(map(float, low_window)),
          "high_window": list(map(float, high_window)),
          "low_anchor": float(low_anchor),
          "high_anchor": float(high_anchor),
          "minima_x": grid[mins].tolist(),
          "minima_fx": fx[mins].tolist(),
          "chosen_low_idx": j_low,
          "chosen_high_idx": j_high,
          "tau_low": tau_low,
          "tau_high": tau_high
          }
      return tau_low, tau_high, info

# Dsp quality checks_________________________
## Dsp sensitivity analysis
def Dsp_tau_sensitivity_binned(
    x,
    low_window=(0.45, 0.58),
    high_window=(0.58, 0.72),
    low_anchor=0.50,
    high_anchor=0.62,
    bins_list=(2048, 4096, 8192),
    bw_scales=(0.8, 1.0, 1.2),
    tail = "both"
):
    """
    Sweep #bins and bandwidth to assess tau stability.
    Returns (summary, rows) where rows is a list of dicts per (bins, bw_scale).
    """
    x = np.asarray(x, float)
    # First call to get a baseline bandwidth
    _, _, info0 = Dsp_thresholds_kde_windows_binned(
        x, bins=bins_list[0],
        low_window=low_window, high_window=high_window,
        low_anchor=low_anchor, high_anchor=high_anchor
    ) # type: ignore
    bw0 = info0["bandwidth"]
    rows = []
    for bins in bins_list:
        for s in bw_scales:
            tau_lo, tau_hi, info = Dsp_thresholds_kde_windows_binned(
                x, bins=bins, bandwidth=bw0 * s,
                low_window=low_window, high_window=high_window,
                low_anchor=low_anchor, high_anchor=high_anchor
            ) # type: ignore
            rows.append({
                "bins": int(bins),
                "bw": float(bw0 * s),
                "scale": float(s),
                "tau_low": None if tau_lo is None else float(tau_lo),
                "tau_high": None if tau_hi is None else float(tau_hi)
            })
    # summarize (ignore None)
    def _summ(vals):
        vals = np.array([v for v in vals if v is not None], float)
        if vals.size == 0:
            return {"mean": None, "std": None, "min": None, "max": None, "drift": None}
        return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                "min": float(vals.min()), "max": float(vals.max()), "drift": float(vals.max()-vals.min())}
    if tail == "low":
      summary = {
          "low":  _summ([r["tau_low"]  for r in rows])
      }
      summary["baseline_bandwidth"] = float(bw0)
      return summary, rows
    if tail == "high":
      summary = {
          "high": _summ([r["tau_high"] for r in rows])
      }
      summary["baseline_bandwidth"] = float(bw0)
      return summary, rows
    if tail == "both":
      summary = {
          "low":  _summ([r["tau_low"]  for r in rows]),
          "high": _summ([r["tau_high"] for r in rows])
      }
    summary["baseline_bandwidth"] = float(bw0)
    return summary, rows


## Dsp Quality check of thresholds
def Dsp_threshold_quality(x, tau_low=None, tau_high=None, *, bins=4096, bandwidth=None):
    """
    Returns quality metrics for each available cut:
      - robust separation d_robust = |median_R - median_L| / sqrt(0.5*(mad_L^2+mad_R^2))
      - mass proportions on each side
      - KDE 'prominence' at the valley: min(peak_left, peak_right) / valley
    """
    # Helper functions
    ## KDE with reflection at 0 and 1
    def _hist_smooth_reflect(x, bins=4096, bandwidth=None):
        x = np.asarray(x, float)
        eps = 1e-12
        x = np.clip(x, eps, 1.0 - eps)
        n = x.size
        counts, edges = np.histogram(x, bins=bins, range=(0.0,1.0))
        grid = 0.5*(edges[:-1] + edges[1:])
        dx = edges[1] - edges[0]
        # bandwidth on original scale (Silverman) with small floor
        if bandwidth is None:
            std = np.std(x)
            bandwidth = max(1.06 * std * n**(-1/5), 2.5*dx)
        sigma_bins = max(bandwidth/dx, 1.0)
        khalf = int(np.ceil(3.0*sigma_bins))
        u = np.arange(-khalf, khalf+1, dtype=float)
        K = np.exp(-0.5*(u/sigma_bins)**2); K /= K.sum()
        ref = np.concatenate([counts[::-1], counts, counts[::-1]])
        sm = np.convolve(ref, K, mode="same")[bins:2*bins]
        fx = sm / (n*dx)
        return grid, fx, bandwidth, dx
    ## local minima and maxima
    def _minima_indices(y):
        dy = np.diff(y); s = np.sign(dy)
        return np.where((s[:-1] < 0) & (s[1:] > 0))[0] + 1
    ## local minima and maxima
    def _maxima_indices(y):
        dy = np.diff(y); s = np.sign(dy)
        return np.where((s[:-1] > 0) & (s[1:] < 0))[0] + 1
    ## robust std estimate via MAD
    def _mad_std(x):
        # robust sigma estimate: 1.4826 * MAD
        m = np.median(x)
        return 1.4826 * np.median(np.abs(x - m))
    # Main
    x = np.asarray(x, float)
    grid, fx, bw, dx = _hist_smooth_reflect(x, bins=bins, bandwidth=bandwidth)
    mins = _minima_indices(fx); maxs = _maxima_indices(fx)
    def _side_stats(cut):
        if cut is None: return None
        # robust separation on the raw data
        left  = x[x <= cut]; right = x[x > cut]
        if left.size < 10 or right.size < 10:
            sep = None
        else:
            mL, mR = np.median(left), np.median(right)
            sL, sR = _mad_std(left), _mad_std(right)
            sep = float(abs(mR - mL) / np.sqrt(0.5*(sL**2 + sR**2) + 1e-12))
        mass_L = float((x <= cut).mean())
        mass_R = 1.0 - mass_L
        # valley prominence from KDE
        j = int(np.argmin(np.abs(grid - cut)))
        # nearest peak to the left and right
        left_peaks  = maxs[maxs < j]
        right_peaks = maxs[maxs > j]
        peakL = fx[left_peaks[-1]] if left_peaks.size else np.nan
        peakR = fx[right_peaks[0]] if right_peaks.size else np.nan
        valley = fx[j]
        prominence = float(min(peakL, peakR) / valley) if np.isfinite(peakL) and np.isfinite(peakR) and valley>0 else None
        return {
            "threshold": float(cut),
            "mass_left": mass_L, "mass_right": mass_R,
            "sep_robust": sep,
            "valley_density": float(valley),
            "peak_left": None if not np.isfinite(peakL) else float(peakL),
            "peak_right": None if not np.isfinite(peakR) else float(peakR),
            "prominence_min_over_valley": prominence,
            "bandwidth": float(bw),
            "bin_width": float(dx)
        }
    return {
        "low":  _side_stats(tau_low),
        "high": _side_stats(tau_high)
    }


# Function for finding adaptive FDR thresholds for B_node__________________________

#> “Node-wise responding maps were obtained by adaptive FDR control on posterior probabilities q_n,t. 
#> We used a nominal FDR of α = 0.25 to balance sensitivity and precision in a high-dimensional setting; 
# downstream decisions remained robust because 
  #> (i) Dsp operates on the raw probabilities, 
  # (ii) trial calls were gated by rho and Pt, and 
  # (iii) we report the realized FDR and perform sensitivity analyses over α ∈ {0.10–0.30}.”
                                                                                                                                                                                                                                                                                                                                                                                                                           
def bnode_fdr(B_node, alpha=0.25, return_mask=False):
    """
    Adaptive FDR on a (T, N) matrix of probabilities.
    Returns:
      tau  : (T,) per-timepoint thresholds
      k    : (T,) number of nodes called responding
      fdr  : (T,) realized expected FDR
      mask : (T, N) optional boolean selection (if return_mask=True)
    """
    Q = np.asarray(B_node, float)
    T, N = Q.shape
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
    k_star = ok.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1)
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


# Function for finding percentile based thresholds


def perc_threshold(
        val_list,
        q = (0.25,0.5,0.75),
        alpha = 0.05,
        n_boot=2000,
        random_state = 5743
        ):
    """
    Function for quantile based thresholding perfomring the following
    1) extract quantiles for all subjects in a list individually
    2) identify the high and low quantiles across subjects
    3) bootstrap parameters around qunatiles for diagnostics
    
    Returns
    -------
    dist_q: (array) the emperial quantile distribution
    thresholds: (dict) the high/low quantile based thresholds
    threshold_ci: (dict) the conficence interval for thresholds
    """
    # helper functions
    def _subject_quantiles(arr_list, q=q):
        q_lo, q_md, q_hi = [], [], []
        for a in arr_list:
            x = np.asarray(a)
            q_lo.append(np.percentile(x, 100*q[0]))
            q_md.append(np.percentile(x, 100*q[1]))
            q_hi.append(np.percentile(x, 100*q[2]))
        return np.array(q_lo), np.array(q_md), np.array(q_hi)
    def _group_thresholds(q_lo, q_hi, alpha=alpha, n_boot=n_boot, random_state=random_state):
        rng = np.random.RandomState(random_state)
        # Empirical group thresholds: low/high in q list, by default: 25th of q_lo and 75th of q_hi
        tau_low  = np.percentile(q_lo, (np.min(q)*100))
        tau_high = np.percentile(q_hi, (np.max(q)*100))
        # Bootstrap CIs across subjects
        boots_low, boots_high = [], []
        n = len(q_lo)
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            boots_low.append(np.percentile(q_lo[idx], (np.min(q)*100)))
            boots_high.append(np.percentile(q_hi[idx], (np.max(q)*100)))
        ci_low  = np.percentile(boots_low, [100*alpha/2, 100*(1-alpha/2)])
        ci_high = np.percentile(boots_high,[100*alpha/2, 100*(1-alpha/2)])
        return (tau_low, ci_low), (tau_high, ci_high)
    q_lo, q_md, q_hi = _subject_quantiles(val_list)
    (q_tau_low, q_ci_low), (q_tau_high, q_ci_high) = _group_thresholds(q_lo, q_hi)
    dist_q = np.concatenate([q_lo, q_md, q_hi])
    thresholds = {
        "tau_high":q_tau_high,
        "tau_low":q_tau_low
    }
    threshold_ci = {
        "tau_low_ci95": q_ci_low,
        "tau_high_ci95":q_ci_high
    }
    return dist_q, thresholds, threshold_ci
    

