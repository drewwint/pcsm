#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import nilearn
import nilearn.plotting
import nilearn.datasets
import nilearn.image
import nilearn.maskers

atlas_loc = "/home/dwinters/ascend-lab/resources/atlases"

# Plotting marked timeperiods-------------------------------------------------------
def plot_subject_fir(subject_idx,
                     bold_v,
                     events_v,
                     nodes=None,
                     trial_type_colors=None,
                     figsize=(20, 4),
                     legend_show=True,
                     legend_loc="best",
                     legend_anchor = None,
                     label_nodes=False,
                     flagged_idx=None,
                     flagged_label="Flagged",
                     flagged_kwargs=None,
                     title = None,
                     save_loc = None, # os.path.join(base,"name.tiff")
                     show=True):
    """
    Plot FIR responses for selected nodes for one subject with trial-type spans
    and optional vertical markers at specified concatenated timepoints.

    flagged_idx : array-like or None
        Indices along the concatenated x-axis (0..n_trials*n_bins-1)
        to mark with vertical black lines.
    flagged_label : str
        Label used in the legend for the flagged timepoints.
    flagged_kwargs : dict or None
        Extra kwargs passed to ax.vlines for the flagged markers.
        Defaults are sensible (black, lw=1.5, alpha=0.9).
    show : bool
        If True, show the figure. If you want to add more annotations
        after the call, set False and call plt.show() later.
    """
    if nodes is None:
        nodes = np.arange(15)
    else:
        nodes = np.asarray(nodes)
    if trial_type_colors is None:
        trial_type_colors = {"Go": "olivedrab", "Stop": "orangered"}
    if flagged_kwargs is None:
        flagged_kwargs = dict(color="black", linewidth=1.5, alpha=0.9)
    # Extract subject data
    subject_fir = bold_v[subject_idx]  # (n_trials, n_bins, n_parcels)
    subject_labels = events_v[subject_idx].trial_type.reset_index(drop=True)
    if subject_fir.ndim != 3:
        raise ValueError("Expected (n_trials, n_bins, n_parcels).")
    n_trials, n_bins, n_parcels = subject_fir.shape
    if subject_labels.shape[0] != n_trials:
        raise ValueError(
            f"Labels vs trials mismatch: {subject_labels.shape[0]} vs {n_trials}"
        )
    # Concatenate trials -> (n_trials*n_bins, n_parcels)
    all_trials = subject_fir.reshape(n_trials * n_bins, n_parcels)
    selected = all_trials[:, nodes]
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(selected.shape[0])
    for i in range(selected.shape[1]):
        lbl = f"Node {nodes[i]}" if label_nodes else None
        ax.plot(x, selected[:, i], label=lbl)
    # Background spans per trial
    current_onset = 0
    span_handles = []
    span_labels = []
    for t in range(n_trials):
        trial_label = subject_labels.iloc[t]
        color = trial_type_colors.get(trial_label)
        if color is None:
            color = trial_type_colors.get(str(trial_label).capitalize(), "white")
        span = ax.axvspan(current_onset,
                          current_onset + n_bins,
                          facecolor=color,
                          alpha=0.25,
                          label=trial_label)
        if trial_label not in span_labels:
            span_handles.append(span)
            span_labels.append(trial_label)
        current_onset += n_bins
    # Optional flagged markers
    flagged_handle = None
    if flagged_idx is not None and len(flagged_idx) > 0:
        flagged_idx = np.asarray(flagged_idx, dtype=int)
        # keep within bounds
        flagged_idx = flagged_idx[(flagged_idx >= 0) & (flagged_idx < x.size)]
        if flagged_idx.size > 0:
            # Draw vertical lines across current y-lims
            y_min, y_max = ax.get_ylim()
            ax.vlines(flagged_idx, y_min, y_max, **flagged_kwargs)
            # Create a single legend handle representing these lines
            flagged_handle = Line2D([0], [0],
                                    color=flagged_kwargs.get("color", "black"),
                                    linewidth=flagged_kwargs.get("linewidth", 1.5),
                                    alpha=flagged_kwargs.get("alpha", 0.9),
                                    label=flagged_label)
    # Labels & title
    ax.set_xlabel("Concatenated Timepoints (FIR Bins)")
    ax.set_ylabel("FIR Response (Effect Size)")
    if title == None:
        ax.set_title(
            f"Subject {subject_idx + 1} – Individual Node FIR Responses Across All Trials"
        )
    if title != None:
        ax.set_title(
            f"Subject {subject_idx + 1} – {str(title)}"
        )
    # Legend assembly
    handles = []
    labels = []
    if label_nodes:
        line_handles, line_labels = ax.get_legend_handles_labels()
        for h, l in zip(line_handles, line_labels):
            if isinstance(l, str) and l.startswith("Node"):
                handles.append(h)
                labels.append(l)
    handles.extend(span_handles)
    labels.extend(span_labels)
    if flagged_handle is not None:
        handles.append(flagged_handle)
        labels.append(flagged_label)
    if legend_show and handles:
        ax.legend(handles, labels, bbox_to_anchor= legend_anchor, loc=legend_loc)
    ax.grid(True)
    if save_loc != None:
        plt.savefig(save_loc, dpi=400)
    if show:
        plt.show()
    return fig, ax


# plotting timeseries with trial backgrond color -------------------------------------

def plot_timeseries_with_trials(
    series,
    trial_labels,
    fir_bins=5,
    onsets_bins=None,
    durations_bins=None,
    trial_colors=None,          # dict like {'Go':'olivedrab','Stop':'orangered'}
    trial_alpha=0.20,
    stack=False,                # False=overlay; True=stack vertically
    line_kwargs=None,           # dict per series-name OR default kwargs
    include_trial_labels=True,  # include shaded trial labels in legend
    include_line_labels=True,   # include line labels in legend
    line_labels=None,           # dict or list of custom labels for lines
    show_legend=True,           # master legend toggle
    legend_loc="best",
    legend_kwargs=None,         # extra kwargs for legend()
    figsize=(20, 4),
    xlabel="Time (bins)",
    ylabel="Value",
    hlines=None,
    hline_kwargs=None,
    title=None,
    save_loc=None,
    dpi=400,
    show=True
):
    """
    Plot one or multiple time series aligned to trials, with shaded trial backgrounds.

    Inputs
    ------
    series : 1D array | list[1D array] | dict[str, 1D array]
        The time series to plot (same length T).
        - If list, names auto-assigned: s0, s1, ...
        - If dict, keys used as labels.
    trial_labels : array-like of length n_trials
        Label per trial (any strings, e.g., 'Go','Stop','Wait', ...).

    Parameters
    ----------
    fir_bins : int, default 5
        If provided (and onsets/durations omitted), trials are assumed back-to-back
        with constant length `fir_bins`. Total T must equal n_trials * fir_bins.
    onsets_bins, durations_bins : array-like of length n_trials, optional
        Explicit onset (start index) and duration for each trial in bins.
        If given, `fir_bins` is ignored.
    trial_colors : dict[str, str], optional
        Map from label -> color (any Matplotlib color). Unmapped labels get a default cycle.
    trial_alpha : float, default 0.20
        Transparency of trial background trials.
    stack : bool, default False
        If False, overlay all series on a single axis. If True, one axis per series (vertical stack).
    line_kwargs : dict or dict[str, dict], optional
        Global kwargs for all lines (e.g., {"linewidth":2} or {"color":"black"}), or per-series kwargs
        keyed by series name (e.g., {"demand":{"color":"k"}}).
    figsize : tuple, default (20,4)
        Figure size.
    xlabel, ylabel, title : str
        Axis labels and title.
    include_trial_labels : bool, default True
        Include shaded trial type labels in the legend.
    include_line_labels : bool, default True
        Include line labels in the legend.
    line_labels : dict or list, optional
        Custom labels for lines. If dict, keyed by series name.
        If list, aligned order with list/array `series`.
    show_legend : bool, default True
        Show legend if any labels exist.
    legend_loc : str
        Legend location string.
    legend_kwargs : dict, optional
        Extra kwargs forwarded to `Axes.legend`.
    save_loc : str or None
        If provided, saves the figure to this path (format inferred from extension).
    dpi: integer, defalut 400
        dots per inch (dpi) specifyihng resoulution of figure when saving.
    show : bool, default True
        Whether to call plt.show(). (Figure is always returned.)

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes or np.ndarray of Axes
    """
    # normalize series to dict[name] -> 1D array
    if isinstance(series, dict):
        series_dict = {str(k): np.asarray(v).ravel() for k, v in series.items()}
        series_names = list(series_dict.keys())
    elif isinstance(series, (list, tuple)):
        series_dict = {f"s{i}": np.asarray(v).ravel() for i, v in enumerate(series)}
        series_names = list(series_dict.keys())
    else:
        series_dict = {"s0": np.asarray(series).ravel()}
        series_names = ["s0"]
    # lengths
    lengths = {k: v.shape[0] for k, v in series_dict.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All series must have the same length, got lengths: {lengths}")
    T = next(iter(lengths.values()))
    # trial windows
    trial_labels = np.asarray(trial_labels)
    n_trials = trial_labels.shape[0]
    if (onsets_bins is None) or (durations_bins is None):
        if fir_bins is None:
            raise ValueError("Provide either (onsets_bins & durations_bins) or fir_bins.")
        if n_trials * int(fir_bins) != T:
            raise ValueError(f"T={T} must equal n_trials * fir_bins = {n_trials} * {fir_bins}.")
        onsets_bins = np.arange(0, T, int(fir_bins))
        durations_bins = np.full(n_trials, int(fir_bins), dtype=int)
    else:
        onsets_bins = np.asarray(onsets_bins, dtype=int).ravel()
        durations_bins = np.asarray(durations_bins, dtype=int).ravel()
        if onsets_bins.shape[0] != n_trials or durations_bins.shape[0] != n_trials:
            raise ValueError("onsets_bins and durations_bins must match number of trials.")
        ends = onsets_bins + durations_bins
        if np.any(onsets_bins < 0) or np.any(ends > T):
            raise ValueError("Some trial windows fall outside the series length.")
    # colors for trials
    if trial_colors is None:
        trial_colors = {'Go': 'olivedrab', 'Stop': 'orangered'}
    unique_labels = list(dict.fromkeys(trial_labels.tolist()))
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['#1f77b4'])
    color_cycle_iter = iter(prop_cycle)
    resolved_colors = {}
    for lab in unique_labels:
        if lab in trial_colors:
            resolved_colors[lab] = trial_colors[lab]
        else:
            try:
                resolved_colors[lab] = next(color_cycle_iter)
            except StopIteration:
                color_cycle_iter = iter(prop_cycle)
                resolved_colors[lab] = next(color_cycle_iter)
    # figure/axes
    if stack:
        fig, axes = plt.subplots(len(series_dict), 1, figsize=figsize, sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = np.array([ax])
    # shaded trials on each axis
    labeled_types = set()
    for ax in axes:
        for lab, start, dur in zip(trial_labels, onsets_bins, durations_bins):
            color = resolved_colors.get(lab, '#cccccc')
            label = (lab if (include_trial_labels and lab not in labeled_types) else None)
            ax.axvspan(int(start), int(start + dur), facecolor=color, alpha=trial_alpha, label=label)
            if include_trial_labels:
                labeled_types.add(lab)
        ax.set_xlim(0, T)
        ax.grid(True, alpha=0.25, linewidth=0.8)
    # line labels (custom or default)
    # build mapping name -> label string (or None)
    if line_labels is None:
        name_to_label = {name: (name if include_line_labels else None) for name in series_names}
    else:
        if isinstance(line_labels, dict):
            name_to_label = {name: (line_labels.get(name, name) if include_line_labels else None)
                             for name in series_names}
        else:  # list/tuple aligned with order of series_names
            if len(line_labels) != len(series_names):
                raise ValueError("line_labels list length must match the number of series.")
            name_to_label = {name: (lab if include_line_labels else None)
                             for name, lab in zip(series_names, line_labels)}
    # plot lines
    for i, (name, y) in enumerate(series_dict.items()):
        ax_i = axes[i] if stack else axes[0]
        # kwargs resolution
        kw_default = dict(linewidth=1.5)
        if isinstance(line_kwargs, dict) and any(isinstance(v, dict) for v in line_kwargs.values()):
            kw = {**kw_default, **line_kwargs.get(name, {})}
        else:
            kw = {**kw_default, **(line_kwargs or {})}
        if 'color' not in kw:
            colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['#1f77b4'])
            kw['color'] = colors[i % len(colors)]
        ax_i.plot(np.arange(T), y, label=name_to_label[name], **kw)
        if stack:
            ax_i.set_ylabel(ylabel)
    # labels/titles/legend
    axes[-1].set_xlabel(xlabel)
    if not stack:
        axes[0].set_ylabel(ylabel)
    if title:
        axes[0].set_title(title)
    if show_legend:
        # collect handles/labels (spans from first axis; lines from last axis)
        handles_spans, labels_spans = axes[0].get_legend_handles_labels()
        handles_lines, labels_lines = axes[-1].get_legend_handles_labels()
        # de-duplicate preserving order
        seen = set()
        handles, labels = [], []
        for h, l in list(zip(handles_spans, labels_spans)) + list(zip(handles_lines, labels_lines)):
            if l and l not in seen:
                handles.append(h); labels.append(l); seen.add(l)
        if labels:
            (legend_kwargs := legend_kwargs or {})
            axes[0].legend(handles, labels, loc=legend_loc, frameon=True, **legend_kwargs)
    if hlines is not None:
        if not isinstance(hlines, (list,tuple,np.ndarray)):
            hlines = [hlines]
        for ax in axes:
            for y in hlines:
                ax.axhline(y, **(hline_kwargs or dict(color='k' , linestyle='--', linewidth=1)))
    fig.tight_layout()
    if save_loc:
        fig.savefig(save_loc, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        plt.close()
    else:
        plt.close(fig)
    return fig, (axes if stack else axes[0])




# heatmap of Dsp-----------------------------------------------------------------------
def plot_dsp_timeseries_heatmaps(
    dsp_1d,
    n_bins=None,
    subject_idx=None,
    figsize=(14, 5),
    vmin=0.0,
    vmax=1.0,
    cmap_dsp="viridis",
    class_colors=None,
    class_labels=None,
    origin="upper",
    annotate_trials=True,
    trial_line_kw=None,
    show=True,
    legend_outside=True,
    legend_right_pad=0.82,
    cbar_inset=True,
    cbar_width="2.0%",     # width as % of ax width
    cbar_height="70%",     # height as % of ax height
    cbar_loc="center right",
    cbar_pad=0.0,
    save_loc=None
    ):
    low_th=0.52233887; high_th=0.61220267
    dsp = np.asarray(dsp_1d, dtype=float).ravel()
    T = dsp.size
    if not np.isfinite(low_th) or not np.isfinite(high_th):
        raise ValueError("low_th and high_th must be finite.")
    if low_th >= high_th:
        raise ValueError("Require low_th < high_th.")
    # classify 1-D vector
    class_vec = np.full(T, -1, dtype=int)
    valid = np.isfinite(dsp)
    class_vec[valid & (dsp <  low_th)] = 0
    class_vec[valid & (dsp >= low_th) & (dsp < high_th)] = 1
    class_vec[valid & (dsp >= high_th)] = 2
    # colors/labels
    if class_colors is None:
        class_colors = {0: "royalblue", 1: "gold", 2: "crimson"}
    if class_labels is None:
        class_labels = {0: "Parallel", 1: "Mixed", 2: "Serial"}
    cmap_cls = ListedColormap(
        ["lightgray", class_colors[0], class_colors[1], class_colors[2]]
    )
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm_cls = BoundaryNorm(bounds, cmap_cls.N)
    # optional reshape for trial-by-trial view
    class_mat = None
    if n_bins is not None:
        if T % n_bins != 0:
            raise ValueError("T must be divisible by n_bins to reshape.")
        n_trials = T // n_bins
        dsp_img = dsp.reshape(n_trials, n_bins)
        class_img = class_vec.reshape(n_trials, n_bins)
        class_mat = class_img
    else:
        dsp_img = dsp[np.newaxis, :]
        class_img = class_vec[np.newaxis, :]
        n_trials, n_bins = dsp_img.shape  # here n_trials==1, n_bins==T
    if trial_line_kw is None:
        trial_line_kw = {"color": "k", "lw": 0.8, "alpha": 0.3}
    # figure (manual right margin for legend space)
    fig, (ax_dsp, ax_cls) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    if legend_outside:
        fig.subplots_adjust(right=legend_right_pad)
    # D_SP heatmap
    im_dsp = ax_dsp.imshow(
        dsp_img, aspect="auto", origin=origin, vmin=vmin, vmax=vmax, cmap=cmap_dsp
    )
    ax_dsp.set_ylabel("Trial" if dsp_img.shape[0] > 1 else "")
    subtxt = f" – Subject {subject_idx + 1}" if subject_idx is not None else ""
    ax_dsp.set_title(r"$D^{SP}_t$ heatmap" + subtxt)
    # SMALL inset colorbar on the right of the TOP plot (doesn't affect layout)
    if cbar_inset:
        cax = inset_axes(
            ax_dsp,
            width=cbar_width,
            height=cbar_height,
            loc=cbar_loc,
            bbox_to_anchor=(0.0, 0.0, 1.05, 1.0),
            bbox_transform=ax_dsp.transAxes,
            borderpad=cbar_pad,
        )
        cbar = fig.colorbar(im_dsp, cax=cax, orientation="vertical")
    else:
        # fallback: regular right-side colorbar only for top axis (will narrow top plot)
        cbar = fig.colorbar(im_dsp, ax=ax_dsp, fraction=0.046, pad=0.04)
    cbar.set_label(r"$D^{SP}_t$")
    # Classification heatmap
    im_cls = ax_cls.imshow(
        class_img, aspect="auto", origin=origin, cmap=cmap_cls, norm=norm_cls
    )
    ax_cls.set_xlabel("Concatenated time (bins)")
    ax_cls.set_ylabel("Trial" if class_img.shape[0] > 1 else "")
    ax_cls.set_title(
        f"Parallel / Mixed / Serial{subtxt}  (τ low={low_th:.3f}, τ high={high_th:.3f})"
    )
    # Legend outside to the right of the second plot
    legend_handles = [
        Patch(facecolor=class_colors[0], edgecolor="none", label=class_labels[0]),
        Patch(facecolor=class_colors[1], edgecolor="none", label=class_labels[1]),
        Patch(facecolor=class_colors[2], edgecolor="none", label=class_labels[2]),
    ]
    if np.any(class_vec == -1):
        legend_handles.insert(
            0, Patch(facecolor="lightgray", edgecolor="none", label="Invalid/NaN")
        )
    if legend_outside:
        ax_cls.legend(
            handles=legend_handles,
            labels=[h.get_label() for h in legend_handles],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True
        )
    else:
        ax_cls.legend(
            handles=legend_handles,
            labels=[h.get_label() for h in legend_handles],
            loc="upper right",
            frameon=True
        )
    # Trial boundaries
    if (dsp_img.shape[0] > 1) and annotate_trials:
        for k in range(1, dsp_img.shape[0]):
            ax_dsp.axhline(k - 0.5, **trial_line_kw)
            ax_cls.axhline(k - 0.5, **trial_line_kw)
    if (class_img.shape[0] == 1) and (n_bins is not None) and annotate_trials:
        for x in range(n_bins, class_img.shape[1], n_bins):
            ax_dsp.axvline(x - 0.5, **trial_line_kw)
            ax_cls.axvline(x - 0.5, **trial_line_kw)
    # Ticks
    ax_cls.set_xticks(np.linspace(0, n_bins - 1, num=min(n_bins, 11), dtype=int))
    for ax in (ax_dsp, ax_cls):
        ax.set_yticks(np.linspace(0, dsp_img.shape[0] - 1,
                                  num=min(dsp_img.shape[0], 11), dtype=int))
    if save_loc != None:
        plt.savefig(save_loc, dpi=400)
    if show:
        plt.show()
    return fig, (ax_dsp, ax_cls), class_vec, class_mat


# Heatmap of Bnode------------------------------------------------------------------------
def plot_bnode_heatmap(
    bnode,
    transpose=False,          # True if input is (T, n_nodes)
    node_subset=None,         # indices or boolean mask over NODES (rows)
    n_bins=None,              # vertical line every n_bins
    vmin=None,
    vmax=None,
    cmap="viridis",
    width=12,                 # figure width (inches)
    height_per_node=0.03,     # inches per node for auto height
    min_height=3.0,           # clamp min height (inches)
    max_height=10.0,          # clamp max height (inches)
    dpi=150,
    show=True,
    save_loc=None,
):
    """
    Minimal B_node heatmap with NODES on Y and TIME on X.
    Figure height auto-scales with number of nodes (after subsetting).
    """
    A = np.asarray(bnode, dtype=float)
    X = A.T if transpose else A
    if X.ndim != 2:
        raise ValueError("bnode must be 2D.")
    # rows = nodes, cols = time
    n_nodes, T = X.shape
    # Apply subset on rows (nodes) only, AFTER orientation is fixed
    if node_subset is not None:
        node_subset = np.asarray(node_subset)
        if node_subset.dtype == bool:
            if node_subset.size != n_nodes:
                raise ValueError(
                    "Boolean node_subset must match number of nodes."
                )
            rowsel = node_subset
        else:
            rowsel = node_subset.astype(int)
        X = X[rowsel, :]
        orig_idx = np.arange(n_nodes)[rowsel]
        n_nodes = X.shape[0]
    else:
        orig_idx = np.arange(n_nodes)
    # auto figure height based on nodes
    fig_height = height_per_node * n_nodes
    fig_height = max(min_height, min(max_height, fig_height))
    # Layout: main axis + slim full-height colorbar
    fig = plt.figure(figsize=(width, fig_height), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=1, ncols=2, width_ratios=[1.0, 0.045], wspace=0.06
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    # Heatmap
    im = ax.imshow(
        X,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_title("B_node heatmap (nodes × time)")
    ax.set_xlabel("Time (bins)")
    ax.set_ylabel("Node")
    # Sparse ticks
    y_ticks = np.linspace(0, n_nodes - 1, num=min(n_nodes, 16), dtype=int)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(orig_idx[r])) for r in y_ticks])
    x_ticks = np.linspace(0, T - 1, num=min(T, 16), dtype=int)
    ax.set_xticks(x_ticks)
    # Trial boundaries
    if (n_bins is not None) and (n_bins > 0):
        for x in range(n_bins, T, n_bins):
            ax.axvline(x - 0.5, color="k", lw=0.6, alpha=0.25)
    # Colorbar
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.set_label("B_node")
    fig.tight_layout()
    if save_loc is not None:
        fig.savefig(save_loc, dpi=400)
    if show:
        plt.show()
    return fig, ax, cb, orig_idx


# Brain Plot of Simulated timeseries
## Note this is for simulation data only
def plot_node_process_state_trans_simulation(
  node_idx, 
  bold, 
  nodes = 200, 
  networks = 7,
  res = 2,
  bold_boost= 2,
  plt_type = "stat",              # "stat" | "glass"
  atlas_loc = atlas_loc,
  plt_save_dir = None,
  plt_subj_lab = None,
  plt_show = True,
  trial_node_idx = None,          # dict like {'proc_state': {'Go': {'serial':...}}, 'proc_transition': {...}}
  trial_types_order = None        # optional list to control trial plotting order (e.g., ['Go','Stop'])
  ):
  """
  This function reshapes the simulated timeseries for plotting. Specifically,
  this function uses the schaefer atlas with 7 yeo networks for shaping. Then,
  glass brain function from nilearn is used for plotting.

  If trial_node_idx is provided, this will additionally plot per-trial-type
  state/transition node maps using the same conventions and filenames, with
  the trial type appended to the title and filename.
  """
  # fetch atlas
  sch_atl = nilearn.datasets.fetch_atlas_schaefer_2018(
    n_rois = nodes, 
    yeo_networks= networks, 
    resolution_mm= res, 
    data_dir= atlas_loc
    )
  atlas = nilearn.image.load_img(sch_atl.maps)
  atlas_data = atlas.get_fdata()
  atlas_vals = np.zeros_like(atlas_data)
  # summarize bold per node
  bld0 = bold.reshape(-1,nodes)
  if bold_boost > 1:
    bld0_vals = np.array([np.mean(ts)*bold_boost for ts in bld0])
  else:
    bld0_vals = np.array([np.mean(ts) for ts in bld0])
  for ii in range(1, nodes + 1):
    atlas_vals[atlas_data == ii] = bld0_vals[ii-1]
  stat_data = nilearn.image.new_img_like(atlas, atlas_vals)
  msker = nilearn.maskers.NiftiLabelsMasker(atlas)
  msk_vals_all = msker.fit_transform(stat_data)
  # fixed labels
  proc_st = ["State Processing Nodes: ", "Transition Processing Nodes: "]
  proc_st_fn = ["state_processing_nodes_", "transition_processing_nodes_"]
  proc_lab_l = [['Serial', 'Mixed', 'Parallel'],['Serial', 'Mixed', 'Parallel', 'General']]
  proc_lab_fn_l = [['serial', 'mixed', 'parallel'],['serial', 'mixed', 'parallel','general']]
  # helper: plot one category dict (global or per-trial)
  def _plot_category_dict(cat_name, cat_dict, title_prefix="", file_suffix=""):
    # cat_dict should be {'serial': arr, 'mixed': arr, 'parallel': arr, ['general': arr]}
    # cat_name either 'proc_state' or 'proc_transition'
    if cat_name == 'proc_state':
      ind = 0
      proc_lab = proc_lab_l[ind]
      proc_lab_fn = proc_lab_fn_l[ind]
      keys = ['serial','mixed','parallel']
    else:
      ind = 1
      proc_lab = proc_lab_l[ind]
      proc_lab_fn = proc_lab_fn_l[ind]
      keys = ['serial','mixed','parallel','general']
    for idx_mode, jj in enumerate(keys):
      if jj not in cat_dict:
        continue
      node_list = cat_dict[jj]
      if node_list is None or (hasattr(node_list, "size") and node_list.size == 0) or (isinstance(node_list, list) and len(node_list) == 0):
        continue
      msk_vals = np.zeros_like(msk_vals_all)
      msk_vals[node_list] = msk_vals_all[node_list]
      msk_vals_img = msker.inverse_transform(msk_vals)
      title_txt = str(title_prefix + proc_st[ind] + proc_lab[idx_mode])
      if plt_type == "glass":
        nilearn.plotting.plot_glass_brain(
          msk_vals_img, plot_abs = False,
          colorbar = True,
          title= title_txt)
      elif plt_type == "stat":
        nilearn.plotting.plot_stat_map(
          msk_vals_img,
          colorbar = True, 
          title= title_txt,
          draw_cross=False
          )
      if plt_save_dir is not None:
        if plt_subj_lab is not None:
          fname = "brain-plot-" + plt_type + "_" + plt_subj_lab + "_" + proc_lab_fn[idx_mode] + "_" + proc_st_fn[ind] + file_suffix + ".tiff"
        else:
          fname = "brain-plot-" + plt_type + "_" + proc_lab_fn[idx_mode] + "_" + proc_st_fn[ind] + file_suffix + ".tiff"
        plt.savefig(os.path.join(plt_save_dir, fname), dpi=400)
      if plt_show == True:
        plt.show(), plt.close()
      elif plt_show == False:
        plt.close()
  # GLOBAL PLOTS
  # node_idx is expected to be {'proc_state': {'serial':..., 'mixed':..., 'parallel':...},
  #                             'proc_transition': {'serial':..., 'mixed':..., 'parallel':..., 'general':...}}
  if isinstance(node_idx, dict):
    # proc_state
    if 'proc_state' in node_idx:
      _plot_category_dict('proc_state', node_idx['proc_state'], title_prefix="", file_suffix="")
    # proc_transition
    if 'proc_transition' in node_idx:
      _plot_category_dict('proc_transition', node_idx['proc_transition'], title_prefix="", file_suffix="")
  # TRIAL-CONDITIONED PLOTS (OPTIONAL)
  # trial_node_idx structure:
  # {'proc_state': {'Go': {'serial':..., 'mixed':..., 'parallel':...},
  #                 'Stop': {...}, ...},
  #  'proc_transition': {'Go': {...}, 'Stop': {...}, ...}}
  if trial_node_idx is not None and isinstance(trial_node_idx, dict):
    # determine trial order
    all_trials = []
    for k in ['proc_state','proc_transition']:
      if k in trial_node_idx:
        all_trials.extend(list(trial_node_idx[k].keys()))
    all_trials = sorted(list(set(all_trials)))
    if (trial_types_order is not None) and len(trial_types_order) > 0:
      # keep only those that actually exist in data; preserve requested order
      ordered = [t for t in trial_types_order if t in all_trials]
      # append any remaining unseen types at the end (if present)
      remainder = [t for t in all_trials if t not in ordered]
      trial_list = ordered + remainder
    else:
      trial_list = all_trials
    # plot per trial type
    for tr in trial_list:
      title_prefix = "[" + str(tr) + "] "
      file_suffix  = "_" + str(tr).replace(" ","-")
      # state
      if 'proc_state' in trial_node_idx and tr in trial_node_idx['proc_state']:
        _plot_category_dict('proc_state', trial_node_idx['proc_state'][tr], title_prefix=title_prefix, file_suffix=file_suffix)
      # transition
      if 'proc_transition' in trial_node_idx and tr in trial_node_idx['proc_transition']:
        _plot_category_dict('proc_transition', trial_node_idx['proc_transition'][tr], title_prefix=title_prefix, file_suffix=file_suffix)




