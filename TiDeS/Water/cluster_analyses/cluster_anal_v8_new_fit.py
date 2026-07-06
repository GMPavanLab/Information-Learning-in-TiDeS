#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster Coordination Analysis Tool (adjusted)
=============================================
Analyzes collective molecular motion in clusters (size ≥2) from MD simulations.
Focuses on: size distributions, temporal evolution, spectral characteristics,
and shape anisotropy of coordinated molecular clusters.

Key adjustments vs original:
- Fixed stray print-line syntax error at end of script
- Robust integer casting for `cluster_id` and `size`
- Safe log-y plotting in size distribution (no zeros) & per-frame normalization
- Optional frame-based frequency handling in spectral analysis (dt=1 if --use_frames)
- More robust parsing of size labels in PSD/peaks
- Save extra CSVs (time series, anisotropy, raw PSD)
- Despine helper applied consistently; removed ineffective rcParams for spines
- Rg scaling now fits on per-size medians for robustness
- Theoretical size distribution fitting with N(s;f) model
- NEW PLOTS:
  * Polarization: P=(M-S)/(M+S) in [-1,1]
  * Coordination intensity: I = N_multi · M
  * Singleton molecules trend: S(t)
  * Ratio panel (T/S, T/M, S/M, M/S) with optional smoothing
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.ticker import MaxNLocator, ScalarFormatter, LogFormatterSciNotation
from scipy.optimize import curve_fit

# Visual styling
FREEZE_COLOR = "#2E86AB"  # Deep blue
MELT_COLOR = "#A23B72"    # Deep rose
NEUTRAL_COLOR = "#4A4A4A"
FRAME_A, FRAME_B = 215, 260  # red, blue
trend_min_window = 5

COLS = [
    "cluster_id", "size",
    "com_x", "com_y", "com_z",
    "rog",
    "g_xx", "g_yy", "g_zz", "g_xy", "g_xz", "g_yz"
]


def set_plot_style():
    """Configure publication-quality plot aesthetics."""
    plt.rcParams.update({
        "figure.figsize": (8, 6),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.7,
        "axes.labelsize": 16,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "legend.fontsize": 11,
        "legend.framealpha": 0.9,
    })


def style_axes(ax, xlabel="Time [ns]", ylabel="Population fraction", logy=False):
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=16)

    fx = ScalarFormatter(useMathText=False)
    ax.xaxis.set_major_formatter(fx)
    ax.xaxis.get_offset_text().set_fontsize(10)

    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    else:
        fy = ScalarFormatter(useMathText=True)
        fy.set_powerlimits((-10, 10))
        ax.yaxis.set_major_formatter(fy)

    ax.yaxis.offsetText.set_fontsize(10)


def maybe_scientific_y(ax: plt.Axes, threshold: float = 100.0) -> None:
    """
    If plotted y-values have magnitude >= threshold, force scientific notation on Y.
    """
    try:
        ymax = 0.0
        for line in ax.lines:
            y = np.asarray(line.get_ydata(), dtype=float)
            if y.size == 0:
                continue
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            ymax = max(ymax, float(np.max(np.abs(y))))
        if np.isfinite(ymax) and ymax >= threshold:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    except Exception:
        pass


def plot_state_ratio_timeseries(
    pop_freeze: pd.DataFrame,
    pop_melt: pd.DataFrame,
    time_scale: float,
    outdir: Path,
    use_frames: bool = False,
    smooth: bool = False,
    window: int = 15,
):
    """Plot ratio time-series (Freeze vs Melt), optionally smoothed."""

    def _ma_ignore_nan(arr: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return arr.astype(float, copy=True)
        arr = arr.astype(float, copy=False)
        m = np.isfinite(arr).astype(float)
        arr0 = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        k = np.ones(w, dtype=float) / float(w)
        num = np.convolve(arr0, k, mode="same")
        den = np.convolve(m, k, mode="same")
        out = np.full_like(num, np.nan, dtype=float)
        ok = den > 0
        out[ok] = num[ok] / den[ok]
        return out

    def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        num = num.astype(float, copy=False)
        den = den.astype(float, copy=False)
        out = np.full_like(num, np.nan, dtype=float)
        ok = den > 0
        out[ok] = num[ok] / den[ok]
        return out

    def _get_series(pop: pd.DataFrame):
        if pop is None or pop.empty:
            return None
        x = pop["frame"].values.astype(float) if use_frames else pop["frame"].values.astype(float) * float(time_scale)
        total = pop["total_molecules"].values.astype(float)
        single_mol = pop["n_singletons"].values.astype(float)       # size==1 => molecules
        multi_mol = pop["multimer_molecules"].values.astype(float)  # size>=2 => molecules
        return x, total, single_mol, multi_mol

    def _plot_one(y_freeze, y_melt, ylabel, fname, logy=False):
        fig, ax = plt.subplots(figsize=(6, 5))

        if y_freeze is not None:
            ax.plot(x_f, y_freeze, color=FREEZE_COLOR, lw=2.2, alpha=0.9, label="Freeze")
        if y_melt is not None:
            ax.plot(x_m, y_melt, color=MELT_COLOR, lw=2.2, alpha=0.9, label="Melt")

        xlabel = "Frame" if use_frames else "Time [ns]"
        style_axes(ax, xlabel=xlabel, ylabel=ylabel, logy=logy)
        maybe_scientific_y(ax, threshold=100.0)

        ax.legend(loc="best", framealpha=0.9, fontsize=12)
        fig.tight_layout()
        ax.set_box_aspect(0.8)

        plt.savefig(outdir / fname, dpi=300)
        plt.close(fig)

    s_f = _get_series(pop_freeze)
    s_m = _get_series(pop_melt)

    x_f = x_m = None
    y1_f = y2_f = y3_f = None
    y1_m = y2_m = y3_m = None
    y4_f = y4_m = None

    if s_f is not None:
        x_f, total_f, single_f, multi_f = s_f
        y1_f = _ratio(total_f, single_f)   # T/S
        y2_f = _ratio(total_f, multi_f)    # T/M
        y3_f = _ratio(single_f, multi_f)   # S/M
        y4_f = _ratio(multi_f, single_f)   # M/S

    if s_m is not None:
        x_m, total_m, single_m, multi_m = s_m
        y1_m = _ratio(total_m, single_m)
        y2_m = _ratio(total_m, multi_m)
        y3_m = _ratio(single_m, multi_m)
        y4_m = _ratio(multi_m, single_m)

    if smooth:
        if y1_f is not None: y1_f = _ma_ignore_nan(y1_f, window)
        if y2_f is not None: y2_f = _ma_ignore_nan(y2_f, window)
        if y3_f is not None: y3_f = _ma_ignore_nan(y3_f, window)
        if y4_f is not None: y4_f = _ma_ignore_nan(y4_f, window)
        if y1_m is not None: y1_m = _ma_ignore_nan(y1_m, window)
        if y2_m is not None: y2_m = _ma_ignore_nan(y2_m, window)
        if y3_m is not None: y3_m = _ma_ignore_nan(y3_m, window)
        if y4_m is not None: y4_m = _ma_ignore_nan(y4_m, window)

    tag = f"_smooth{window}" if smooth else ""
    # --- NEW: population-style plot for M/S ---
    fig, ax = plt.subplots(figsize=(10, 6))

    if y4_f is not None:
        ax.plot(x_f, y4_f, label="Freeze", color=FREEZE_COLOR, linewidth=2.5, alpha=0.85)
    if y4_m is not None:
        ax.plot(x_m, y4_m, label="Melt", color=MELT_COLOR, linewidth=2.5, alpha=0.85)

    ax.set_xlabel("Time [ns]", fontsize=20)
    ax.set_ylabel("M/S", fontsize=20)  # short + readable
    ax.tick_params(axis="both", labelsize=20)

    # optional: scientific notation if values get large
    ymax = 0.0
    for line in ax.lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        y = y[np.isfinite(y)]
        if y.size:
            ymax = max(ymax, float(np.max(np.abs(y))))
    if ymax >= 100:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

    ax.legend(loc="best", fontsize=14)
    plt.tight_layout()
    plt.savefig(outdir / f"population_ratio_M_over_S{tag}.png", dpi=300)
    plt.close(fig)
    _plot_one(y1_f, y1_m, "T/S", f"ratio_T_over_S{tag}.png")
    _plot_one(y2_f, y2_m, "T/M", f"ratio_T_over_M{tag}.png")
    _plot_one(y3_f, y3_m, "S/M", f"ratio_S_over_M{tag}.png")
    _plot_one(y4_f, y4_m, "M/S", f"ratio_M_over_S{tag}.png")


def moving_average(arr, window):
    if window <= 1:
        return arr.copy()
    kernel = np.ones(window) / window
    res = np.convolve(arr, kernel, mode='same')
    half = window // 2
    for i in range(half):
        res[i] = np.mean(arr[:i+half+1])
        res[-(i+1)] = np.mean(arr[-(i+half+1):])
    return res


def auto_trend_window(T):
    w = max(trend_min_window, int(round(T / 500)))
    if w % 2 == 0:
        w += 1
    return w


def _despine(ax: plt.Axes):
    """Hide top/right spines (rcParams cannot do this globally)."""
    try:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    except Exception:
        pass


def parse_frame_id(path: Path) -> int:
    """Extract frame number from filename (e.g., freeze_cluster_data.123 -> 123)."""
    match = re.search(r"\.(\d+)$", path.name)
    if not match:
        raise ValueError(f"Cannot parse frame ID from: {path.name}")
    return int(match.group(1))


def find_cluster_files(folder: Path, prefix: str) -> list[tuple[int, Path]]:
    """Find and sort cluster data files by frame number."""
    if not folder.exists():
        return []

    files: list[tuple[int, Path]] = []
    for p in folder.glob(f"{prefix}_cluster_data.*"):
        try:
            frame = parse_frame_id(p)
            files.append((frame, p))
        except ValueError:
            continue

    return sorted(files, key=lambda x: x[0])


def read_cluster_file(path: Path, frame: int, label: str) -> pd.DataFrame:
    """Parse a single cluster data file into DataFrame."""
    rows: list[list[float]] = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", ";")):
                continue
            parts = line.split()
            if len(parts) < len(COLS):
                continue
            try:
                vals = [float(x) for x in parts[: len(COLS)]]
                rows.append(vals)
            except ValueError:
                continue

    if not rows:
        return pd.DataFrame(columns=["frame", "label"] + COLS)

    df = pd.DataFrame(rows, columns=COLS)

    # Enforce integer types for identifiers early (robust to floats in source)
    for col in ("cluster_id", "size"):
        df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")

    df.insert(0, "frame", frame)
    df.insert(1, "label", label)
    return df


def load_dataset(folder: Path, prefix: str, label: str) -> pd.DataFrame:
    """Load all cluster files from a directory into single DataFrame."""
    files = find_cluster_files(folder, prefix)
    if not files:
        print(f"Warning: No {label} cluster files found in {folder}")
        return pd.DataFrame(columns=["frame", "label"] + COLS)

    dfs: list[pd.DataFrame] = []
    for frame, path in files:
        df = read_cluster_file(path, frame, label)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["frame", "label"] + COLS)

    combined = pd.concat(dfs, ignore_index=True)

    if "cluster_id" in combined.columns:
        combined["cluster_id"] = combined["cluster_id"].astype("Int64")
    if "size" in combined.columns:
        combined["size"] = combined["size"].astype("Int64")

    combined.sort_values(["frame", "cluster_id"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print(f"[{label}] Loaded {len(dfs)} frames: {files[0][0]} → {files[-1][0]}")
    return combined


def compute_shape_anisotropy(row) -> float:
    """
    Calculate shape anisotropy κ² from gyration tensor eigenvalues.
    κ² = 0: spherical, κ² → 1: rod-like or disk-like
    """
    G = np.array([
        [row.g_xx, row.g_xy, row.g_xz],
        [row.g_xy, row.g_yy, row.g_yz],
        [row.g_xz, row.g_yz, row.g_zz],
    ])
    G = 0.5 * (G + G.T)  # Ensure symmetry

    try:
        eigs = np.linalg.eigvalsh(G)  # Sorted ascending
        s1, s2, s3 = eigs
        s = s1 + s2 + s3
        if s <= 0:
            return float("nan")
        numerator = (s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2
        return float(numerator / (2 * s * s))
    except np.linalg.LinAlgError:
        return float("nan")


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_multimer_population(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-frame statistics focusing on multimers (size ≥2).
    These represent coordinated molecular groups.
    """
    if df.empty:
        return pd.DataFrame()

    df_multi = df[df["size"] >= 2].copy()

    results = []
    for frame in sorted(df["frame"].unique()):
        frame_data = df[df["frame"] == frame]
        multi_data = df_multi[df_multi["frame"] == frame]

        stats = {
            "frame": frame,
            "n_multimers": int(len(multi_data)),
            "n_singletons": int((frame_data["size"] == 1).sum()),
            "multimer_molecules": int(multi_data["size"].sum()) if len(multi_data) > 0 else 0,
            "max_cluster_size": int(multi_data["size"].max()) if len(multi_data) > 0 else 0,
            "mean_cluster_size": float(multi_data["size"].mean()) if len(multi_data) > 0 else 0.0,
            "total_molecules": int(frame_data["size"].sum()),
        }
        results.append(stats)

    return pd.DataFrame(results)


def compute_size_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time series of cluster counts for each size (multimers only).
    Returns: DataFrame with frames as index, sizes as columns (all zeros filled).
    """
    if df.empty:
        return pd.DataFrame()

    df_multi = df[df["size"] >= 2].copy()
    if df_multi.empty:
        return pd.DataFrame()

    df_multi["size"] = pd.to_numeric(df_multi["size"], errors="coerce").round().astype("Int64")

    counts = (
        df_multi.groupby(["frame", "size"], dropna=True).size().rename("count").reset_index()
    )

    frame_min, frame_max = int(df["frame"].min()), int(df["frame"].max())
    all_frames = range(frame_min, frame_max + 1)
    all_sizes = sorted(x for x in df_multi["size"].dropna().unique())

    pivot = counts.pivot(index="frame", columns="size", values="count")
    pivot = pivot.reindex(index=all_frames, columns=all_sizes, fill_value=0)
    pivot.columns = [int(c) for c in pivot.columns]

    return pivot.fillna(0).astype(int)


def compute_spectral_analysis(
    timeseries: pd.DataFrame,
    time_scale: float,
    window_type: str = "hann",
    dt_override: bool | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Perform FFT-based spectral analysis on cluster size time series.
    Returns: (frequencies, psd_norm_df, psd_raw_df)
      - psd_norm_df: each column normalized to area=1 (shape-only)
      - psd_raw_df: raw periodogram-like power (comparable magnitudes)
    """
    if timeseries.empty or len(timeseries) < 4:
        return np.array([]), pd.DataFrame(), pd.DataFrame()

    N = len(timeseries)
    if dt_override is True:
        dt = 1.0
    else:
        dt = time_scale if time_scale > 0 else 1.0

    if window_type == "hann":
        window = np.hanning(N)
    elif window_type == "hamming":
        window = np.hamming(N)
    else:
        window = np.ones(N)

    window_norm = (window ** 2).sum()

    freqs = np.fft.rfftfreq(N, d=dt)
    psd_norm: dict[str, np.ndarray] = {}
    psd_raw: dict[str, np.ndarray] = {}

    for size in timeseries.columns:
        signal_data = timeseries[size].values.astype(float)
        signal_data = signal_data - signal_data.mean()

        fft_result = np.fft.rfft(signal_data * window)
        psd = (np.abs(fft_result) ** 2) / (window_norm * N)

        psd_raw[f"size_{int(size)}"] = psd.copy()

        total_power = psd.sum()
        if total_power > 0:
            psd = psd / total_power
        psd_norm[f"size_{int(size)}"] = psd

    psd_norm_df = pd.DataFrame(psd_norm, index=freqs)
    psd_raw_df = pd.DataFrame(psd_raw, index=freqs)
    return freqs, psd_norm_df, psd_raw_df


def identify_dominant_frequencies(
    freqs: np.ndarray,
    psd_df: pd.DataFrame,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Identify dominant frequency peaks for each cluster size.
    Returns DataFrame with size, dominant frequency, and relative power.
    """
    if psd_df.empty or len(freqs) == 0:
        return pd.DataFrame()

    results = []
    for col in psd_df.columns:
        try:
            size = int(float(col.split("_")[1]))
        except Exception:
            continue
        psd = psd_df[col].values
        peaks, properties = signal.find_peaks(psd, height=threshold, prominence=threshold / 2)
        if len(peaks) > 0:
            max_idx = peaks[np.argmax(properties["peak_heights"])]
            results.append(
                {
                    "size": size,
                    "dominant_freq": freqs[max_idx],
                    "relative_power": float(psd[max_idx]),
                    "n_peaks": int(len(peaks)),
                }
            )

    return pd.DataFrame(results)


def theoretical_size_distribution(s: np.ndarray, A: float, B: float, f: float) -> np.ndarray:
    """
    N(s;f) = A * s^(-τ) * exp[-B * s * |f-0.5|^(1/σ)]
    """
    TAU = 187.0 / 91.0
    SIGMA = 36.0 / 91.0
    power_law = s ** (-TAU)
    exponential = np.exp(-B * s * np.abs(f - 0.5) ** (1.0 / SIGMA))
    return A * power_law * exponential


def compute_molecule_fraction(df: pd.DataFrame, total_molecules: int = 2048) -> float:
    """f = (sum cluster molecules across frames) / (total_molecules * n_frames)."""
    if df.empty:
        return 0.0
    n_frames = df["frame"].nunique()
    total_cluster_molecules = df["size"].sum()
    f = total_cluster_molecules / (total_molecules * n_frames)
    return float(f)


def fit_size_distribution(
    df: pd.DataFrame,
    label: str,
    total_molecules: int = 2048,
    min_size: int = 2,
    max_size: int | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Fit theoretical size distribution to observed multimer data (counts per frame)."""
    if df.empty:
        return {}, np.array([]), np.array([]), np.array([])

    df_multi = df[df["size"] >= min_size].copy()
    if df_multi.empty:
        return {}, np.array([]), np.array([]), np.array([])

    f = compute_molecule_fraction(df_multi, total_molecules)

    sizes_raw = df_multi["size"].astype(int).values
    n_frames = df["frame"].nunique()

    if max_size is None:
        max_size = int(sizes_raw.max())

    bins = np.arange(min_size - 0.5, max_size + 1.5, 1)
    counts, edges = np.histogram(sizes_raw, bins=bins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    observed = counts / n_frames

    mask = observed > 0
    sizes_fit = centers[mask]
    observed_fit = observed[mask]

    if len(sizes_fit) < 3:
        print(f"    Warning: {label} has too few data points for fitting")
        return {}, centers, observed, np.zeros_like(observed)

    A_init = observed_fit.max() * (sizes_fit[np.argmax(observed_fit)] ** (187/91))
    B_init = 0.1

    try:
        def fit_func(s, A, B):
            return theoretical_size_distribution(s, A, B, f)

        popt, pcov = curve_fit(
            fit_func,
            sizes_fit,
            observed_fit,
            p0=[A_init, B_init],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=10000,
        )

        A_fit, B_fit = popt
        fitted = theoretical_size_distribution(centers, A_fit, B_fit, f)

        residuals = observed_fit - theoretical_size_distribution(sizes_fit, A_fit, B_fit, f)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((observed_fit - observed_fit.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        perr = np.sqrt(np.diag(pcov))

        params = {
            "label": label,
            "f": f,
            "A": A_fit,
            "A_err": perr[0],
            "B": B_fit,
            "B_err": perr[1],
            "tau": 187/91,
            "sigma": 36/91,
            "r_squared": r_squared,
            "n_frames": n_frames,
            "n_points": len(sizes_fit),
            "size_range": f"{int(sizes_fit.min())}-{int(sizes_fit.max())}",
        }

        print(f"    {label}: f={f:.4f}, A={A_fit:.2e}±{perr[0]:.2e}, B={B_fit:.4f}±{perr[1]:.4f}, R²={r_squared:.4f}")
        return params, centers, observed, fitted

    except Exception as e:
        print(f"    Warning: {label} fitting failed: {e}")
        return {}, centers, observed, np.zeros_like(observed)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_population_evolution(
    pop_freeze: pd.DataFrame,
    pop_melt: pd.DataFrame,
    time_scale: float,
    outdir: Path,
    use_frames: bool = False,
):
    """Plot evolution of multimer population and molecules in coordination."""
    xlabel = "Frame" if use_frames else "Time [ns]"

    # --- Figure 1: number of multimers over time ---
    if not pop_freeze.empty or not pop_melt.empty:
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        if not pop_freeze.empty:
            x_f = pop_freeze["frame"].values if use_frames else pop_freeze["frame"].values * time_scale
            y_f = moving_average(pop_freeze["n_multimers"].values.astype(float), window=15)
            ax1.plot(x_f, y_f, label="Freeze", color=FREEZE_COLOR, linewidth=2.5, alpha=0.85)

        if not pop_melt.empty:
            x_m = pop_melt["frame"].values if use_frames else pop_melt["frame"].values * time_scale
            y_m = moving_average(pop_melt["n_multimers"].values.astype(float), window=15)
            ax1.plot(x_m, y_m, label="Melt", color=MELT_COLOR, linewidth=2.5, alpha=0.85)

        ax1.set_xlabel(xlabel, fontsize=20)
        ax1.set_ylabel("Multimers count (N_multi)", fontsize=20)
        ax1.tick_params(axis="both", labelsize=20)
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        maybe_scientific_y(ax1, threshold=100.0)
        ax1.legend(loc="best", fontsize=14)

        plt.tight_layout()
        plt.savefig(outdir / "population_n_multimers.png")
        plt.close(fig1)

    # --- Figure 2: molecules in coordination over time ---
    if not pop_freeze.empty or not pop_melt.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        if not pop_freeze.empty:
            x_f = pop_freeze["frame"].values if use_frames else pop_freeze["frame"].values * time_scale
            y_f = moving_average(pop_freeze["multimer_molecules"].values.astype(float), window=15)
            ax2.plot(x_f, y_f, label="Freeze", color=FREEZE_COLOR, linewidth=2.5, alpha=0.85)

        if not pop_melt.empty:
            x_m = pop_melt["frame"].values if use_frames else pop_melt["frame"].values * time_scale
            y_m = moving_average(pop_melt["multimer_molecules"].values.astype(float), window=15)
            ax2.plot(x_m, y_m, label="Melt", color=MELT_COLOR, linewidth=2.5, alpha=0.85)

        ax2.set_xlabel(xlabel, fontsize=20)
        ax2.set_ylabel("Multimer molecules (M)", fontsize=20)
        ax2.tick_params(axis="both", labelsize=20)
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        maybe_scientific_y(ax2, threshold=100.0)
        ax2.legend(loc="best", fontsize=14)

        plt.tight_layout()
        plt.savefig(outdir / "population_multimer_molecules.png")
        plt.close(fig2)


def plot_singleton_count_timeseries(
    pop_freeze: pd.DataFrame,
    pop_melt: pd.DataFrame,
    time_scale: float,
    outdir: Path,
    use_frames: bool = False,
):
    """Simple trend plot of singleton molecules S(t) for Freeze vs Melt (same layout as population_n_multimers)."""
    xlabel = "Frame" if use_frames else "Time [ns]"
    fig, ax = plt.subplots(figsize=(10, 6))

    if not pop_freeze.empty:
        x_f = pop_freeze["frame"].values if use_frames else pop_freeze["frame"].values * time_scale
        y_f = moving_average(pop_freeze["n_singletons"].values.astype(float), window=15)
        ax.plot(x_f, y_f, label="Freeze", color=FREEZE_COLOR, linewidth=2.5, alpha=0.85)

    if not pop_melt.empty:
        x_m = pop_melt["frame"].values if use_frames else pop_melt["frame"].values * time_scale
        y_m = moving_average(pop_melt["n_singletons"].values.astype(float), window=15)
        ax.plot(x_m, y_m, label="Melt", color=MELT_COLOR, linewidth=2.5, alpha=0.85)

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel("Singleton molecules (S)", fontsize=20)
    ax.tick_params(axis="both", labelsize=20)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    maybe_scientific_y(ax, threshold=100.0)
    ax.legend(loc="best", fontsize=14)

    plt.tight_layout()
    plt.savefig(outdir / "population_n_singletons.png")
    plt.close(fig)


def plot_size_distribution(df_freeze: pd.DataFrame, df_melt: pd.DataFrame, outdir: Path):
    """Plot cluster size distribution in multiple variants."""

    def _plot_variant(include_singletons: bool, use_log: bool, suffix: str, title_addon: str):
        fig, ax = plt.subplots(figsize=(8, 6))

        for df, label, color in [(df_freeze, "Freeze", FREEZE_COLOR),
                                 (df_melt, "Melt", MELT_COLOR)]:
            if df.empty:
                continue

            if include_singletons:
                df_filtered = df.copy()
                min_size = 1
            else:
                df_filtered = df[df["size"] >= 2]
                min_size = 2

            if df_filtered.empty:
                continue

            sizes = df_filtered["size"].astype(int).values
            max_size = int(sizes.max())

            bins = np.arange(min_size - 0.5, max_size + 1.5, 1)
            counts, edges = np.histogram(sizes, bins=bins)
            centers = 0.5 * (edges[1:] + edges[:-1])

            n_frames = max(df["frame"].nunique(), 1)
            y = counts / n_frames

            ax.step(centers, y, where="mid", label=label, color=color, linewidth=2.5, alpha=0.85)

        ax.set_xlabel("Cluster size (s)")
        ax.set_ylabel("Clusters per frame N(s)")
        ax.set_title(f"Cluster size distribution{title_addon}")

        if ax.lines and use_log:
            ax.set_yscale("log")
            ymin = min([line.get_ydata()[line.get_ydata() > 0].min()
                        for line in ax.lines if (line.get_ydata() > 0).any()])
            ax.set_ylim(bottom=ymin * 0.5)

        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(outdir / f"size_distribution{suffix}.png")
        plt.close()

    _plot_variant(False, False, "_multimers_linear", " (multimers, linear)")
    _plot_variant(False, True,  "_multimers_log",    " (multimers, log)")
    _plot_variant(True,  False, "_all_linear",       " (all, linear)")
    _plot_variant(True,  True,  "_all_log",          " (all, log)")

    print("    → Created 4 size distribution variants")


def plot_size_timeseries(
    ts_freeze: pd.DataFrame,
    ts_melt: pd.DataFrame,
    time_scale: float,
    outdir: Path,
    max_sizes: int = 11,
    use_frames: bool = False,
):
    """Plot time evolution of grouped cluster counts with smoothing."""
    def _aggregate_and_plot(ts: pd.DataFrame, label: str):
        if ts.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 6))

        x = ts.index.values if use_frames else ts.index.values * time_scale
        xlabel = "Frame" if use_frames else "Time [ns]"

        size_cols = sorted(int(c) for c in ts.columns)

        def sum_range(low, high=None):
            if high is None:
                cols = [s for s in size_cols if s >= low]
            else:
                cols = [s for s in size_cols if low <= s <= high]
            if not cols:
                return np.zeros(len(ts), dtype=float)
            return ts[cols].sum(axis=1).astype(float).values

        series = {
            "2":   ts[2].astype(float).values if 2 in ts.columns else np.zeros(len(ts), float),
            "3-4": sum_range(3, 4),
            "5-8": sum_range(5, 8),
            ">9":  sum_range(9, None),
        }

        plot_order = ["2", "5-8", "3-4", ">9"]
        colors = [plt.cm.viridis(i / max(len(plot_order) - 1, 1)) for i in range(len(plot_order))]

        for i, k in enumerate(plot_order):
            y = moving_average(series[k], window=15)
            ax.plot(x, y, label=k, color=colors[i], linewidth=2.2, alpha=0.95)

        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel("Clusters count", fontsize=20)
        ax.tick_params(axis="both", labelsize=20)
        leg = ax.legend(title="Cluster size:", ncol=2, loc="best", framealpha=0.9, fontsize=14)
        if leg and leg.get_title():
            leg.get_title().set_fontsize(14)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        maybe_scientific_y(ax, threshold=100.0)

        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(outdir / f"{label.lower()}_size_timeseries.png")
        plt.close()

    _aggregate_and_plot(ts_freeze, "Freeze")
    _aggregate_and_plot(ts_melt, "Melt")


def plot_individual_size_timeseries(
    ts: pd.DataFrame,
    label: str,
    time_scale: float,
    outdir: Path,
    use_frames: bool = False,
):
    """Create individual plots for each cluster size time series."""
    if ts.empty:
        return

    individual_dir = outdir / f"{label.lower()}_individual_sizes"
    individual_dir.mkdir(exist_ok=True)

    x = ts.index.values if use_frames else ts.index.values * time_scale
    xlabel = "Frame" if use_frames else "Time [ns]"
    sizes = sorted([int(c) for c in ts.columns])

    cmap = plt.cm.viridis

    for i, size in enumerate(sizes):
        if size not in ts.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))

        counts = ts[size].values.astype(float)
        color = cmap(i / max(len(sizes) - 1, 1))

        ax.plot(x, counts, color=color, linewidth=2.5, alpha=0.95, label=f"s={size}")
        ax.fill_between(x, counts, alpha=0.25, color=color)

        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel("Clusters count", fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_title(f"{label}: size {size}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        maybe_scientific_y(ax, threshold=100.0)

        mean_count = counts.mean()
        max_count = counts.max()
        nonzero_frac = (counts > 0).sum() / len(counts) * 100
        stats_text = f"Mean: {mean_count:.1f}\nMax: {max_count:.0f}\nOcc.: {nonzero_frac:.1f}%"
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )
        _despine(ax)

        plt.tight_layout()
        plt.savefig(individual_dir / f"size_{size:02d}.png")
        plt.close()

    print(f"    → Created {len(sizes)} individual plots in {individual_dir.name}/")


def plot_spectral_analysis(
    freqs_freeze: np.ndarray,
    psd_freeze: pd.DataFrame,
    freqs_melt: np.ndarray,
    psd_melt: pd.DataFrame,
    outdir: Path,
    max_sizes: int = 8,
    freq_unit: str = "1/ns",
):
    """Plot power spectral density for different cluster sizes (normalized)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, freqs, psd, label in [
        (ax1, freqs_freeze, psd_freeze, "Freeze"),
        (ax2, freqs_melt, psd_melt, "Melt"),
    ]:
        if psd.empty or len(freqs) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"{label}: No data")
            _despine(ax)
            continue

        cols = sorted(psd.columns, key=lambda x: int(float(x.split("_")[1])))[:max_sizes]
        cmap = plt.cm.viridis
        for i, col in enumerate(cols):
            size = int(float(col.split("_")[1]))
            color_val = cmap(i / max(len(cols) - 1, 1))
            psd_vals = psd[col].values
            if len(psd_vals) >= 11:
                psd_smooth = signal.savgol_filter(psd_vals, window_length=11, polyorder=2)
            else:
                psd_smooth = psd_vals
            ax.plot(freqs, psd_smooth, label=f"s={size}", color=color_val, linewidth=2.0, alpha=0.9)

        ax.set_xlabel(f"Frequency [{freq_unit}]")
        ax.set_ylabel("PSD (norm.)")
        ax.set_title(f"{label}: spectrum")
        ax.legend(loc="best", ncol=2, framealpha=0.9)
        ax.set_xlim(left=0)
        _despine(ax)

    plt.tight_layout()
    plt.savefig(outdir / "spectral_analysis.png")
    plt.close()


def plot_spectral_heatmap(
    freqs: np.ndarray,
    psd: pd.DataFrame,
    label: str,
    outdir: Path,
    freq_unit: str = "1/ns",
):
    """Create heatmap of PSD across sizes and frequencies (normalized)."""
    if psd.empty or len(freqs) == 0:
        return

    sizes = sorted([int(float(col.split("_")[1])) for col in psd.columns])
    matrix = np.zeros((len(sizes), len(freqs)))

    for i, size in enumerate(sizes):
        col_name = f"size_{size}"
        if col_name in psd.columns:
            matrix[i, :] = psd[col_name].values
        else:
            col_name_float = f"size_{float(size)}"
            if col_name_float in psd.columns:
                matrix[i, :] = psd[col_name_float].values

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[freqs[0], freqs[-1], sizes[0] - 0.5, sizes[-1] + 0.5],
    )

    ax.set_xlabel(f"Frequency [{freq_unit}]")
    ax.set_ylabel("Cluster size (s)")
    ax.set_title(f"{label}: PSD heatmap")
    _despine(ax)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PSD (norm.)")

    plt.tight_layout()
    plt.savefig(outdir / f"{label.lower()}_spectral_heatmap.png")
    plt.close()


def plot_shape_analysis(df_freeze: pd.DataFrame, df_melt: pd.DataFrame, outdir: Path):
    """Analyze and plot shape anisotropy vs cluster size."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for df, label, color in [(df_freeze, "Freeze", FREEZE_COLOR), (df_melt, "Melt", MELT_COLOR)]:
        if df.empty:
            continue
        df_multi = df[df["size"] >= 2].copy()
        if df_multi.empty:
            continue

        df_multi["size"] = pd.to_numeric(df_multi["size"], errors="coerce").round().astype("Int64")
        df_multi["anisotropy"] = df_multi.apply(compute_shape_anisotropy, axis=1)
        df_multi = df_multi.dropna(subset=["anisotropy"])
        if df_multi.empty:
            continue

        df_multi[["frame", "cluster_id", "size", "anisotropy"]].to_csv(
            outdir / f"{label.lower()}_anisotropy.csv", index=False
        )

        ax.scatter(
            df_multi["size"],
            df_multi["anisotropy"],
            alpha=0.3,
            s=20,
            color=color,
            label=label,
            edgecolors="none",
        )

        size_bins = np.arange(2, int(df_multi["size"].max()) + 1)
        means = []
        for s in size_bins:
            subset = df_multi[df_multi["size"] == s]["anisotropy"]
            means.append(subset.mean() if len(subset) > 0 else np.nan)
        ax.plot(size_bins, means, color=color, linewidth=2.5, label=f"{label} mean", marker="o", markersize=5)

    ax.set_xlabel("Cluster size (s)")
    ax.set_ylabel("Anisotropy κ²")
    ax.set_title("Shape vs size")
    ax.legend()
    ax.set_ylim(0, 1)
    _despine(ax)

    plt.tight_layout()
    plt.savefig(outdir / "shape_anisotropy.png")
    plt.close()


def plot_rg_scaling(df_freeze: pd.DataFrame, df_melt: pd.DataFrame, outdir: Path):
    """Plot radius of gyration vs cluster size (scaling relationship)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for df, label, color in [(df_freeze, "Freeze", FREEZE_COLOR), (df_melt, "Melt", MELT_COLOR)]:
        if df.empty:
            continue
        df_multi = df[df["size"] >= 2].copy()
        if df_multi.empty:
            continue

        df_multi["size"] = pd.to_numeric(df_multi["size"], errors="coerce").round().astype("Int64")
        ax.scatter(
            df_multi["size"],
            df_multi["rog"],
            alpha=0.25,
            s=15,
            color=color,
            label=label,
            edgecolors="none",
        )

        grp = df_multi.groupby("size", dropna=True)["rog"].median().dropna()
        if len(grp) > 3:
            sizes = grp.index.values.astype(float)
            rgs = grp.values.astype(float)
            valid = (sizes > 0) & (rgs > 0)
            sizes, rgs = sizes[valid], rgs[valid]
            if len(sizes) > 3:
                coeffs = np.polyfit(np.log(sizes), np.log(rgs), 1)
                nu = coeffs[0]
                size_range = np.linspace(sizes.min(), sizes.max(), 100)
                rg_fit = np.exp(coeffs[1]) * size_range ** nu
                ax.plot(size_range, rg_fit, color=color, linewidth=2, linestyle="--", label=f"{label} ν={nu:.2f}")

    ax.set_xlabel("Cluster size (s)")
    ax.set_ylabel("Rg")
    ax.set_title("Rg scaling")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    _despine(ax)

    plt.tight_layout()
    plt.savefig(outdir / "rg_scaling.png")
    plt.close()


def plot_fitted_size_distribution(
    params_freeze: dict,
    sizes_freeze: np.ndarray,
    observed_freeze: np.ndarray,
    fitted_freeze: np.ndarray,
    params_melt: dict,
    sizes_melt: np.ndarray,
    observed_melt: np.ndarray,
    fitted_melt: np.ndarray,
    outdir: Path,
):
    """Plot observed vs fitted size distributions (linear + log-log)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for sizes, observed, fitted, params, color, label in [
        (sizes_freeze, observed_freeze, fitted_freeze, params_freeze, FREEZE_COLOR, "Freeze"),
        (sizes_melt, observed_melt, fitted_melt, params_melt, MELT_COLOR, "Melt"),
    ]:
        if len(sizes) == 0 or not params:
            continue
        mask = observed > 0
        ax1.scatter(sizes[mask], observed[mask], alpha=0.6, s=50, color=color,
                    label=f"{label} obs", edgecolors="white", linewidths=0.5, zorder=3)

        mask_fit = fitted > 0
        ax1.plot(sizes[mask_fit], fitted[mask_fit], color=color, linewidth=2.5, linestyle="--",
                 label=f"{label} fit (R²={params.get('r_squared', 0):.3f})", alpha=0.85, zorder=2)

    ax1.set_xlabel("s")
    ax1.set_ylabel("N(s) per frame")
    ax1.set_title("Fit (linear)")
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    _despine(ax1)

    for sizes, observed, fitted, params, color, label in [
        (sizes_freeze, observed_freeze, fitted_freeze, params_freeze, FREEZE_COLOR, "Freeze"),
        (sizes_melt, observed_melt, fitted_melt, params_melt, MELT_COLOR, "Melt"),
    ]:
        if len(sizes) == 0 or not params:
            continue
        mask = observed > 0
        ax2.scatter(sizes[mask], observed[mask], alpha=0.6, s=50, color=color,
                    label=f"{label} obs", edgecolors="white", linewidths=0.5, zorder=3)

        mask_fit = fitted > 0
        ax2.plot(sizes[mask_fit], fitted[mask_fit], color=color, linewidth=2.5, linestyle="--",
                 label=f"{label} fit", alpha=0.85, zorder=2)

    ax2.set_xlabel("s")
    ax2.set_ylabel("N(s) per frame")
    ax2.set_title("Fit (log-log)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(loc="best", framealpha=0.9)
    ax2.grid(True, alpha=0.3, which="both")
    _despine(ax2)

    text_lines = [
        "N(s;f)=A·s^(-τ)·exp[-B·s·|f-0.5|^(1/σ)]",
        f"τ={187/91:.3f}, σ={36/91:.3f}",
        "",
    ]
    if params_freeze:
        text_lines += [
            f"Freeze: f={params_freeze['f']:.4f}",
            f"  A={params_freeze['A']:.2e}±{params_freeze['A_err']:.2e}",
            f"  B={params_freeze['B']:.4f}±{params_freeze['B_err']:.4f}",
            "",
        ]
    if params_melt:
        text_lines += [
            f"Melt:   f={params_melt['f']:.4f}",
            f"  A={params_melt['A']:.2e}±{params_melt['A_err']:.2e}",
            f"  B={params_melt['B']:.4f}±{params_melt['B_err']:.4f}",
        ]

    fig.text(
        0.5, 0.01, "\n".join(text_lines),
        ha="center", va="bottom", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(outdir / "fitted_size_distribution.png", dpi=300)
    plt.close()

    print("    → Saved fitted size distribution plot")


def plot_singleton_vs_multimer_polarization(
    pop_freeze: pd.DataFrame,
    pop_melt: pd.DataFrame,
    time_scale: float,
    outdir: Path,
    use_frames: bool = False,
):
    """P(t) = (M - S) / (M + S), with M=multimer_molecules, S=n_singletons."""
    xlabel = "Frame" if use_frames else "Time [ns]"
    fig, ax = plt.subplots(figsize=(10, 6))

    def _compute_P(pop: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = pop["frame"].values if use_frames else pop["frame"].values * time_scale
        M = pop["multimer_molecules"].values.astype(float)
        S = pop["n_singletons"].values.astype(float)
        denom = M + S
        P = np.full_like(denom, np.nan, dtype=float)
        ok = denom > 0
        P[ok] = (M[ok] - S[ok]) / denom[ok]
        return x, P

    if not pop_freeze.empty:
        x_f, P_f = _compute_P(pop_freeze)
        P_f = moving_average(P_f, window=15)
        ax.plot(x_f, P_f, label="Freeze", color="#1743D5", linewidth=2.5, alpha=0.85)

    if not pop_melt.empty:
        x_m, P_m = _compute_P(pop_melt)
        P_m = moving_average(P_m, window=15)
        ax.plot(x_m, P_m, label="Melt", color="#D11E39", linewidth=2.5, alpha=0.85)

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel("P", fontsize=22)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0.0, linewidth=1.5, alpha=0.35)
    ax.legend(loc="best", fontsize=14)

    plt.tight_layout()
    plt.savefig(outdir / "population_polarization.png")
    plt.close(fig)


def plot_coordination_intensity_product(
    pop_freeze: pd.DataFrame,
    pop_melt: pd.DataFrame,
    time_scale: float,
    outdir: Path,
    use_frames: bool = False,
):
    """I(t) = N_multi(t) · M(t)."""
    xlabel = "Frame" if use_frames else "Time [ns]"
    fig, ax = plt.subplots(figsize=(10, 6))

    if not pop_freeze.empty:
        x_f = pop_freeze["frame"].values if use_frames else pop_freeze["frame"].values * time_scale
        y_f = (pop_freeze["n_multimers"].values.astype(float) *
               pop_freeze["multimer_molecules"].values.astype(float))
        y_f = moving_average(y_f, window=15)
        ax.plot(x_f, y_f, label="Freeze", color=FREEZE_COLOR, linewidth=2.5, alpha=0.85)

    if not pop_melt.empty:
        x_m = pop_melt["frame"].values if use_frames else pop_melt["frame"].values * time_scale
        y_m = (pop_melt["n_multimers"].values.astype(float) *
               pop_melt["multimer_molecules"].values.astype(float))
        y_m = moving_average(y_m, window=15)
        ax.plot(x_m, y_m, label="Melt", color=MELT_COLOR, linewidth=2.5, alpha=0.85)

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel("I=N_multi·M", fontsize=20)
    ax.tick_params(axis="both", labelsize=20)
    maybe_scientific_y(ax, threshold=100.0)
    ax.legend(loc="best", fontsize=14)

    plt.tight_layout()
    plt.savefig(outdir / "population_coordination_intensity.png")
    plt.close(fig)


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze coordinated molecular clusters from MD simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--freeze_dir", type=Path, default=Path("./freeze_4"),
                        help="Directory with freeze_cluster_data.* files")
    parser.add_argument("--melt_dir", type=Path, default=Path("./melt_4"),
                        help="Directory with melt_cluster_data.* files")
    parser.add_argument("--time_scale", type=float, default=0.1,
                        help="Frame to time conversion factor in ns/frame (e.g., 0.1 for 0.1 ns/frame)")
    parser.add_argument("--out_dir", type=Path, default=Path("./cluster_analysis"),
                        help="Output directory for plots and data")
    parser.add_argument("--max_display_sizes", type=int, default=10,
                        help="Maximum cluster sizes to show in detailed plots")
    parser.add_argument("--use_frames", action="store_true",
                        help="Use frame numbers instead of time for x-axis")
    parser.add_argument("--total_molecules", type=int, default=2048,
                        help="Total number of molecules in the system (for f in fit)")

    args = parser.parse_args()

    set_plot_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CLUSTER COORDINATION ANALYSIS")
    print("=" * 70 + "\n")

    print("Loading datasets...")
    df_freeze = load_dataset(args.freeze_dir, "freeze", "freeze")
    df_melt = load_dataset(args.melt_dir, "melt", "melt")

    if df_freeze.empty and df_melt.empty:
        print("\nERROR: No data found in either directory!")
        return

    print("\nAnalyzing multimer populations...")
    pop_freeze = analyze_multimer_population(df_freeze)
    pop_melt = analyze_multimer_population(df_melt)

    if not pop_freeze.empty:
        pop_freeze.to_csv(args.out_dir / "freeze_population.csv", index=False)
        print(f"  Freeze: {len(pop_freeze)} frames, avg {pop_freeze['n_multimers'].mean():.1f} multimers/frame")

    if not pop_melt.empty:
        pop_melt.to_csv(args.out_dir / "melt_population.csv", index=False)
        print(f"  Melt: {len(pop_melt)} frames, avg {pop_melt['n_multimers'].mean():.1f} multimers/frame")

    print("\nGenerating visualizations...")

    print("  [1/9] Population evolution...")
    plot_population_evolution(pop_freeze, pop_melt, args.time_scale, args.out_dir, args.use_frames)

    print("  [NEW] Singleton trend + polarization + coordination intensity...")
    plot_singleton_count_timeseries(pop_freeze, pop_melt, args.time_scale, args.out_dir, args.use_frames)
    plot_singleton_vs_multimer_polarization(pop_freeze, pop_melt, args.time_scale, args.out_dir, args.use_frames)
    plot_coordination_intensity_product(pop_freeze, pop_melt, args.time_scale, args.out_dir, args.use_frames)

    print("  [2/9] Size distributions...")
    plot_size_distribution(df_freeze, df_melt, args.out_dir)

    print("  [3/9] Size time series...")
    ts_freeze = compute_size_timeseries(df_freeze)
    ts_melt = compute_size_timeseries(df_melt)

    if not ts_freeze.empty:
        ts_freeze.to_csv(args.out_dir / "freeze_size_timeseries.csv")
    if not ts_melt.empty:
        ts_melt.to_csv(args.out_dir / "melt_size_timeseries.csv")

    plot_size_timeseries(ts_freeze, ts_melt, args.time_scale, args.out_dir, args.max_display_sizes, args.use_frames)

    print("  [4/9] Individual size time series...")
    plot_individual_size_timeseries(ts_freeze, "Freeze", args.time_scale, args.out_dir, args.use_frames)
    plot_individual_size_timeseries(ts_melt, "Melt", args.time_scale, args.out_dir, args.use_frames)

    print("  [5/9] Spectral analysis (FFT)...")
    freqs_f, psd_f, psd_f_raw = compute_spectral_analysis(ts_freeze, args.time_scale, dt_override=args.use_frames)
    freqs_m, psd_m, psd_m_raw = compute_spectral_analysis(ts_melt, args.time_scale, dt_override=args.use_frames)

    freq_unit = "1/frame" if args.use_frames else "1/ns"

    if not psd_f.empty:
        psd_f.to_csv(args.out_dir / "freeze_psd.csv")
        psd_f_raw.to_csv(args.out_dir / "freeze_psd_raw.csv")
        plot_spectral_heatmap(freqs_f, psd_f, "Freeze", args.out_dir, freq_unit=freq_unit)

    if not psd_m.empty:
        psd_m.to_csv(args.out_dir / "melt_psd.csv")
        psd_m_raw.to_csv(args.out_dir / "melt_psd_raw.csv")
        plot_spectral_heatmap(freqs_m, psd_m, "Melt", args.out_dir, freq_unit=freq_unit)

    plot_spectral_analysis(freqs_f, psd_f, freqs_m, psd_m, args.out_dir,
                           args.max_display_sizes, freq_unit=freq_unit)

    print("  [6/9] Identifying dominant frequencies...")
    dom_f = identify_dominant_frequencies(freqs_f, psd_f, threshold=0.03)
    dom_m = identify_dominant_frequencies(freqs_m, psd_m, threshold=0.03)

    if not dom_f.empty:
        dom_f.to_csv(args.out_dir / "freeze_dominant_frequencies.csv", index=False)
        print(f"    Freeze: {len(dom_f)} sizes with dominant frequencies")

    if not dom_m.empty:
        dom_m.to_csv(args.out_dir / "melt_dominant_frequencies.csv", index=False)
        print(f"    Melt: {len(dom_m)} sizes with dominant frequencies")

    print("  [7/9] Shape anisotropy...")
    plot_shape_analysis(df_freeze, df_melt, args.out_dir)

    print("  [8/9] Rg scaling analysis...")
    plot_rg_scaling(df_freeze, df_melt, args.out_dir)

    print("  [9/9] Fitting theoretical size distribution...")
    params_f, sizes_f, obs_f, fit_f = fit_size_distribution(df_freeze, "Freeze", total_molecules=args.total_molecules)
    params_m, sizes_m, obs_m, fit_m = fit_size_distribution(df_melt, "Melt", total_molecules=args.total_molecules)

    fit_params_list = []
    if params_f:
        fit_params_list.append(params_f)
    if params_m:
        fit_params_list.append(params_m)

    if fit_params_list:
        fit_df = pd.DataFrame(fit_params_list)
        fit_df.to_csv(args.out_dir / "fitted_distribution_parameters.csv", index=False)
        print("    → Saved fit parameters to fitted_distribution_parameters.csv")

    plot_fitted_size_distribution(
        params_f, sizes_f, obs_f, fit_f,
        params_m, sizes_m, obs_m, fit_m,
        args.out_dir
    )

    print("  [NEW] Ratio plots (raw + smoothed)...")
    plot_state_ratio_timeseries(pop_freeze, pop_melt, args.time_scale, args.out_dir,
                               use_frames=args.use_frames, smooth=False)
    plot_state_ratio_timeseries(pop_freeze, pop_melt, args.time_scale, args.out_dir,
                               use_frames=args.use_frames, smooth=True, window=15)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {args.out_dir.resolve()}")
    print("\nKey outputs:")
    print("  - population_n_multimers.png")
    print("  - population_multimer_molecules.png")
    print("  - population_n_singletons.png")
    print("  - population_polarization.png")
    print("  - population_coordination_intensity.png")
    print("  - size_distribution_*.png (4 variants)")
    print("  - *_size_timeseries.png")
    print("  - *_individual_sizes/size_*.png")
    print("  - spectral_analysis.png")
    print("  - *_spectral_heatmap.png")
    print("  - shape_anisotropy.png")
    print("  - rg_scaling.png")
    print("  - fitted_size_distribution.png")
    print("  - ratio_*.png (+ smooth variants)")
    print("  - *.csv (populations, time series, PSD raw+norm, anisotropy, peaks, fit params)")

    if not dom_f.empty or not dom_m.empty:
        print("\nDominant coordination frequencies detected:")
        if not dom_f.empty:
            print(f"  Freeze: {dom_f['dominant_freq'].min():.4f} - {dom_f['dominant_freq'].max():.4f} [{freq_unit}]")
        if not dom_m.empty:
            print(f"  Melt:   {dom_m['dominant_freq'].min():.4f} - {dom_m['dominant_freq'].max():.4f} [{freq_unit}]")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()

