import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from numpy.fft import rfft, rfftfreq


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def prompt_str(message, default=None, allow_blank=True):
    """Prompt the user for a string and safely handle defaults."""
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{message}{suffix}: ").strip()
    if raw == "":
        if default is not None:
            return str(default)
        if allow_blank:
            return ""
    return raw


def prompt_int(message, default=None, min_value=None, max_value=None):
    """Prompt the user for an integer with optional bounds checking."""
    while True:
        raw = prompt_str(message, default=default)
        try:
            value = int(raw)
        except ValueError:
            print(f"[WARN] '{raw}' is not a valid integer. Please try again.")
            continue

        if min_value is not None and value < min_value:
            print(f"[WARN] Value must be >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"[WARN] Value must be <= {max_value}.")
            continue
        return value


def prompt_float(message, default=None, min_value=None, max_value=None):
    """Prompt the user for a float with optional bounds checking."""
    while True:
        raw = prompt_str(message, default=default)
        try:
            value = float(raw)
        except ValueError:
            print(f"[WARN] '{raw}' is not a valid number. Please try again.")
            continue

        if min_value is not None and value < min_value:
            print(f"[WARN] Value must be >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"[WARN] Value must be <= {max_value}.")
            continue
        return value


def prompt_yes_no(message, default=True):
    """Prompt the user for a yes/no answer and return a boolean."""
    default_text = "y" if default else "n"
    while True:
        raw = prompt_str(f"{message} [y/n]", default=default_text).lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("[WARN] Please answer with 'y' or 'n'.")


def make_dir(path):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------------
# Loading and validation
# -----------------------------------------------------------------------------

def load_npz_signals(npz_path):
    """Load a 2D signal array from an NPZ file and validate its contents."""
    print(f"[STEP] Loading NPZ file: {npz_path}")
    try:
        data = np.load(npz_path)
    except Exception as exc:
        print(f"[ERROR] Failed to load '{npz_path}': {exc}")
        sys.exit(1)

    if not hasattr(data, "files") or len(data.files) == 0:
        print("[ERROR] NPZ file contains no arrays.")
        sys.exit(1)

    selected_key = None
    if "arr_0" in data.files and np.asarray(data["arr_0"]).ndim == 2:
        selected_key = "arr_0"
    else:
        two_d_keys = [k for k in data.files if np.asarray(data[k]).ndim == 2]
        if len(two_d_keys) == 1:
            selected_key = two_d_keys[0]
        elif len(two_d_keys) > 1:
            print("[INFO] Multiple 2D arrays found in NPZ:")
            for idx, key in enumerate(two_d_keys):
                print(f"    {idx}: key='{key}', shape={np.asarray(data[key]).shape}")
            chosen_idx = prompt_int(
                "Select the array index to use",
                default=0,
                min_value=0,
                max_value=len(two_d_keys) - 1,
            )
            selected_key = two_d_keys[chosen_idx]

    if selected_key is None:
        print("[ERROR] Could not find a suitable 2D array inside the NPZ file.")
        for key in data.files:
            print(f"    key='{key}', shape={np.asarray(data[key]).shape}")
        sys.exit(1)

    signals = np.asarray(data[selected_key], dtype=float)
    if signals.ndim != 2:
        print(f"[ERROR] Selected array '{selected_key}' is not 2D. shape={signals.shape}")
        sys.exit(1)

    if not np.all(np.isfinite(signals)):
        bad_count = np.size(signals) - np.count_nonzero(np.isfinite(signals))
        print(f"[ERROR] Dataset contains {bad_count} non-finite values (NaN or Inf). Clean the data first.")
        sys.exit(1)

    n_atoms, n_frames = signals.shape
    if n_atoms < 1 or n_frames < 2:
        print(f"[ERROR] Dataset shape {signals.shape} is too small. Need at least (1, 2).")
        sys.exit(1)

    print(f"[INFO] Loaded key='{selected_key}' with shape={signals.shape}")
    print(
        "[INFO] Signal stats: "
        f"min={signals.min():.6g}, max={signals.max():.6g}, "
        f"mean={signals.mean():.6g}, std={signals.std():.6g}"
    )
    return signals, selected_key


# -----------------------------------------------------------------------------
# FFT and cutoff suggestion
# -----------------------------------------------------------------------------

def fft_all_signals(signals, dt_seconds, subtract_row_mean=True):
    """Compute a summed one-sided FFT magnitude over all signals."""
    print("[STEP] Computing summed FFT over all signals...")
    working = np.asarray(signals, dtype=float)
    if subtract_row_mean:
        working = working - working.mean(axis=1, keepdims=True)
        print("[INFO] Row-wise mean subtraction enabled for FFT suggestion.")
    else:
        print("[INFO] Row-wise mean subtraction disabled for FFT suggestion.")

    n_frames = working.shape[1]
    freq = rfftfreq(n_frames, d=dt_seconds)
    fft_vals = rfft(working, axis=1)
    total_mag = np.abs(fft_vals).sum(axis=0)

    if freq.size <= 1:
        print("[WARN] FFT result is too short to analyze.")
        return None, None

    return freq[1:], total_mag[1:]



def suggest_cutoff_by_amplitude(freq, mag, fraction=0.2):
    """Suggest a cutoff from the first spectral drop below a chosen amplitude fraction."""
    print(f"[STEP] Suggesting cutoff by amplitude threshold: fraction={fraction:.3f} of max magnitude")
    if freq is None or mag is None or len(freq) == 0:
        return None
    max_val = np.max(mag)
    threshold = fraction * max_val
    idx = np.where(mag < threshold)[0]
    if len(idx) == 0:
        suggested = freq[-1]
        print(f"[WARN] No amplitude drop below threshold found. Fallback cutoff={suggested:.6e} Hz")
    else:
        suggested = freq[idx[0]]
        print(f"[INFO] Amplitude-based cutoff suggestion={suggested:.6e} Hz")
    return suggested



def suggest_cutoff_by_energy(freq, mag, retained_energy=0.95):
    """Suggest a cutoff that retains a chosen fraction of cumulative spectral energy."""
    print(f"[STEP] Suggesting cutoff by cumulative spectral energy: retained_energy={retained_energy:.3f}")
    if freq is None or mag is None or len(freq) == 0:
        return None
    power = mag ** 2
    total_power = np.sum(power)
    if total_power <= 0:
        print("[WARN] Total spectral power is zero. Using highest frequency as fallback.")
        return freq[-1]
    cumulative = np.cumsum(power) / total_power
    idx = np.searchsorted(cumulative, retained_energy)
    idx = min(idx, len(freq) - 1)
    suggested = freq[idx]
    print(f"[INFO] Energy-based cutoff suggestion={suggested:.6e} Hz")
    return suggested



def plot_summed_fft(freq, total_mag, title, filename, log_scale=False):
    """Save a summed FFT plot for quick inspection of spectral content."""
    print(f"[STEP] Saving FFT plot -> {filename}")
    if freq is None or total_mag is None:
        print("[WARN] FFT plot skipped because FFT data is missing.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(freq / 1e9, total_mag, linewidth=1.0)
    plt.title(title)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Summed magnitude")
    if log_scale:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Filtering and trimming
# -----------------------------------------------------------------------------

def trim_edges(signals, frames_each_side):
    """Trim the same number of frames from both ends of a 2D signal array."""
    n_frames = signals.shape[1]
    if frames_each_side <= 0:
        return signals.copy(), 0
    if n_frames <= 2 * frames_each_side:
        print(
            f"[WARN] Cannot trim {frames_each_side} frames from both sides of length {n_frames}. "
            "No shared edge trim applied."
        )
        return signals.copy(), 0
    trimmed = signals[:, frames_each_side:n_frames - frames_each_side]
    print(f"[INFO] Applied shared edge trim of {frames_each_side} frames per side. New shape={trimmed.shape}")
    return trimmed, frames_each_side



def design_lowpass_sos(cutoff_hz, fs_hz, order):
    """Design a stable Butterworth low-pass filter in second-order-section form."""
    nyquist = 0.5 * fs_hz
    if cutoff_hz is None or cutoff_hz <= 0:
        print("[INFO] Non-positive cutoff received. Filtering will be skipped.")
        return None, None

    actual_cutoff = min(cutoff_hz, 0.9999 * nyquist)
    if cutoff_hz >= nyquist:
        print(f"[WARN] Requested cutoff {cutoff_hz:.6e} Hz is >= Nyquist {nyquist:.6e} Hz.")
        print(f"[WARN] Clamping cutoff to {actual_cutoff:.6e} Hz to keep the filter valid.")

    normalized_cutoff = actual_cutoff / nyquist
    print("[INFO] Butterworth filter design summary:")
    print(f"       sampling frequency = {fs_hz:.6e} Hz")
    print(f"       Nyquist frequency  = {nyquist:.6e} Hz")
    print(f"       requested cutoff   = {cutoff_hz:.6e} Hz")
    print(f"       actual cutoff used = {actual_cutoff:.6e} Hz")
    print(f"       normalized cutoff  = {normalized_cutoff:.6f}")
    print(f"       filter order       = {order}")

    sos = butter(order, normalized_cutoff, btype="low", analog=False, output="sos")
    return sos, actual_cutoff



def apply_lowpass_filter(signals, cutoff_hz, fs_hz, order=4):
    """Apply the low-pass filter row-wise while preserving signal phase with sosfiltfilt."""
    sos, actual_cutoff = design_lowpass_sos(cutoff_hz, fs_hz, order)
    if sos is None:
        return signals.copy(), None
    filtered = sosfiltfilt(sos, signals, axis=1)
    print(f"[INFO] Filtering complete. Output shape={filtered.shape}")
    return filtered, actual_cutoff


# -----------------------------------------------------------------------------
# Derivatives and alignment
# -----------------------------------------------------------------------------

def compute_derivatives(signals, num_derivatives):
    """Compute successive time derivatives using finite differences along frames."""
    derivs = [signals]
    current = signals
    for order in range(1, num_derivatives + 1):
        if current.shape[1] < 2:
            raise ValueError(
                f"Cannot compute derivative {order}: current signal length is {current.shape[1]}, need at least 2 frames."
            )
        current = np.diff(current, axis=1)
        derivs.append(current)
    return derivs



def trim_derivatives_to_common_length(deriv_list):
    """Trim all derivative arrays so every order shares one common final time length."""
    num_derivatives = len(deriv_list) - 1
    base_length = deriv_list[0].shape[1]
    final_length = base_length - num_derivatives
    if final_length <= 0:
        raise ValueError(
            f"Final aligned length would be {final_length}. Reduce the number of derivatives or provide more frames."
        )

    trimmed = []
    for k, arr in enumerate(deriv_list):
        remove_from_start = num_derivatives - k
        trimmed_arr = arr[:, remove_from_start:remove_from_start + final_length]
        trimmed.append(trimmed_arr)
    return trimmed, final_length


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_many_signals(signals, title, filename, color="blue"):
    """Plot all trajectories together to inspect spread, noise, and outliers."""
    plt.figure(figsize=(8, 5))
    for row in signals:
        plt.plot(row, linewidth=0.08, alpha=0.6, color=color)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()



def plot_mean_overlay(original_signals, filtered_signals, filename):
    """Overlay mean original and mean filtered trajectories for a quick global sanity check."""
    mean_original = np.mean(original_signals, axis=0)
    mean_filtered = np.mean(filtered_signals, axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(mean_original, linewidth=1.5, label="Mean original")
    plt.plot(mean_filtered, linewidth=1.5, label="Mean filtered")
    plt.title("Mean signal overlay")
    plt.xlabel("Frame")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()



def plot_single_atom_overlay(original_signals, filtered_signals, atom_idx, filename):
    """Overlay one selected atom before and after filtering to inspect local filter behavior."""
    n_atoms = original_signals.shape[0]
    if atom_idx < 0 or atom_idx >= n_atoms:
        print(f"[WARN] Requested atom index {atom_idx} is out of range. Using atom 0 instead.")
        atom_idx = 0

    plt.figure(figsize=(8, 5))
    plt.plot(original_signals[atom_idx], linewidth=1.2, label=f"Original atom {atom_idx}")
    plt.plot(filtered_signals[atom_idx], linewidth=1.2, label=f"Filtered atom {atom_idx}")
    plt.title(f"Single-atom overlay (atom {atom_idx})")
    plt.xlabel("Frame")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()



def plot_filter_residuals(original_signals, filtered_signals, filename):
    """Plot the residual signal removed by filtering to gauge the filter impact."""
    residual = original_signals - filtered_signals
    mean_residual = np.mean(residual, axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(mean_residual, linewidth=1.5)
    plt.title("Mean residual: original - filtered")
    plt.xlabel("Frame")
    plt.ylabel("Residual amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Dataset selection and component workspace setup
# -----------------------------------------------------------------------------

def list_3d_npy_files(search_dir):
    """List valid 3D NPY datasets inside a directory so the user only sees usable stack files."""
    print(f"[STEP] Scanning for 3D .npy datasets inside: {search_dir}")
    if not os.path.isdir(search_dir):
        print(f"[WARN] Search directory does not exist: {search_dir}")
        return []

    candidates = []
    for name in sorted(os.listdir(search_dir)):
        if not name.endswith('.npy'):
            continue
        full_path = os.path.join(search_dir, name)
        try:
            arr = np.load(full_path, mmap_mode='r')
            if getattr(arr, 'ndim', None) == 3:
                candidates.append((full_path, tuple(arr.shape), str(arr.dtype)))
        except Exception as exc:
            print(f"[WARN] Skipping '{full_path}' because it could not be inspected: {exc}")

    if not candidates:
        print('[WARN] No valid 3D .npy datasets were found.')
    else:
        print('[INFO] Available 3D datasets:')
        for idx, (full_path, shape, dtype_str) in enumerate(candidates):
            print(f"       [{idx}] {os.path.basename(full_path)} | shape={shape} | dtype={dtype_str}")
    return candidates



def choose_dataset_file(search_dir):
    """Ask the user which 3D dataset should be split into components for the next workflow stage."""
    datasets = list_3d_npy_files(search_dir)
    if not datasets:
        return None
    choice_idx = prompt_int(
        'Pick dataset index for component separation',
        default=0,
        min_value=0,
        max_value=len(datasets) - 1,
    )
    chosen_path, chosen_shape, _ = datasets[choice_idx]
    print(f"[INFO] Selected dataset: {chosen_path}")
    print(f"[INFO] Selected dataset shape: {chosen_shape}")
    return chosen_path



def split_selected_dataset(npy_path, workspace_dir):
    """Split a chosen 3D dataset along its last axis and save each component as a standalone 2D file."""
    print(f"[STEP] Loading dataset for component split: {npy_path}")
    data = np.load(npy_path)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D dataset, got shape {data.shape} from '{npy_path}'.")

    n_atoms, n_frames, n_components = data.shape
    print(
        '[INFO] Dataset split summary: '
        f'n_atoms={n_atoms}, n_frames={n_frames}, n_components={n_components}'
    )

    component_paths = []
    for comp_idx in range(n_components):
        component = data[..., comp_idx]
        component_path = os.path.join(workspace_dir, f'component_{comp_idx}.npy')
        np.save(component_path, component)
        component_paths.append(component_path)
        print(
            f"[INFO] Saved component_{comp_idx}.npy with shape={component.shape} -> {component_path}"
        )
    return component_paths, data.shape



def create_onion_workspace_folders(workspace_dir):
    """Create the three standard onion-analysis folders expected by the following processing stages."""
    folder_names = ['onion_c0', 'onion_c1', 'onion_c0_c1']
    created_paths = {}
    for folder_name in folder_names:
        folder_path = os.path.join(workspace_dir, folder_name)
        make_dir(folder_path)
        created_paths[folder_name] = folder_path
        print(f"[INFO] Prepared folder: {folder_path}")
    return created_paths



def prepare_dataset_component_workspace(npy_path):
    """Create the selected dataset workspace, split all components, and initialize the onion subfolders."""
    dataset_dir = os.path.dirname(npy_path) or '.'
    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    workspace_dir = os.path.join(dataset_dir, f'{base_name}_scompose')
    make_dir(workspace_dir)
    print(f"[INFO] Dataset workspace folder: {workspace_dir}")

    component_paths, dataset_shape = split_selected_dataset(npy_path, workspace_dir)
    onion_paths = create_onion_workspace_folders(workspace_dir)

    metadata = {
        'selected_dataset': os.path.abspath(npy_path),
        'dataset_shape': list(dataset_shape),
        'workspace_dir': os.path.abspath(workspace_dir),
        'component_files': [os.path.abspath(p) for p in component_paths],
        'onion_folders': {name: os.path.abspath(path) for name, path in onion_paths.items()},
    }
    with open(os.path.join(workspace_dir, 'workspace_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Saved workspace metadata -> {os.path.join(workspace_dir, 'workspace_metadata.json')}")

    return workspace_dir, component_paths, onion_paths



def run_dataset_handling_stage(default_search_dir):
    """Run the dataset-selection stage that prepares the component workspace for the next analysis steps."""
    print('[STEP] Starting dataset-handling stage...')
    search_dir = prompt_str('Directory containing the candidate .npy datasets', default=default_search_dir)
    chosen_dataset = choose_dataset_file(search_dir)
    if chosen_dataset is None:
        print('[WARN] Dataset-handling stage skipped because no valid 3D .npy files were available.')
        return None

    workspace_dir, component_paths, onion_paths = prepare_dataset_component_workspace(chosen_dataset)
    print('[INFO] Dataset-handling stage complete.')
    print(f'       workspace: {workspace_dir}')
    print(f'       components: {len(component_paths)} files created')
    print(f"       onion folders: {', '.join(sorted(onion_paths.keys()))}")
    return {
        'workspace_dir': workspace_dir,
        'component_paths': component_paths,
        'onion_paths': onion_paths,
        'chosen_dataset': chosen_dataset,
    }


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main():
    """Run the full interactive preprocessing, filtering, alignment, saving, and sanity-check workflow."""
    print("[STEP] Starting preprocessing and filtering workflow...")

    npz_file = prompt_str("Path to NPZ file", default="signals_dataset.npz")
    signals_loaded, selected_key = load_npz_signals(npz_file)

    out_prefix = prompt_str("Output prefix", default=os.path.splitext(os.path.basename(npz_file))[0])
    out_dir = prompt_str("Output directory", default=f"{out_prefix}_preprocess_outputs")
    make_dir(out_dir)
    plot_dir = os.path.join(out_dir, "plots")
    make_dir(plot_dir)

    n_atoms, total_frames = signals_loaded.shape
    print(f"[INFO] Dataset overview: n_atoms={n_atoms}, n_frames={total_frames}")

    skip_frames = prompt_int(
        "Frames to skip from the start",
        default=0,
        min_value=0,
        max_value=max(0, total_frames - 1),
    )
    signals_skipped = signals_loaded[:, skip_frames:]
    print(f"[INFO] After initial skip of {skip_frames} frames -> shape={signals_skipped.shape}")

    frame_res_ps = prompt_float("Frame resolution in ps", default=80.0, min_value=1e-15)
    dt_seconds = frame_res_ps * 1e-12
    fs_hz = 1.0 / dt_seconds
    nyquist_hz = 0.5 * fs_hz
    print(f"[INFO] Time resolution: dt={dt_seconds:.6e} s, fs={fs_hz:.6e} Hz, Nyquist={nyquist_hz:.6e} Hz")

    subtract_row_mean_fft = prompt_yes_no("Subtract each row mean before FFT-based cutoff suggestion?", default=True)
    freq_orig, mag_orig = fft_all_signals(signals_skipped, dt_seconds, subtract_row_mean=subtract_row_mean_fft)
    plot_summed_fft(
        freq_orig,
        mag_orig,
        title="Summed FFT of skipped signals",
        filename=os.path.join(plot_dir, f"{out_prefix}_fft_before_filter.png"),
        log_scale=False,
    )

    amplitude_fraction = prompt_float(
        "Amplitude-threshold fraction for cutoff suggestion",
        default=0.2,
        min_value=1e-12,
    )
    retained_energy = prompt_float(
        "Retained spectral energy fraction for cutoff suggestion",
        default=0.95,
        min_value=1e-6,
        max_value=0.999999,
    )
    suggested_cut_amp = suggest_cutoff_by_amplitude(freq_orig, mag_orig, fraction=amplitude_fraction)
    suggested_cut_energy = suggest_cutoff_by_energy(freq_orig, mag_orig, retained_energy=retained_energy)

    print("[INFO] Available cutoff suggestions:")
    print(f"       amplitude-based -> {suggested_cut_amp}")
    print(f"       energy-based    -> {suggested_cut_energy}")

    cutoff_mode = prompt_str(
        "Choose cutoff mode: 'energy', 'amplitude', 'manual', or 'none'",
        default="energy",
    ).strip().lower()

    if cutoff_mode == "none":
        cutoff_hz = 0.0
    elif cutoff_mode == "amplitude":
        cutoff_hz = suggested_cut_amp if suggested_cut_amp is not None else 0.0
    elif cutoff_mode == "energy":
        cutoff_hz = suggested_cut_energy if suggested_cut_energy is not None else 0.0
    elif cutoff_mode == "manual":
        cutoff_hz = prompt_float(
            "Enter manual cutoff frequency in Hz",
            default=suggested_cut_energy or suggested_cut_amp or 0.0,
            min_value=0.0,
        )
    else:
        print(f"[WARN] Unknown cutoff mode '{cutoff_mode}'. Falling back to energy-based suggestion.")
        cutoff_hz = suggested_cut_energy if suggested_cut_energy is not None else 0.0

    filter_order = prompt_int("Butterworth filter order", default=4, min_value=1)
    shared_edge_trim = prompt_int(
        "Frames to trim from BOTH ends after the filtering/alignment stage",
        default=10,
        min_value=0,
    )

    filtered_untrimmed, actual_cutoff_used = apply_lowpass_filter(signals_skipped, cutoff_hz, fs_hz, order=filter_order)

    original_edge_trimmed, applied_edge_trim = trim_edges(signals_skipped, shared_edge_trim)
    filtered_edge_trimmed, _ = trim_edges(filtered_untrimmed, applied_edge_trim)

    if actual_cutoff_used is not None:
        freq_filt, mag_filt = fft_all_signals(filtered_edge_trimmed, dt_seconds, subtract_row_mean=subtract_row_mean_fft)
        plot_summed_fft(
            freq_filt,
            mag_filt,
            title="Summed FFT of filtered signals",
            filename=os.path.join(plot_dir, f"{out_prefix}_fft_after_filter.png"),
            log_scale=False,
        )
    else:
        freq_filt, mag_filt = None, None
        print("[INFO] Post-filter FFT skipped because no valid filter was applied.")

    available_frames_for_derivatives = filtered_edge_trimmed.shape[1]
    max_allowed_derivatives = max(0, available_frames_for_derivatives - 1)
    default_num_derivatives = min(10, max_allowed_derivatives)
    num_derivatives = prompt_int(
        "Number of derivatives to compute",
        default=default_num_derivatives,
        min_value=0,
        max_value=max_allowed_derivatives,
    )

    if num_derivatives == 0:
        print("[INFO] No derivatives requested. Only D0 will be kept.")

    deriv_list = compute_derivatives(filtered_edge_trimmed, num_derivatives=num_derivatives)
    trimmed_derivs, final_length = trim_derivatives_to_common_length(deriv_list)
    filtered_with_derivs = np.stack(trimmed_derivs, axis=-1)
    print(f"[INFO] Filtered stack with derivatives shape={filtered_with_derivs.shape}")

    original_matched = original_edge_trimmed[:, num_derivatives:num_derivatives + final_length]
    if original_matched.shape[1] != final_length:
        raise RuntimeError(
            f"Original aligned shape {original_matched.shape} does not match final_length={final_length}."
        )
    print(f"[INFO] Original aligned shape={original_matched.shape}")

    derivative_only = (
        filtered_with_derivs[..., 1:]
        if num_derivatives > 0
        else np.empty((*filtered_with_derivs.shape[:2], 0), dtype=filtered_with_derivs.dtype)
    )
    original_3d = np.expand_dims(original_matched, axis=-1)

    original_signals_and_derivs = np.concatenate([original_3d, derivative_only], axis=-1)
    filtered_signals_and_derivs = filtered_with_derivs
    combined_all = np.concatenate(
        [
            original_3d,
            filtered_with_derivs[..., 0:1],
            derivative_only,
        ],
        axis=-1,
    )

    np.save(os.path.join(out_dir, "original_signals_and_10derivs.npy"), original_signals_and_derivs)
    np.save(os.path.join(out_dir, "filtered_signals_and_10derivs.npy"), filtered_signals_and_derivs)
    np.save(os.path.join(out_dir, "combined_original_filtered_derivs.npy"), combined_all)
    np.save(os.path.join(out_dir, "original_aligned_2d.npy"), original_matched)
    np.save(os.path.join(out_dir, "filtered_aligned_2d.npy"), filtered_with_derivs[..., 0])

    metadata = {
        "input_file": npz_file,
        "selected_key": selected_key,
        "original_shape": list(signals_loaded.shape),
        "skip_frames": int(skip_frames),
        "frame_resolution_ps": float(frame_res_ps),
        "sampling_frequency_hz": float(fs_hz),
        "nyquist_hz": float(nyquist_hz),
        "subtract_row_mean_for_fft": bool(subtract_row_mean_fft),
        "cutoff_mode": cutoff_mode,
        "requested_cutoff_hz": float(cutoff_hz),
        "actual_cutoff_used_hz": None if actual_cutoff_used is None else float(actual_cutoff_used),
        "filter_order": int(filter_order),
        "shared_edge_trim_each_side": int(applied_edge_trim),
        "num_derivatives": int(num_derivatives),
        "final_aligned_length": int(final_length),
        "output_shapes": {
            "original_signals_and_10derivs": list(original_signals_and_derivs.shape),
            "filtered_signals_and_10derivs": list(filtered_signals_and_derivs.shape),
            "combined_original_filtered_derivs": list(combined_all.shape),
        },
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    plot_many_signals(
        original_matched,
        "Original signals (aligned)",
        os.path.join(plot_dir, "plot_original_signals.png"),
    )
    plot_many_signals(
        filtered_with_derivs[..., 0],
        "Filtered signals (aligned, D0)",
        os.path.join(plot_dir, "plot_filtered_signals.png"),
    )
    plot_mean_overlay(
        original_matched,
        filtered_with_derivs[..., 0],
        os.path.join(plot_dir, "plot_mean_overlay_original_vs_filtered.png"),
    )
    plot_filter_residuals(
        original_matched,
        filtered_with_derivs[..., 0],
        os.path.join(plot_dir, "plot_mean_residual_original_minus_filtered.png"),
    )

    example_atom = prompt_int(
        "Atom index for single-atom original vs filtered sanity plot",
        default=0,
        min_value=0,
        max_value=max(0, original_matched.shape[0] - 1),
    )
    plot_single_atom_overlay(
        original_matched,
        filtered_with_derivs[..., 0],
        example_atom,
        os.path.join(plot_dir, f"plot_atom_{example_atom}_original_vs_filtered.png"),
    )

    for i in range(1, num_derivatives + 1):
        plot_many_signals(
            filtered_with_derivs[..., i],
            f"Derivative {i}",
            os.path.join(plot_dir, f"plot_derivative_{i}.png"),
        )

    print("\n[INFO] Saved outputs:")
    print(f"       {os.path.join(out_dir, 'original_signals_and_10derivs.npy')}")
    print(f"       {os.path.join(out_dir, 'filtered_signals_and_10derivs.npy')}")
    print(f"       {os.path.join(out_dir, 'combined_original_filtered_derivs.npy')}")
    print(f"       {os.path.join(out_dir, 'original_aligned_2d.npy')}")
    print(f"       {os.path.join(out_dir, 'filtered_aligned_2d.npy')}")
    print(f"       {os.path.join(out_dir, 'metadata.json')}")
    print(f"       plots in: {plot_dir}")

    continue_to_dataset_stage = prompt_yes_no(
        "Do you want to continue to the dataset-handling stage now?",
        default=True,
    )
    if continue_to_dataset_stage:
        run_dataset_handling_stage(out_dir)

    print("[INFO] Workflow complete.")


if __name__ == "__main__":
    main()
