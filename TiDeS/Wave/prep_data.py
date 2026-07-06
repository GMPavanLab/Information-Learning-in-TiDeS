import sys
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

def suggest_cutoff(freq, mag, fraction=0.2):
    """
    Suggest a cutoff frequency where amplitude < fraction * max amplitude.
    If none is found, return the highest frequency in `freq`.
    """
    print(f"[STEP] Suggesting cutoff using fraction={fraction:.2f} of max amplitude...")
    max_val = np.max(mag)
    threshold = fraction * max_val
    idx = np.where(mag < threshold)[0]
    if len(idx) == 0:
        suggested = freq[-1]
        print(f"[WARN] No freq below threshold. Fallback to {suggested:.2e} Hz")
    else:
        suggested = freq[idx[0]]
        print(f"[INFO] Suggested cutoff={suggested:.2e} Hz (amp < {fraction*100:.0f}% of max)")
    return suggested

def plot_summed_fft(freq, total_mag, title, filename, log_scale=False):
    """
    Plot the summed FFT (freq vs. amplitude).

    Args:
        log_scale (bool): If True, y-axis is log-scale.
    """
    print(f"[STEP] Plotting summed FFT -> {filename}")
    if freq is None or total_mag is None:
        print("[WARN] Cannot plot FFT (freq or total_mag is None).")
        return

    freq_ghz = freq / 1e9
    plt.figure()
    plt.plot(freq_ghz, total_mag, linewidth=1.0, marker='o', markersize=3)
    plt.title(title)
    plt.xlabel("Frequency (GHz)")
    if log_scale:
        plt.yscale('log')
        plt.ylabel("Summed Magnitude (log-scale)")
    else:
        plt.ylabel("Summed Magnitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def fft_all_signals(signals, dt):
    """
    Compute FFT for all signals (row-wise) and sum magnitudes across rows.
    Returns:
        freq (1D array): Frequencies (positive side only)
        total_mag (1D array): Summed amplitude across all rows
    """
    print("[STEP] Computing FFT for signals array...")
    n_atoms, n_frames = signals.shape
    freq = None
    total_mag = None

    for row in signals:
        fft_vals = fft(row)
        f = fftfreq(n_frames, d=dt)
        pos_mask = f > 0  # skip zero freq
        row_mag = np.abs(fft_vals[pos_mask])
        if freq is None:
            freq = f[pos_mask]
            total_mag = row_mag
        else:
            total_mag += row_mag

    if freq is None or total_mag is None:
        print("[WARN] FFT not computed. Possibly no frames to process.")
    else:
        print(f"[INFO] FFT computed. freq size={freq.size}, total_mag size={total_mag.size}")
    return freq, total_mag

def remove_artifacts(signals, frames_to_remove=10):
    """
    Removes the first/last `frames_to_remove` frames if possible.
    If not enough frames, does nothing.
    """
    n_atoms, n_frames = signals.shape
    if n_frames <= 2 * frames_to_remove:
        print(f"[WARN] Not enough frames to remove {frames_to_remove} from both ends. Skipping.")
        return signals
    return signals[:, frames_to_remove:-frames_to_remove]

def butter_lowpass_filter(signals, cutoff, fs, order=4):
    """
    Apply a Butterworth low-pass filter using filtfilt to each row of `signals`.
    """
    nyq = 0.5 * fs
    if cutoff >= nyq:
        print(f"[WARN] cutoff={cutoff:.3g} >= Nyquist={nyq:.3g}, skipping filter.")
        return signals
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return np.array([filtfilt(b, a, row) for row in signals])

def compute_derivatives(signals, num_derivatives=10):
    """
    Given 'signals' (2D array), compute up to 'num_derivatives' times.
    D0 = signals
    D1..Dnum_derivatives = successive derivatives
    Returns a list [D0, D1, ..., Dnum_derivatives].
    """
    derivs = [signals]
    current = signals
    for _ in range(num_derivatives):
        d = np.diff(current, axis=1)
        derivs.append(d)
        current = d
    return derivs

def trim_for_consistent_length(deriv_list):
    """
    If deriv_list = [D0, D1, ..., Dn],
    - Dk has shape (n_atoms, T - k).
    We want them all to end up with shape (n_atoms, T - n).
    Each Dk must remove (n - k) frames from the start.
    Returns a list of trimmed arrays (each shape => (n_atoms, T-n)).
    """
    n_derivs = len(deriv_list) - 1  # e.g. if D0..D10 => n_derivs=10
    T = deriv_list[0].shape[1]      # length of D0
    final_length = T - n_derivs

    trimmed = []
    for k, arr in enumerate(deriv_list):
        remove_from_start = (n_derivs - k)  # how many frames to drop at start
        if remove_from_start < 0:
            remove_from_start = 0
        # check that we have enough frames to do this slicing
        if arr.shape[1] < remove_from_start + final_length:
            print(f"[WARN] Not enough frames in D{k} to trim properly. Skipping.")
            continue
        trimmed_k = arr[:, remove_from_start : remove_from_start + final_length]
        trimmed.append(trimmed_k)

    return trimmed

def main():
    # 1) Load NPZ
    npz_file = input("Path to NPZ file [default='signals_dataset.npz']: ").strip() or "signals_dataset.npz"
    data = np.load(npz_file)
    print("Available arrays:", list(data.keys()))
    if 'fi_matrix' not in data:
        print("[ERROR] NPZ must contain 'fi_matrix'. Exiting.")
        sys.exit(1)
    signals_loaded = data['fi_matrix']
    print(f"[INFO] Loaded signals shape={signals_loaded.shape}")

    # 2) Ask how many frames to skip
    skip_input = input("Frames to skip from the start? [0]: ").strip()
    skip_frames = int(skip_input) if skip_input else 0
    if skip_frames >= signals_loaded.shape[1]:
        print("[WARN] skip_frames too large, ignoring.")
        skip_frames = 0
    # => signals_skipped used for both final "original" data & for filtering
    signals_skipped = signals_loaded[:, skip_frames:]
    print(f"[INFO] After skipping {skip_frames} frames => shape={signals_skipped.shape}")

    # 3) Copy 1: for final "original" data (we remove artifacts on it right away)
    signals_original_final = remove_artifacts(signals_skipped, frames_to_remove=10)
    print(f"[INFO] 'signals_original_final' after artifact removal => shape={signals_original_final.shape}")

    # 4) Remove an extra 10 from the start (per your steps)
    if signals_original_final.shape[1] > 10:
        signals_original_final = signals_original_final[:, 10:]
        print(f"[INFO] Removed extra 10 from start => shape={signals_original_final.shape}")
    else:
        print("[WARN] Not enough frames for the extra 10. Skipping.")

    # 5) Copy 2: signals_for_filtering = just the skip version, no artifact removal yet
    signals_for_filtering = signals_skipped.copy()

    # ============== FILTERING ==============
    dt_input = input("Frame resolution in ps [80 ps]: ").strip()
    frame_res_ps = float(dt_input) if dt_input else 80.0
    dt = frame_res_ps * 1e-12
    fs = 1.0 / dt
    print(f"[INFO] dt={dt:.3g}s => fs={fs:.3g}Hz")


    # Compute FFT and suggest cutoff
    freq_orig, total_mag_orig = fft_all_signals(signals_for_filtering, dt)

    if freq_orig is not None and total_mag_orig is not None:
        plot_summed_fft(freq_orig, total_mag_orig, title="Summed FFT (GHz)", filename="summed_fft_original.png", log_scale=False)
        suggested_cut = suggest_cutoff(freq_orig, total_mag_orig, fraction=0.2)
        print(f"[INFO] Suggested cutoff is ~{suggested_cut:.2e} Hz")
    else:
        suggested_cut = None

    # Ask user for cutoff
    user_cut = input(f"Cutoff frequency in Hz [suggested={suggested_cut}, 'enter'=use suggestion, blank=none]: ").strip()
    if user_cut == "":
        cutoff = 0.0
    elif user_cut.lower() == "enter":
        cutoff = suggested_cut
    else:
        try:
            cutoff = float(user_cut)
        except:
            cutoff = suggested_cut
            print("[WARN] Invalid input. Using suggested cutoff.")

    # Filter if cutoff is set
    if cutoff and cutoff > 0:
        filtered_signals = butter_lowpass_filter(signals_for_filtering, cutoff, fs)
        filtered_signals = remove_artifacts(filtered_signals, frames_to_remove=10)
        print("[INFO] Filter applied.")
    else:
        filtered_signals = signals_for_filtering.copy()
        print("[INFO] No filtering applied.")

    # user_cut = input("Cutoff frequency in Hz [blank=no filtering]: ").strip()
    # cutoff = float(user_cut) if user_cut else 0.0

    # if cutoff > 0:
    #     # Filter signals_for_filtering
    #     filtered_signals = butter_lowpass_filter(signals_for_filtering, cutoff, fs)
    #     # THEN remove artifacts
    #     filtered_signals = remove_artifacts(filtered_signals, frames_to_remove=10)
    #     print(f"[INFO] Filtered signals => shape={filtered_signals.shape}")
    # else:
    #     print("[INFO] No filtering. Using signals_for_filtering as-is.")
    #     filtered_signals = signals_for_filtering.copy()

    # 6) Derivatives from the newly filtered signals
    print("[INFO] Computing 10 derivatives from filtered signals...")
    deriv_list = compute_derivatives(filtered_signals, num_derivatives=10)
    trimmed_derivs = trim_for_consistent_length(deriv_list)
    # => each shape => (n_atoms, final_length)
    # => stacked => (n_atoms, final_length, 1+10)
    filtered_with_derivs = np.stack(trimmed_derivs, axis=-1)
    print(f"[INFO] 'filtered_with_derivs' shape={filtered_with_derivs.shape}")
    # D0..D10 => D0=filtered, D1..D10=derivatives

    # final_length = # of frames after trimming
    final_length = filtered_with_derivs.shape[1]

    # ============== TRIM 'signals_original_final' TO MATCH LENGTH ==============
    # We want to ensure final arrays line up in time dimension
    if signals_original_final.shape[1] >= final_length:
        original_matched = signals_original_final[:, :final_length]
    else:
        print("[WARN] 'signals_original_final' is shorter than 'filtered_with_derivs'. Not trimming.")
        original_matched = signals_original_final
    print(f"[INFO] 'original_matched' shape={original_matched.shape}")

    # ============== BUILD THE OUTPUT DATASETS ==============

    # 1) original_signals_and_10derivs
    # We want: [D0=original signals] + [D1..D10 from the filtered derivs]
    # So we skip the first “channel” of the filtered deriv array (that is the filtered signals).
    derivative_only = filtered_with_derivs[..., 1:]  # shape => (n_atoms, final_length, 10)
    original_3d = np.expand_dims(original_matched, axis=-1)  # shape => (n_atoms, final_length, 1)
    # => combine => (n_atoms, final_length, 11)
    original_signals_and_10derivs = np.concatenate([original_3d, derivative_only], axis=-1)
    np.save("original_signals_and_10derivs.npy", original_signals_and_10derivs)
    print("[INFO] Saved 'original_signals_and_10derivs.npy' with shape:", original_signals_and_10derivs.shape)

    # 2) filtered_signals_and_10derivs
    # This is the entire stack from D0..D10 => shape (n_atoms, final_length, 11)
    np.save("filtered_signals_and_10derivs.npy", filtered_with_derivs)
    print("[INFO] Saved 'filtered_signals_and_10derivs.npy' with shape:", filtered_with_derivs.shape)

    # 3) combined_original_filtered_derivs
    # Combine them along axis=-1:
    # => channel 0: original
    # => channel 1: filtered
    # => channels 2..11: 10 derivatives
    combined_list = [
        np.expand_dims(original_matched, axis=-1),          # original => shape (n_atoms, final_length, 1)
        filtered_with_derivs[..., 0:1],                     # filtered => shape (n_atoms, final_length, 1)
        filtered_with_derivs[..., 1:]                       # derivatives => shape (n_atoms, final_length, 10)
    ]
    combined_all = np.concatenate(combined_list, axis=-1)    # => (n_atoms, final_length, 12)
    np.save("combined_original_filtered_derivs.npy", combined_all)
    print("[INFO] Saved 'combined_original_filtered_derivs.npy' with shape:", combined_all.shape)

    # 1) Plot original signals
    plt.figure()
    for row in original_matched:
        plt.plot(row, linewidth=0.05, alpha=0.6,color='blue')  # all same color, low line width
    plt.title("Original Signals")
    plt.xlabel("Frame")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_original_signals.png", dpi=300)
    plt.close()

    # 2) Plot filtered signals (D0)
    filtered_signals_2d = filtered_with_derivs[..., 0]  # shape => (n_atoms, final_length)
    plt.figure()
    for row in filtered_signals_2d:
        plt.plot(row, linewidth=0.05, alpha=0.6,color='blue')
    plt.title("Filtered Signals (D0)")
    plt.xlabel("Frame")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_filtered_signals.png", dpi=300)
    plt.close()

    # 3) Plot the 10 derivatives (D1..D10)
    for i in range(1, 11):
        derivative_2d = filtered_with_derivs[..., i]  # shape => (n_atoms, final_length)
        plt.figure()
        for row in derivative_2d:
            plt.plot(row, linewidth=0.05, alpha=0.6,color='blue')
        plt.title(f"Derivative {i}")
        plt.xlabel("Frame")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plot_derivative_{i}.png", dpi=300)
        plt.close()

    print("\n[INFO] Done.")

if __name__ == "__main__":
    main()
