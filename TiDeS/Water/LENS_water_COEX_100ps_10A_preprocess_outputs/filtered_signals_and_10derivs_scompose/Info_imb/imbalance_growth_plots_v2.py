import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import warnings
import sys
import os

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

warnings.filterwarnings("ignore")
np.seterr(all='ignore')

n = 11
datafile = "imbalance_grow_sets_data.npz"

def smooth_plot(x, y, filename, color):
    x = x + 1  # shift to 1-based indexing
    xnew = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=1)
    y_smooth = spl(xnew)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(xnew, y_smooth, color=color, linewidth=2)
    ax.scatter(x, y, color=color, s=70, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=16)
    ax.set_xlabel("N", fontsize=16)
    ax.set_ylabel(r"$I_{\mathrm{IMB}}(D^{(n)} \rightarrow D^{(n-1)})$", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)

    y_max = np.nanmax(y)
    ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)

    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"   [💾] Plot saved: {filename}")



if os.path.exists(datafile):
    print(f"[📁] Found {datafile}, loading data and skipping computation.")
    saved = np.load(datafile, allow_pickle=True)
    a2b_means = saved['a2b_means']
    b2a_means = saved['b2a_means']
    labels = saved['labels']
    x = np.arange(len(labels))
else:
    print("[1/5] Loading components...")
    components = []
    for i in range(n):
        fname = f'component_{i}_1D.npy'
        try:
            c = np.load(fname).flatten()
            print(f"   [✔️] Loaded: {fname} (shape: {c.shape})")
            components.append(c)
        except Exception as e:
            print(f"   [❌ ERROR] Failed to load {fname}: {e}")
            sys.exit(1)

    if len(set(len(c) for c in components)) != 1:
        print("[❌ ERROR] All components must have the same length.")
        sys.exit(1)

    print(f"[2/5] Stacking data and making subsets...")
    components = [normalize(c) for c in components]
    data = np.column_stack(components)
    subset_size = 1000
    n_subsets = data.shape[0] // subset_size
    if n_subsets == 0:
        print("[❌ ERROR] Subset size too big for data length.")
        sys.exit(1)
    data_subsets = [data[i*subset_size:(i+1)*subset_size] for i in range(n_subsets)]
    print(f"   [✔️] Split data into {n_subsets} subsets of {subset_size} samples each.")

    # --- Build selected pairs ---
    print("[3/5] Preparing selected pairs...")
    selected_pairs = []
    labels = []
    for k in range(2, n+1):
        a = list(range(k))
        b = list(range(k-1))
        selected_pairs.append((a, b))
        a_str = ''.join(str(x) for x in a)
        b_str = ''.join(str(x) for x in b)
        labels.append(f"{a_str}→{b_str}")
    print(f"   [✔️] Prepared {len(selected_pairs)} pairs.")

    # --- Compute imbalance for all pairs ---
    print("[4/5] Computing information imbalance for each pair...")
    a2b_means = []
    b2a_means = []

    for idx, (a, b) in enumerate(selected_pairs):
        a2b_all = []
        b2a_all = []
        print(f"   [{idx+1:02d}/{len(selected_pairs)}] {labels[idx]}: ", end="", flush=True)
        for sidx, subset in enumerate(data_subsets):
            try:
                d = MetricComparisons(subset, maxk=subset.shape[0] - 1)
                a2b, b2a = d.return_inf_imb_two_selected_coords(a, b)
                a2b_all.append(a2b)
                b2a_all.append(b2a)
            except Exception as e:
                print(f"\n      [⚠️] Subset {sidx+1}: {e}")
                continue
        mean_a2b = np.mean(a2b_all) if a2b_all else np.nan
        mean_b2a = np.mean(b2a_all) if b2a_all else np.nan
        a2b_means.append(mean_a2b)
        b2a_means.append(mean_b2a)
        print(f"A→B={mean_a2b:.4f} | B→A={mean_b2a:.4f}")

    x = np.arange(len(labels))
    np.savez(datafile, a2b_means=a2b_means, b2a_means=b2a_means, labels=labels)
    print(f"[💾] Saved results to {datafile}")

print("[5/5] Generating plots...")
# For A→B plot (labels as is)
#smooth_plot(
#    x, a2b_means, labels,
#    "Info Imbalance: Growing sets (A→B)",
#    "imbalance_grow_A_to_B.png", 'royalblue'
#)

# For B→A plot (swap labels, e.g. "01→012")
#swapped_labels = [lbl.split("→")[-1] + "→" + lbl.split("→")[0] for lbl in labels]
#smooth_plot(
#    x, b2a_means, swapped_labels,
#    "Info Imbalance: Growing sets (B→A)",
#    "imbalance_grow_B_to_A_v2.png", 'orange'
#)

smooth_plot(x, a2b_means, "imbalance_grow_A_to_B_v2.png", 'royalblue')
smooth_plot(x, b2a_means, "imbalance_grow_B_to_A_v2.png", 'orange')

print("[✔️] All done.")

