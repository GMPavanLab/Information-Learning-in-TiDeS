import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import os
import sys

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

n = 11  # number of components
subset_size = 1000

# NEW DATAFILE NAME: does not overwrite the previous A -> B result
datafile = "imbalance_greedy_select_data_b2a.npz"

# ---- LOAD DATA ----
components = []
for i in range(n):
    fname = f'component_{i}_1D.npy'
    try:
        c = np.load(fname).flatten()
        print(f"   [✔️] Loaded: {fname} (shape: {c.shape})")
        components.append(normalize(c))
    except Exception as e:
        print(f"   [❌ ERROR] Failed to load {fname}: {e}")
        sys.exit(1)

if len(set(len(c) for c in components)) != 1:
    print("[❌ ERROR] All components must have the same length.")
    sys.exit(1)

data = np.column_stack(components)

n_subsets = data.shape[0] // subset_size
if n_subsets == 0:
    print("[❌ ERROR] Subset size too big for data length.")
    sys.exit(1)

data_subsets = [
    data[i * subset_size:(i + 1) * subset_size]
    for i in range(n_subsets)
]

print(f"   [✔️] Split data into {n_subsets} subsets of {subset_size} samples each.")

# --- GREEDY SELECTION ROUTINE ---
print("\n[INFO] Starting greedy selection using B -> A imbalance...")

if os.path.exists(datafile):
    print(f"[INFO] Found existing file: {datafile}")
    saved = np.load(datafile, allow_pickle=True)

    selected = list(saved["selected"])
    min_imbalances = list(saved["min_imbalances"])
    selected_labels = list(saved["selected_labels"])

    print("[✔️] Loaded previous greedy selection results.")
else:
    selected = []
    min_imbalances = []
    selected_labels = []

    remaining = list(range(n))
    all_components = list(range(n))

    for step in range(n):
        print(f"   [STEP {step + 1}] Selecting next component...")

        best_imb = np.inf
        best_candidate = None

        for candidate in remaining:
            candidate_set = selected + [candidate]

            b2a_all = []

            # Loop over all data subsets
            for subset in data_subsets:
                try:
                    d = MetricComparisons(subset, maxk=subset.shape[0] - 1)

                    # A: candidate_set
                    # B: all components
                    # We now keep B -> A instead of A -> B
                    _, b2a = d.return_inf_imb_two_selected_coords(
                        candidate_set,
                        all_components
                    )

                    b2a_all.append(b2a)

                except Exception as e:
                    print(f"      [⚠️] Subset: {e}")
                    continue

            mean_imb = np.mean(b2a_all) if b2a_all else np.nan

            print(f"      Try add comp {candidate}: mean B -> A imbalance = {mean_imb:.5f}")

            if mean_imb < best_imb:
                best_imb = mean_imb
                best_candidate = candidate

        if best_candidate is None:
            print("   [❌ ERROR] No valid candidate found at this step.")
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        min_imbalances.append(best_imb)
        selected_labels.append(str(best_candidate))

        print(f"   [✔️] Selected comp {best_candidate} (min B -> A imbalance: {best_imb:.5f})\n")

    print("[INFO] Greedy selection complete!")
    print(f"Selected order: {selected}")
    print(f"Min B -> A imbalance at each step: {min_imbalances}")

    # Save results for later analysis/plotting
    np.savez(
        datafile,
        selected=selected,
        min_imbalances=min_imbalances,
        selected_labels=selected_labels
    )

    print(f"[💾] Saved B -> A selection and imbalance trace to {datafile}")

# ---- PLOT ----
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

    ax.plot(
        range(0, len(min_imbalances)),
        min_imbalances,
        '-o',
        c='royalblue'
    )

    ax.set_xlabel("N", fontsize=17)

    ax.set_ylabel(
        r"$I_{\mathrm{IMB}}(D^{(0-10)} \rightarrow D^{N})$",
        fontsize=17
    )

    ax.set_xticks(range(0, len(min_imbalances)))
    ax.set_xticklabels([str(i) for i in range(0, len(min_imbalances))])

    ax.tick_params(axis='both', labelsize=15)

    plt.savefig("greedy_min_imbalance_progression_b2a.png", dpi=300)
    plt.show()

    print("[📈] Plot saved: greedy_min_imbalance_progression_b2a.png")

except Exception as e:
    print(f"[⚠️] Could not plot: {e}")

print("[✔️] Done.")
