import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import os
import sys

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

n = 11  # number of components
subset_size = 1000
datafile = "imbalance_greedy_select_data.npz"

max_datafile = "imbalance_greedy_select_max_data.npz"

# --- Check for cache file ---
if os.path.exists(max_datafile):
    print(f"[📁] Found {max_datafile}, loading and plotting cached results.")
    cached = np.load(max_datafile, allow_pickle=True)
    selected_max = cached['selected_max']
    max_imbalances = cached['max_imbalances']
    selected_max_labels = cached['selected_max_labels']
    try:
        import matplotlib.pyplot as plt
        n = len(selected_max_labels)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(max_imbalances)+1), max_imbalances, '-o', c='crimson')
        plt.xlabel("Selection step")
        plt.ylabel("Maximum imbalance (A→B, B=all)")
        plt.title("Greedy selection: maximum imbalance vs step")
        plt.xticks(range(1, n+1), selected_max_labels, rotation=45)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("greedy_max_imbalance_progression.png", dpi=300)
        plt.show()
        print("[📈] Plot saved: greedy_max_imbalance_progression.png")
    except Exception as e:
        print(f"[⚠️] Could not plot: {e}")
    print("[✔️] Done.")
    sys.exit(0)   # skip rest of code



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
data_subsets = [data[i*subset_size:(i+1)*subset_size] for i in range(n_subsets)]
print(f"   [✔️] Split data into {n_subsets} subsets of {subset_size} samples each.")

print("\n[INFO] Starting greedy MAX selection...")

selected_max = []
max_imbalances = []
selected_max_labels = []
all_components = list(range(n))
remaining_max = list(range(n))

for step in range(n):
    print(f"   [STEP {step+1}] Selecting next component for MAX imbalance...")
    worst_imb = -np.inf
    worst_candidate = None
    for candidate in remaining_max:
        candidate_set = selected_max + [candidate]
        a2b_all = []
        # Loop over all data subsets
        for subset in data_subsets:
            try:
                d = MetricComparisons(subset, maxk=subset.shape[0] - 1)
                # A: candidate_set, B: all components
                a2b, _ = d.return_inf_imb_two_selected_coords(candidate_set, all_components)
                a2b_all.append(a2b)
            except Exception as e:
                print(f"      [⚠️] Subset: {e}")
                continue
        mean_imb = np.mean(a2b_all) if a2b_all else np.nan
        print(f"      Try add comp {candidate}: mean imbalance = {mean_imb:.5f}")
        if mean_imb > worst_imb:
            worst_imb = mean_imb
            worst_candidate = candidate

    if worst_candidate is None:
        print("   [❌ ERROR] No valid candidate found at this step.")
        break

    selected_max.append(worst_candidate)
    remaining_max.remove(worst_candidate)
    max_imbalances.append(worst_imb)
    selected_max_labels.append(str(worst_candidate))
    print(f"   [✔️] Selected comp {worst_candidate} (max imbalance: {worst_imb:.5f})\n")

print("[INFO] Greedy MAX selection complete!")
print(f"Selected order (max): {selected_max}")
print(f"Max imbalance at each step: {max_imbalances}")

# Save results for later analysis/plotting
np.savez("imbalance_greedy_select_max_data.npz", selected_max=selected_max, max_imbalances=max_imbalances, selected_max_labels=selected_max_labels)
print(f"[💾] Saved MAX selection and imbalance trace to imbalance_greedy_select_max_data.npz")

# Optional: plot the imbalance progression (MAX)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(max_imbalances)+1), max_imbalances, '-o', c='crimson')
    plt.xlabel("Selection step")
    plt.ylabel("Maximum imbalance (A→B, B=all)")
    plt.title("Greedy selection: maximum imbalance vs step")
    plt.xticks(range(1, n+1), selected_max_labels, rotation=45)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("greedy_max_imbalance_progression.png", dpi=300)
    plt.show()
    print("[📈] Plot saved: greedy_max_imbalance_progression.png")
except Exception as e:
    print(f"[⚠️] Could not plot: {e}")

print("[✔️] Done.")
