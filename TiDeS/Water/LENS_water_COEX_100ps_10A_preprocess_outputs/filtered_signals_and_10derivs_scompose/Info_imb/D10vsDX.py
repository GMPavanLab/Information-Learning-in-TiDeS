import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import matplotlib.pyplot as plt
import sys

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

n = 11
subset_size = 1000

# ---- LOAD DATA ----
components = []
for i in range(n):
    fname = f"component_{i}_1D.npy"
    try:
        c = np.load(fname).flatten()
        print(f"[✔️] Loaded: {fname} (shape: {c.shape})")
        components.append(normalize(c))
    except Exception as e:
        print(f"[❌ ERROR] Failed to load {fname}: {e}")
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
print(f"[✔️] Split data into {n_subsets} subsets of {subset_size} samples each.")

# ---- COMPUTE IMBALANCES ----
all_components = list(range(n))

comp_to_all_means = []
all_to_comp_means = []

print("\n[INFO] Computing imbalance for each single component vs full dataset...")

for comp in range(n):
    print(f"[INFO] Component {comp}")
    a2b_all = []
    b2a_all = []

    for subset in data_subsets:
        try:
            d = MetricComparisons(subset, maxk=subset.shape[0] - 1)

            # A = [comp], B = all components
            a2b, b2a = d.return_inf_imb_two_selected_coords([comp], all_components)

            a2b_all.append(a2b)   # comp -> full dataset
            b2a_all.append(b2a)   # full dataset -> comp

        except Exception as e:
            print(f"   [⚠️] Subset failed: {e}")
            continue

    mean_a2b = np.mean(a2b_all) if a2b_all else np.nan
    mean_b2a = np.mean(b2a_all) if b2a_all else np.nan

    comp_to_all_means.append(mean_a2b)
    all_to_comp_means.append(mean_b2a)

    print(f"   comp {comp} -> all : {mean_a2b:.5f}")
    print(f"   all -> comp {comp} : {mean_b2a:.5f}")

# ---- SAVE NUMERICAL RESULTS ----
np.savez(
    "single_component_vs_all_imbalance.npz",
    component_index=np.arange(n),
    comp_to_all=np.array(comp_to_all_means),
    all_to_comp=np.array(all_to_comp_means)
)
print("[💾] Saved numerical results to single_component_vs_all_imbalance.npz")

# ---- PLOT 1: comp X -> all ----
plt.figure(figsize=(8, 5))
plt.plot(range(n), comp_to_all_means, "-o", c="royalblue")
plt.xlabel("Component X")
plt.ylabel("Mean imbalance (comp X → full dataset)")
plt.title("Single component to full dataset")
plt.xticks(range(n))
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("imbalance_component_to_all.png", dpi=300)
plt.show()
print("[📈] Saved: imbalance_component_to_all.png")

# ---- PLOT 2: all -> comp X ----
plt.figure(figsize=(8, 5))
plt.plot(range(n), all_to_comp_means, "-o", c="crimson")
plt.xlabel("Component X")
plt.ylabel("Mean imbalance (full dataset → comp X)")
plt.title("Full dataset to single component")
plt.xticks(range(n))
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("imbalance_all_to_component.png", dpi=300)
plt.show()
print("[📈] Saved: imbalance_all_to_component.png")

print("[✔️] Done.")
