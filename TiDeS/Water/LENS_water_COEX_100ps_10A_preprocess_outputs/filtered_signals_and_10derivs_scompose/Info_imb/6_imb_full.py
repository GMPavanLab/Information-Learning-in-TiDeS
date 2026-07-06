import numpy as np
import matplotlib.pyplot as plt
from dadapy.metric_comparisons import MetricComparisons
import os
import sys
import warnings

# Suppress UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

def safe_load(file):
    try:
        return np.load(file)
    except Exception as e:
        print(f"[❌ ERROR] Failed to load {file}: {e}")
        sys.exit(1)

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# === Load and normalize components ===
n_components = 6
components = []

for i in range(n_components):
    raw = safe_load(f"component_{i}_1D.npy")
    components.append(raw)

lengths = [len(c) for c in components]
if len(set(lengths)) != 1:
    print("[❌ ERROR] All components must have the same length.")
    sys.exit(1)

# === Plot densities before normalization ===
plt.figure(figsize=(10, 5))
raw_densities = []
for data in components:
    normed = normalize(data)
    density, bins = np.histogram(normed, bins=200, density=True)
    centers = (bins[:-1] + bins[1:]) / 2
    raw_densities.append((centers, density))
max_density = max(np.max(d) for _, d in raw_densities)
for i, (centers, density) in enumerate(raw_densities):
    plt.plot(centers, density / max_density, label=f"Component {i}", linewidth=2)
plt.title("Density Before Normalization (Y-Normalized, X-Normalized per Component)")
plt.xlabel("Normalized Value")
plt.ylabel("Normalized Density")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("components_density_before_normalization.png", dpi=300)
plt.show()

# === Normalize all ===
components = [normalize(c) for c in components]

# === Plot densities after normalization ===
plt.figure(figsize=(10, 5))
post_densities = []
for data in components:
    density, bins = np.histogram(data, bins=200, density=True)
    centers = (bins[:-1] + bins[1:]) / 2
    post_densities.append((centers, density))
max_post_density = max(np.max(d) for _, d in post_densities)
for i, (centers, density) in enumerate(post_densities):
    plt.plot(centers, density / max_post_density, label=f"Component {i}", linewidth=2)
plt.title("Density After Normalization (Y-Normalized)")
plt.xlabel("Value")
plt.ylabel("Normalized Density")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("components_density_after_normalization.png", dpi=300)
plt.show()

# === Create dataset ===
np.random.seed(1996)
nsamps = 500000
length = len(components[0])
if length < nsamps:
    print(f"[⚠️] Not enough samples, using full length ({length})")
    nsamps = length
indices = np.random.choice(length, nsamps, replace=False)
dataset = np.column_stack([c[indices] for c in components])
print(f"[📦] Dataset shape: {dataset.shape}")

# === Subsets ===
subset_size = 1000
n_subsets = dataset.shape[0] // subset_size
data_subsets = [dataset[i*subset_size:(i+1)*subset_size] for i in range(n_subsets)]
print(f"[🔄] Created {n_subsets} subsets of size {subset_size}")

# === Selected imbalance pairs ===
selected_pairs = [
    ([0], [0]),
    ([0,1], [0,1]),
    ([0], [1]),
    ([0], [2]),
    ([0], [3]),
    ([0], [4]),
    ([0], [5]),
    ([0, 1], [0]),
    ([0, 1, 2], [0]),
    ([0, 1, 2, 3], [0]),
    ([0, 1, 2, 3, 4], [0]),
    ([0, 1, 2, 3, 4, 5], [0]),
    ([0, 1], [0]),
    ([0, 1, 2], [0,1]),
    ([0, 1, 2, 3], [0,1,2]),
    ([0, 1, 2, 3, 4], [0,1,2,3]),
    ([0, 1, 2, 3, 4,5], [0,1,2,3,4]),
    ([1], [1]),
    ([1, 2], [1]),
    ([1, 2, 3], [1,2]),
    ([1, 2, 3, 4], [1,2,3]),
    ([1, 2, 3, 4,5], [1,2,3,4]),
    ([1, 2, 3, 4,5], [0,1,2,3,4]),
    ([0,1, 2, 3, 4,5], [0,1,2,3,4]),
    ([0,1, 2, 3, 4,5], [0,1,2,3,4,5])
]

imbalance_data = {}

for a, b in selected_pairs:
    label = f"{a}→{b}"
    fname = f"imbalance_{'_'.join(map(str, a))}to{'_'.join(map(str, b))}.npy"

    if os.path.exists(fname):
        print(f"[📁] Loading: {fname}")
        imbalance_data[label] = np.load(fname)
    else:
        print(f"[⚙️] Computing: {label}")
        imbalance_data[label] = []
        for i, subset in enumerate(data_subsets):
            try:
                d = MetricComparisons(subset, maxk=subset.shape[0] - 1)
                im = d.return_inf_imb_two_selected_coords(a, b)
                imbalance_data[label].append(im)
            except Exception as e:
                print(f"[⚠️] Subset {i+1} error: {e}")
                continue
        imbalance_data[label] = np.array(imbalance_data[label])
        np.save(fname, imbalance_data[label])
        print(f"[💾] Saved to: {fname}")

# === Plot per-subset time series ===
plt.figure(figsize=(12, 6))
color_palette = plt.cm.tab20(np.linspace(0, 1, len(imbalance_data) * 2))

for idx, (label, arr) in enumerate(imbalance_data.items()):
    if arr.shape[1] == 2:
        a_to_b = arr[:, 0]
        b_to_a = arr[:, 1]
        print(f"\n📊 {label} A→B: min={a_to_b.min():.4f}, max={a_to_b.max():.4f}, mean={a_to_b.mean():.4f}")
        print(f"📊 {label} B→A: min={b_to_a.min():.4f}, max={b_to_a.max():.4f}, mean={b_to_a.mean():.4f}")
        plt.plot(a_to_b, label=f"{label} A→B", color=color_palette[idx * 2], linewidth=1)
        plt.plot(b_to_a, label=f"{label} B→A", color=color_palette[idx * 2 + 1], linestyle="--", linewidth=1)
plt.title("Information Imbalance Across Subsets")
plt.xlabel("Subset Index")
plt.ylabel("Imbalance")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig("imbalance_plot.png", dpi=300)
plt.show()

# === Mean Imbalance Scatter Plot ===
mean_labels = []
mean_values = []
scatter_colors = []
scatter_markers = []

for idx, (label, arr) in enumerate(imbalance_data.items()):
    if arr.shape[1] == 2:
        mean_labels.append(f"{label} A→B")
        mean_values.append(arr[:, 0].mean())
        scatter_colors.append(color_palette[idx * 2])
        scatter_markers.append('o')
        mean_labels.append(f"{label} B→A")
        mean_values.append(arr[:, 1].mean())
        scatter_colors.append(color_palette[idx * 2 + 1])
        scatter_markers.append('x')

plt.figure(figsize=(10, 5))
for i, (x, y, c, m) in enumerate(zip(mean_labels, mean_values, scatter_colors, scatter_markers)):
    plt.scatter(i, y, color=c, marker=m, s=80, label=x)
plt.xticks(ticks=range(len(mean_labels)), labels=mean_labels, rotation=45, ha='right')
plt.ylabel("Mean Imbalance")
plt.title("Mean Information Imbalance per Series")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("imbalance_means_scatter.png", dpi=300)
plt.show()

