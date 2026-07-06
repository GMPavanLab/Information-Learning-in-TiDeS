import numpy as np
import matplotlib.pyplot as plt
import os

# === Ask user for .npy files ===
print("\n📂 Current directory files:")
files = [f for f in os.listdir('.') if f.endswith('.npy')]
for i, f in enumerate(files):
    print(f"[{i}] {f}")

indices = input("\n🔢 Enter indices of imbalance .npy files to analyze means (comma separated): ").split(',')

selected_files = []
for idx in indices:
    try:
        selected_files.append(files[int(idx.strip())])
    except:
        print(f"[❌] Invalid index: {idx}")
        continue

if not selected_files:
    print("[⚠️] No valid files selected. Exiting.")
    exit()

# === Load data ===
imbalance_data = {}
for f in selected_files:
    try:
        arr = np.load(f)
        imbalance_data[f] = arr
        print(f"[✅] Loaded {f} with shape {arr.shape}")
    except Exception as e:
        print(f"[❌] Error loading {f}: {e}")

if not imbalance_data:
    print("[⚠️] No data loaded. Exiting.")
    exit()

# === Prepare means ===
mean_labels = []
mean_values = []
scatter_colors = []
scatter_markers = []
color_palette = plt.cm.tab20(np.linspace(0, 1, len(imbalance_data) * 2))

for idx, (name, arr) in enumerate(imbalance_data.items()):
    if arr.ndim == 2 and arr.shape[1] == 2:
        mean_labels.append(f"{name} A→B")
        mean_values.append(arr[:, 0].mean())
        scatter_colors.append(color_palette[idx * 2])
        scatter_markers.append('o')

        mean_labels.append(f"{name} B→A")
        mean_values.append(arr[:, 1].mean())
        scatter_colors.append(color_palette[idx * 2 + 1])
        scatter_markers.append('x')
    else:
        print(f"[⚠️] Skipped {name}, invalid shape: {arr.shape}")

# === Plot scatter ===
plt.figure(figsize=(10, 5))
for i, (x, y, c, m) in enumerate(zip(mean_labels, mean_values, scatter_colors, scatter_markers)):
    plt.scatter(i, y, color=c, marker=m, s=80, label=x)

plt.xticks(ticks=range(len(mean_labels)), labels=mean_labels, rotation=45, ha='right')
plt.ylabel("Mean Imbalance")
plt.title("Mean Information Imbalance per Series")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("selected_imbalance_means_scatter.png", dpi=300)
plt.show()

