import numpy as np
from dadapy.metric_comparisons import MetricComparisons
import matplotlib.pyplot as plt
import warnings
import sys

# Suppress warnings globally
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

import os

imb_file = "info_imbalance_matrix.npy"

# --- If matrix exists, load and skip computation ---
if os.path.exists(imb_file):
    print(f"[⚠️] Found {imb_file}, skipping computation.")
    imb_matrix = np.load(imb_file)
else:
    # === FULL COMPUTATION BLOCK ===
    n = 11

    # --- Load components ---
    components = []
    for i in range(n):
        fname = f'component_{i}_1D.npy'
        try:
            c = np.load(fname).flatten()
            print(f"[✔️] Loaded: {fname} (shape: {c.shape})")
            components.append(c)
        except Exception as e:
            print(f"[❌ ERROR] Failed to load {fname}: {e}")
            sys.exit(1)

    lengths = [len(c) for c in components]
    if len(set(lengths)) != 1:
        print(f"[❌ ERROR] All components must have the same length: {lengths}")
        sys.exit(1)
    print(f"[INFO] All components loaded, length: {lengths[0]}")

    # --- Stack data ---
    data = np.column_stack(components)  # shape (length, n)
    print(f"[INFO] Data shape: {data.shape}")

    # --- Make subsets ---
    subset_size = 1000
    n_subsets = data.shape[0] // subset_size
    if n_subsets == 0:
        print("[❌ ERROR] Subset size too big for data length.")
        sys.exit(1)
    data_subsets = [data[i*subset_size:(i+1)*subset_size] for i in range(n_subsets)]
    print(f"[INFO] Split data into {n_subsets} subsets of {subset_size} samples each.")

    # --- Compute imbalance matrix ---
    imb_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            imbal_values = []
            print(f"[...] Computing imbalance {i} → {j} ...", end="")
            for sidx, subset in enumerate(data_subsets):
                try:
                    d = MetricComparisons(subset, maxk=subset.shape[0] - 1)
                    im = d.return_inf_imb_two_selected_coords([i], [j])[0]
                    imbal_values.append(im)
                except Exception as e:
                    print(f"\n   [⚠️] Subset {sidx+1}: {e}")
                    continue
            if imbal_values:
                mean_val = np.mean(imbal_values)
                imb_matrix[i, j] = mean_val
                print(f" [OK] mean = {mean_val:.4f}")
            else:
                imb_matrix[i, j] = np.nan
                print(" [!] No valid data.")

    # --- Save result ---
    np.save(imb_file, imb_matrix)
    print(f"[💾] Saved matrix to {imb_file}")


# --- Main matrix plot ---
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(imb_matrix, cmap="magma", interpolation="nearest")

n = imb_matrix.shape[0]
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels([str(i) for i in range(n)], fontsize=24, rotation=45)
ax.set_yticklabels([str(i) for i in range(n)], fontsize=24)


#ax.tick_params(axis='both', labelsize=20)

plt.tight_layout()
plt.savefig("info_imbalance_matrix_magma.png", dpi=300)
plt.close()

# --- Separate colorbar plot ---
fig_cb, ax_cb = plt.subplots(figsize=(1.2, 4))
cbar = fig_cb.colorbar(im, cax=ax_cb, orientation='vertical')
cbar.set_label(r"$I_{\mathrm{IMB}}\ D^i \rightarrow D^j$", fontsize=16)
cbar.ax.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig("info_imbalance_matrix_colorbar.png", dpi=300)
plt.close()

print("[💾] Plots saved: matrix and colorbar.")

