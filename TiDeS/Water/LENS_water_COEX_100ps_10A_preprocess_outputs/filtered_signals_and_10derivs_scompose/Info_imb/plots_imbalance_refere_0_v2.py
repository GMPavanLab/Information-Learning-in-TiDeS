import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline

n = 11
imb_matrix_filename = "info_imbalance_matrix.npy"

if os.path.exists(imb_matrix_filename):
    print(f"[📁] Found {imb_matrix_filename}, using it for plotting.")
    imb_matrix = np.load(imb_matrix_filename)

    # Include 0→0 and 0←0
    a2b_means = imb_matrix[0, :n]
    b2a_means = imb_matrix[:n, 0]
    
    x = np.arange(0, n)
    labels_a2b = [f"0→{i}" for i in x]
    labels_b2a = [f"{i}→0" for i in x]

else:
    print("[❌] info_imbalance_matrix.npy not found! Please run the main computation code first.")
    exit(1)

def smooth_plot(x, y, filename, color):
    from matplotlib import rcParams
    xnew = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=1)
    y_smooth = spl(xnew)

    # Create two vertically stacked axes
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(6, 5),
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05}
    )

    # Filter data for top and bottom axes
    top_mask = y_smooth > 0.6
    bottom_mask = y_smooth < 0.1

    # --- Top plot (0.6–1.0)
    if np.any(top_mask):
        ax_top.plot(xnew[top_mask], y_smooth[top_mask], color=color, linewidth=2)
    top_indices = np.where(y > 0.6)[0]
    ax_top.scatter(x[top_indices], y[top_indices], color=color, s=70, zorder=5)

    # --- Bottom plot (0.0–0.1)
    if np.any(bottom_mask):
        ax_bottom.plot(xnew[bottom_mask], y_smooth[bottom_mask], color=color, linewidth=2)
    bottom_indices = np.where(y < 0.1)[0]
    ax_bottom.scatter(x[bottom_indices], y[bottom_indices], color=color, s=70, zorder=5)

    # Y-limits
    ax_top.set_ylim(0.6 + 1e-4, 1.0)
    ax_bottom.set_ylim(0.0, 0.1 - 1e-4)

    # Remove ticks at break points
    from matplotlib.ticker import MultipleLocator

    ax_top.set_ylim(0.6001, 1.0)
    ax_bottom.set_ylim(0.0, 0.0999)

    ax_top.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_bottom.yaxis.set_major_locator(MultipleLocator(0.1))


    # Remove connecting spines and break lines
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()

    # Optional: diagonal break marks
    d = .5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle='none', color='k', mec='k', mew=1, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

    # Labeling
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([str(i) for i in x], fontsize=16)
    ax_bottom.set_xlabel("I", fontsize=16)
    ax_top.set_ylabel(r"$I_{\mathrm{IMB}}\ D^i \rightarrow D^0$", fontsize=18)
    ax_top.tick_params(axis='both', labelsize=16)
    ax_bottom.tick_params(axis='both', labelsize=16)

    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
    plt.savefig(filename, dpi=300)
    plt.close()





smooth_plot(x, a2b_means, "imbalance_0_to_others_v2.png", 'royalblue')
smooth_plot(x, b2a_means, "imbalance_others_to_0_v2.png", 'orange')
print("[💾] Plots saved.")

