import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deeptime.decomposition import TICA
from matplotlib.patches import Patch


def plot_tica_pie(eigenvector, comp_idx, out_folder, global_colors, global_labels, used_indices):
    abs_weights = np.abs(eigenvector)
    total = abs_weights.sum()
    if np.all(abs_weights / total < 0.001):
        print(f"[SKIP] Component {comp_idx+1}: all weights <0.1%, skipped.")
        return

    keep_mask = abs_weights / total > 0.001
    abs_weights = abs_weights[keep_mask]
    indices = np.arange(len(eigenvector))[keep_mask]

    for i in indices:
        used_indices.add(i)

    colors = [global_colors[i] for i in indices]
    labels = [global_labels[i] for i in indices]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _, autotexts = ax.pie(
        abs_weights,
        labels=None,
        autopct=lambda p: f"{p:.1f}%" if p > 4 else "",
        startangle=90,
        colors=colors,  # keep per-slice colors (we'll make faces transparent next)
        shadow=False,
        textprops={'fontsize': 36}
    )

    # Make slices empty with colored borders
    for i, w in enumerate(wedges):
        w.set_facecolor('none')
        w.set_edgecolor(colors[i])
        w.set_linewidth(4)

    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")

    ax.set_title("")
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, f"pie_comp_{comp_idx+1}.png"), dpi=300)
    plt.close(fig)



def save_combined_pie_legend(used_indices, global_colors, out_folder):
    handles = []
    labels = []
    for idx in sorted(used_indices):
        handles.append(Patch(facecolor=global_colors[idx], edgecolor='black'))
        labels.append(f"D{idx}".translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")))

    fig, ax = plt.subplots(figsize=(2, len(handles) * 0.4))
    ax.axis("off")
    ax.legend(handles, labels, title="Dimension", loc="center", frameon=True, fontsize=12, title_fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, "pie_legend_only.png"), dpi=300)
    plt.close(fig)


def plot_tica_influence(eigenvectors, out_folder):
    D, actual_dim = eigenvectors.shape
    x_vals, y_vals, magnitudes, colors = [], [], [], []
    for d in range(D):
        for comp in range(actual_dim):
            x_vals.append(d)
            y_vals.append(comp)
            magnitudes.append(abs(eigenvectors[d, comp]) * 100)
            colors.append(eigenvectors[d, comp])

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(x_vals, y_vals, s=magnitudes, c=colors, alpha=0.7)
    ax.set_xticks(range(D))
    ax.set_yticks(range(actual_dim))
    ax.set_xlabel("D", fontsize=16)
    ax.set_ylabel("tICA Component", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    fig.colorbar(scatter)
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, "tica_influence_bubble_plot.png"), dpi=300)
    plt.close(fig)


def main():
    npy_files = [f for f in os.listdir('.') if f.endswith('.npy')]
    if not npy_files:
        print("[ERROR] No .npy files found in current directory!")
        return

    print("Available .npy files:")
    for idx, f in enumerate(npy_files):
        print(f"[{idx}] {f}")

    choice_str = input("\nPick file #: ")
    try:
        choice_idx = int(choice_str)
        chosen_file = npy_files[choice_idx]
    except (ValueError, IndexError):
        print("[ERROR] Invalid choice. Exiting.")
        return

    base_name = os.path.splitext(chosen_file)[0]
    out_folder = f"tica_analysis_{base_name}"
    os.makedirs(out_folder, exist_ok=True)
    print(f"[INFO] Outputs will be saved in: {out_folder}")

    data = np.load(chosen_file)
    print(f"[INFO] Loaded data shape: {data.shape}")
    N, T, D = data.shape
    full_dataset = data.reshape(-1, D)

    lag = 5
    dim = 10
    factor = 0.1
    print(f"[INFO] Running TICA with lag={lag}, dim={dim} ...")

    tica = TICA(lagtime=lag, dim=dim)
    tica.fit(full_dataset, scaling='kinetic_map')
    model = tica.fetch_model()

    timescales = model.timescales() * factor
    eigenvectors = model.instantaneous_coefficients
    actual_dim = eigenvectors.shape[1]
    print("Timescales:", timescales)

    plot_tica_influence(eigenvectors, out_folder)

    x_vals = np.arange(len(timescales))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x_vals, timescales)
    ax.set_xticks(x_vals)
    ax.set_xlabel("Component #", fontsize=16)
    ax.set_ylabel("Relaxation Time (scaled)", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, "relaxation_times.png"), dpi=300)
    plt.close(fig)

    global_colors = plt.get_cmap('tab20')(np.linspace(0, 1, D))
    global_labels = [f"D{i}" for i in range(D)]
    used_indices = set()

    for comp_idx in range(actual_dim):
        ev = eigenvectors[:, comp_idx]
        idx_sorted = np.argsort(np.abs(ev))[::-1]
        sorted_data = np.column_stack([idx_sorted, ev[idx_sorted]])
        out_txt = os.path.join(out_folder, f"eigenvector_c{comp_idx+1}.txt")
        np.savetxt(out_txt, sorted_data)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(range(D), ev)
        ax.set_xlabel("D", fontsize=16)
        ax.set_ylabel("Weight", fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(out_folder, f"bar_comp_{comp_idx+1}.png"), dpi=300)
        plt.close(fig)

        plot_tica_pie(ev, comp_idx, out_folder, global_colors, global_labels, used_indices)

        ev_2d = ev.reshape(-1, 1)
        max_abs_val = np.abs(ev).max()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            ev_2d,
            cmap='jet',
            center=0,
            vmin=-max_abs_val,
            vmax=+max_abs_val,
            yticklabels=[f"D{i}" for i in range(D)],
            xticklabels=[f"C{comp_idx+1}"]
        )
        ax.set_ylabel("D", fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join(out_folder, f"heatmap_comp_{comp_idx+1}.png"), dpi=300)
        plt.close(fig)

    save_combined_pie_legend(used_indices, global_colors, out_folder)

    tica_soap = np.array([tica.transform(atom) for atom in data])
    np.save(os.path.join(out_folder, f"tICA_{lag}.npy"), tica_soap)

    for comp_idx in range(actual_dim):
        component_data = tica_soap[..., comp_idx]
        np.save(os.path.join(out_folder, f"tICA_comp_{comp_idx+1}_{lag}.npy"), component_data)

    max_abs_val = np.abs(eigenvectors).max()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        eigenvectors,
        cmap='jet',
        center=0,
        vmin=-max_abs_val,
        vmax=max_abs_val,
        xticklabels=[f"C{i+1}" for i in range(actual_dim)],
        yticklabels=[f"D{i}" for i in range(D)]
    )
    ax.set_xlabel("tICA Component", fontsize=16)
    ax.set_ylabel("D", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(out_folder, "eigenvectors_heatmap.png"), dpi=300)
    plt.close(fig)

    print("[INFO] All done.")


if __name__ == '__main__':
    main()
