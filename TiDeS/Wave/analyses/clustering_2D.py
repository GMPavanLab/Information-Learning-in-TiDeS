import os
import numpy as np
import matplotlib.pyplot as plt
#from dynsight import onion
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import ScalarFormatter
# tropea_clustering is the user-provided package that handles multi-dim onion
from tropea_clustering import helpers, onion_multi
from tropea_clustering.plot import (
    plot_output_multi,
    plot_one_trj_multi,
    plot_medoids_multi,
    plot_state_populations,
    plot_sankey,
    color_trj_from_xyz,
    plot_time_res_analysis,
    plot_pop_fractions,
)

from matplotlib.colors import ListedColormap
COLORMAP = ListedColormap(["white","gray","green","blue","lightblue","red","orange", "yellow","cornflowerblue"])


#COLORMAP = "viridis"
from matplotlib.patches import Ellipse
from numpy.typing import NDArray
from pathlib import Path
from typing import List
from tropea_clustering._internal.main_2d import StateMulti

def plot_output_multi_v2(
    title: str,
    input_data: NDArray[np.float64],
    state_list: List[StateMulti],
    labels: NDArray[np.int64],
    delta_t: int,
    auto_lims: int = 1  # 0 = use default, 1 = auto-set data limits
):
    n_states = len(state_list) + 1
    tmp = plt.get_cmap(COLORMAP, n_states)

    if delta_t==83:
        print("preso")
        tmp= plt.get_cmap(ListedColormap(["purple","gray", "gray","yellow", "gray","red","darkgreen", "red","yellow"]), n_states)
    if delta_t==47:
        print("preso")
        tmp= plt.get_cmap(ListedColormap(["purple","gray", "yellow","green"]), n_states)    

    colors_from_cmap = tmp(np.arange(0, 1, 1 / n_states))
    #print(colors_from_cmap)
    colors_from_cmap[-1] = tmp(1.0)

    m_clean = input_data.transpose(1, 2, 0)
    n_windows = m_clean.shape[1] // delta_t
    tmp_labels = labels.reshape((m_clean.shape[0], n_windows))
    all_the_labels = np.repeat(tmp_labels, delta_t, axis=1)

    if m_clean.shape[2] == 3:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6))
        dir0 = [0, 0, 1]
        dir1 = [1, 2, 2]
        ax0 = [0, 0, 1]
        ax1 = [0, 1, 0]

        for k in range(3):
            d_0 = dir0[k]
            d_1 = dir1[k]
            a_0 = ax0[k]
            a_1 = ax1[k]

            id_max, id_min = 0, 0
            for idx, mol in enumerate(m_clean):
                if np.max(mol) == np.max(m_clean):
                    id_max = idx
                if np.min(mol) == np.min(m_clean):
                    id_min = idx

            line_w = 0.05
            max_t = all_the_labels.shape[1]
            m_resized = m_clean[:, :max_t, :]
            step = 5 if m_resized.size > 1000000 else 1

            for i, mol in enumerate(m_resized[::step]):
                ax[a_0][a_1].plot(
                    mol.T[d_0], mol.T[d_1],
                    c="black", lw=line_w, rasterized=True, zorder=0,
                )
                color_list = all_the_labels[i * step] + 1
                ax[a_0][a_1].scatter(
                    mol.T[d_0], mol.T[d_1],
                    c=color_list, cmap=COLORMAP,
                    vmin=0, vmax=n_states - 1,
                    s=0.5, rasterized=True,
                )

            # Min and max
            color_list = all_the_labels[id_min] + 1
            ax[a_0][a_1].plot(
                m_resized[id_min].T[d_0],
                m_resized[id_min].T[d_1],
                c="black", lw=line_w, rasterized=True, zorder=0,
            )
            ax[a_0][a_1].scatter(
                m_resized[id_min].T[d_0],
                m_resized[id_min].T[d_1],
                c=color_list, cmap=COLORMAP,
                vmin=0, vmax=n_states - 1,
                s=0.5, rasterized=True,
            )
            color_list = all_the_labels[id_max] + 1
            ax[a_0][a_1].plot(
                m_resized[id_max].T[d_0],
                m_resized[id_max].T[d_1],
                c="black", lw=line_w, rasterized=True, zorder=0,
            )
            ax[a_0][a_1].scatter(
                m_resized[id_max].T[d_0],
                m_resized[id_max].T[d_1],
                c=color_list, cmap=COLORMAP,
                vmin=0, vmax=n_states - 1,
                s=0.5, rasterized=True,
            )

            # Plot ellipses for states (only once, e.g. k == 0)
            if k == 0:
                for state in state_list:
                    att = state.get_attributes()
                    ellipse = Ellipse(
                        tuple(att["mean"]),
                        att["axis"][d_0],
                        att["axis"][d_1],
                        color="black",
                        fill=False,
                    )
                    ax[a_0][a_1].add_patch(ellipse)

            # Auto-adjust axis limits if requested
            if auto_lims == 1:
                xvals = m_resized[..., d_0]
                yvals = m_resized[..., d_1]
                x_min, x_max = xvals.min(), xvals.max()
                y_min, y_max = yvals.min(), yvals.max()
                ax[a_0][a_1].set_xlim(0, x_max + 0.1)
                ax[a_0][a_1].set_ylim(y_min - 0.1, y_max + 0.1)

            ax[a_0][a_1].set_xlabel("LENS D0")
            ax[a_0][a_1].set_ylabel("LENS D1")

        ax[1][1].axis("off")
        fig.savefig(title, dpi=600)
        plt.close(fig)

    elif m_clean.shape[2] == 2:
        fig, ax = plt.subplots(figsize=(6, 6))

        id_max, id_min = 0, 0
        for idx, mol in enumerate(m_clean):
            if np.max(mol) == np.max(m_clean):
                id_max = idx
            if np.min(mol) == np.min(m_clean):
                id_min = idx

        line_w = 0.05
        max_t = all_the_labels.shape[1]
        m_resized = m_clean[:, :max_t, :]
        step = 5 if m_resized.size > 1000000 else 1

        for i, mol in enumerate(m_resized[::step]):
            ax.plot(
                mol.T[0], mol.T[1],
                c="black", lw=line_w, rasterized=True, zorder=0,
            )
            color_list = all_the_labels[i * step] + 1
            ax.scatter(
                mol.T[0], mol.T[1],
                c=color_list, cmap=tmp,
                vmin=0, vmax=n_states - 1,
                s=0.5, rasterized=True,
            )

        color_list = all_the_labels[id_min] + 1
        ax.plot(
            m_resized[id_min].T[0],
            m_resized[id_min].T[1],
            c="black", lw=line_w, rasterized=True, zorder=0,
        )
        ax.scatter(
            m_resized[id_min].T[0],
            m_resized[id_min].T[1],
            c=color_list, cmap=tmp,
            vmin=0, vmax=n_states - 1,
            s=0.5, rasterized=True,
        )
        color_list = all_the_labels[id_max] + 1
        ax.plot(
            m_resized[id_max].T[0],
            m_resized[id_max].T[1],
            c="black", lw=line_w, rasterized=True, zorder=0,
        )
        ax.scatter(
            m_resized[id_max].T[0],
            m_resized[id_max].T[1],
            c=color_list, cmap=tmp,
            vmin=0, vmax=n_states - 1,
            s=0.5, rasterized=True,
        )

        # Plot ellipses for states
        #for state in state_list:
        #    att = state.get_attributes()
        #    ellipse = Ellipse(
        #        tuple(att["mean"]),
        #        att["axis"][0],
        #        att["axis"][1],
        #        color="black",
        #        fill=False,
        #    )
        #    ax.add_patch(ellipse)

        if auto_lims == 1:
            xvals = m_resized[..., 0]
            yvals = m_resized[..., 1]
            x_min, x_max = xvals.min(), xvals.max()
            y_min, y_max = yvals.min(), yvals.max()
            margin_x = 0.01 * (x_max - x_min)
            margin_y = 0.01 * (y_max - y_min)

            ax.set_xlim(x_min - margin_x, x_max + margin_x)
            ax.set_ylim(y_min - margin_y, y_max + margin_y)

            ax.set_xlabel(r"$\Phi$", fontsize=19)
            ax.set_ylabel(r"$\dot{\Phi}$", fontsize=19)
            ax.tick_params(axis='both', labelsize=16)


            formatter_x = ScalarFormatter(useMathText=True)
            formatter_x.set_powerlimits((-1, 1))
            ax.xaxis.set_major_formatter(formatter_x)

            formatter_y = ScalarFormatter(useMathText=True)
            formatter_y.set_powerlimits((-1, 1))
            ax.yaxis.set_major_formatter(formatter_y)

            ax.xaxis.get_offset_text().set_fontsize(12)
            ax.yaxis.get_offset_text().set_fontsize(12)

        else:
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")

        fig.tight_layout()
        fig.savefig(title, dpi=600)
        plt.close(fig)

##############################################################################
# These two functions handle the entropy-based computation
##############################################################################
import numpy.typing as npt

def apply_threshold_merge(state_list, labels, threshold):
    """
    Merge states with population below `threshold` into Env0 (label = -1).

    Prints diagnostic info, then returns the updated state_list and labels.
    """
    print(f"[INFO] Applying threshold = {threshold:.3f}")
    
    # 1) Mark small-pop states as -1
    for idx, st in enumerate(state_list):
        print(f"  State {idx} => population={st.perc:.4f}")
        if st.perc < threshold:
            print(f"    -> Marking state {idx} as Env0 (label = -1).")
            labels[labels == idx] = -1

    # 2) Re-map remaining states to consecutive labels
    valid_map = {}
    new_state_list = []
    next_label = 0
    for idx, st in enumerate(state_list):
        if st.perc >= threshold:
            valid_map[idx] = next_label
            new_state_list.append(st)
            next_label += 1

    # 3) Update labels to their new values
    for old_idx, new_idx in valid_map.items():
        print(f"    Re-mapping old label {old_idx} -> new label {new_idx}")
        labels[labels == old_idx] = new_idx

    print(f"[INFO] After threshold merge: #states = {len(new_state_list)}, plus Env0 in -1.")
    return new_state_list, labels

def compute_multivariate_entropy(
    data: npt.NDArray[np.float64],
    data_ranges: list[tuple[float, float]],
    n_bins: list[int],
) -> float:
    """Compute the entropy of a multivariate data distribution.

    It is normalized so that a uniform distribution has unitary entropy.

    Parameters
    ----------
    data : ndarray
        shape (n_points, window_size, n_dims).
        The dataset for which the entropy is to be computed.
    data_ranges : list of (min, max) tuples
        A list of ranges for each dimension.
    n_bins : list of int
        The number of bins for each dimension.

    Returns
    -------
    float
        The normalized Shannon entropy (0 <= entropy <= 1).
    """
    n_points, window_size, n_dims = data.shape
    if data.size == 0:
        raise ValueError("data is empty")
    if n_dims != len(data_ranges) or n_dims != len(n_bins):
        raise ValueError("Mismatch between data dimensions, data_ranges, and n_bins")

    # Flatten time+particles axis: shape -> (n_points * window_size, n_dims)
    data_reshaped = data.reshape((n_points * window_size, n_dims))

    # Build the histogram in n_dims
    counts, _ = np.histogramdd(data_reshaped, bins=n_bins, range=data_ranges)
    probs = counts / np.sum(counts)  # Probability distribution
    # Shannon entropy (base 2). We only sum over nonzero probabilities:
    entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
    # Normalization by max possible entropy log2(prod(n_bins))
    entropy /= np.log2(np.prod(n_bins))

    return entropy


def compute_multivariate_gain(
    data: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    n_bins: list[int],
) -> tuple[float, float, float, float]:
    """
    Compute the information gain of a multi-dimensional dataset
    once it has been clustered by onion_multi.

    Parameters
    ----------
    data : ndarray, shape (n_points, window_size, n_dims)
        The data used in onion_multi() for clustering.
    labels : ndarray, shape (n_points,)
        The clustering labels returned by onion_multi().
    n_bins : list of int
        Number of bins for each dimension (e.g. [25, 25]).

    Returns
    -------
    (info_gain, relative_info_gain, total_entropy, clustered_entropy)
        info_gain : float
            H0 - Hcluster
        relative_info_gain : float
            (H0 - Hcluster) / H0
        total_entropy : float
            Shannon entropy H0 of the data
        clustered_entropy : float
            Weighted Shannon entropy of each cluster
    """
    if data.shape[0] != labels.shape[0]:
        msg = f"data ({data.shape}) and labels ({labels.shape}) must match on data.shape[0]"
        raise RuntimeError(msg)

    # Build the dimension-specific ranges for each dimension
    # data has shape (n_points, window_size, n_dims) -> we want min/max across the entire set
    _, _, n_dims = data.shape
    data_ranges = []
    for dim_i in range(n_dims):
        # Extract all values along dimension dim_i, flatten them
        dim_vals = data[:, :, dim_i].ravel()
        data_ranges.append((float(dim_vals.min()), float(dim_vals.max())))

    # 1) Compute total entropy (H0) of the entire dataset
    total_entropy = compute_multivariate_entropy(data, data_ranges, n_bins)

    # 2) Compute fraction & entropy of each cluster
    unique_labels = np.unique(labels)
    n_clusters = unique_labels.size
    frac = np.zeros(n_clusters)
    entr = np.zeros(n_clusters)

    for i, lab in enumerate(unique_labels):
        mask = (labels == lab)
        frac[i] = np.sum(mask) / labels.size
        # data[mask] has shape (number_of_points_in_that_cluster, window_size, n_dims)
        entr[i] = compute_multivariate_entropy(data[mask], data_ranges, n_bins)

    # Weighted sum to get the clustered entropy
    clustered_entropy = np.dot(frac, entr)
    info_gain = total_entropy - clustered_entropy

    return (
        info_gain,
        info_gain / total_entropy,
        total_entropy,
        clustered_entropy,
    )

def plot_env0_and_nstates_vs_gain(
    title: str,
    env0_values,
    gains,
    nstates
):
    """
    Creates two separate plots:
    1) Env0 fraction vs. Gain
    2) #States vs. Gain

    Each plot is saved to a separate PNG file, using the provided title prefix.

    Parameters
    ----------
    title : str
        The base name for the output PNG files.
    env0_values : array-like
        Env0 fractions (x-axis in the first plot).
    gains : array-like
        Gain values (y-axis in both plots).
    nstates : array-like
        Number of states (x-axis in the second plot).
    """

    # 1) Env0 vs Gain
    plt.figure()
    plt.plot(env0_values, gains, marker="o", label="Gain vs Env0")
    plt.xlabel("Env0 fraction")
    plt.ylabel("Gain")
    plt.legend()
    plt.savefig(f"{title}_env0_vs_gain.png", dpi=600)
    plt.close()

    # 2) #States vs Gain
    plt.figure()
    plt.plot(nstates, gains, marker="o", label="Gain vs #States")
    plt.xlabel("#States")
    plt.ylabel("Gain")
    plt.legend()
    plt.savefig(f"{title}_nstates_vs_gain.png", dpi=600)
    plt.close()

def plot_time_res_and_entropy_metrics(
    title: str,
    all_delta_t: np.ndarray,
    gains: np.ndarray,
    gains_on_tot_entropy: np.ndarray,
    tots_entropy: np.ndarray,
    clustered_entropies: np.ndarray,
):
    """
    Plots the four entropy-based metrics as a function of time resolution (log-scale).

    Parameters
    ----------
    title : str
        Output .png filename.
    all_delta_t : ndarray
        The set of delta_t values (x-axis).
    gains : ndarray
        The Gain metric for each delta_t.
    gains_on_tot_entropy : ndarray
        The Gain_on_tot_entropy metric for each delta_t.
    tots_entropy : ndarray
        The total entropy for each delta_t.
    clustered_entropies : ndarray
        The clustered entropy for each delta_t.
    """
    plt.figure()
    plt.xscale("log")  # Log scale on X
    plt.plot(all_delta_t, gains, marker="o", label="Gain")
    #plt.plot(all_delta_t, gains_on_tot_entropy, marker="o", label="Gain_on_tot_entropy")
    #plt.plot(all_delta_t, tots_entropy, marker="o", label="Tot_entropy")
    #plt.plot(all_delta_t, clustered_entropies, marker="o", label="Clustered_entropy")
    plt.xlabel(r"Time resolution $\Delta t$ [frame] (log scale)")
    plt.ylabel(r"Entropy metrics")
    plt.legend()
    plt.savefig(title, dpi=600)
    plt.close()

def plot_gain_vs_env0(
    title: str,
    all_delta_t: np.ndarray,
    gain_values: np.ndarray,
    env0_values: np.ndarray,
):
    """
    Plots Gain (left axis, red) vs. Env0 fraction (right axis, green) 
    as a function of time resolution (log-scale).

    Parameters
    ----------
    title : str
        Output .png filename.
    all_delta_t : ndarray
        The set of delta_t values (x-axis).
    gain_values : ndarray
        Gain metric for each delta_t.
    env0_values : ndarray
        Env0 fraction for each delta_t.
    """
    fig, ax = plt.subplots()

    # Left axis: Gain (red)
    ax.plot(all_delta_t, gain_values, marker='o', color='red', label='Gain')
    ax.set_xscale("log")
    ax.set_xlabel(r"Time resolution $\Delta t$ [frame] (log scale)")
    ax.set_ylabel("Gain", weight="bold", color="red")
    ax.tick_params(axis='y', labelcolor='red')
    ax.yaxis.set_major_locator(MaxNLocator(integer=False))

    # Right axis: Env0 fraction (green)
    ax_r = ax.twinx()
    ax_r.plot(all_delta_t, env0_values, marker='o', color='green', label='Env0 fraction')
    ax_r.set_ylabel("Env0 fraction", weight="bold", color="green")
    ax_r.tick_params(axis='y', labelcolor='green')
    ax_r.set_ylim(-0.02, 1.02)

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax_r.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.savefig(title, dpi=600)
    plt.close(fig)

def plot_gain_vs_nstates(
    title: str,
    all_delta_t: np.ndarray,
    gains: np.ndarray,
    nstates: np.ndarray,
):
    """
    Plots Gain (left axis) vs. the number of states (right axis) 
    as a function of time resolution (log-scale).

    Parameters
    ----------
    title : str
        Name of the output PNG file.
    all_delta_t : ndarray
        Array of time-resolution values (x-axis).
    gains : ndarray
        Gain values corresponding to each delta_t.
    nstates : ndarray
        Number of states corresponding to each delta_t.
    """
    fig, ax = plt.subplots()

    # Left axis for Gain (in red)
    ax.plot(all_delta_t, gains, marker="o", color="red", label="Gain")
    ax.set_xscale("log")
    ax.set_xlabel(r"Time resolution $\Delta t$ [frame] (log scale)")
    ax.set_ylabel("Gain", weight="bold", color="red")
    ax.tick_params(axis="y", labelcolor="red")
    ax.yaxis.set_major_locator(MaxNLocator(integer=False))

    # Right axis for number of states (in blue)
    ax2 = ax.twinx()
    ax2.plot(all_delta_t, nstates, marker="o", color="blue", label="#States")
    ax2.set_ylabel("#States", weight="bold", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.savefig(title, dpi=600)
    plt.close(fig)

##############################################################################
# This is the integrated main script
##############################################################################
def main() -> None:
    """
    Example main function for clustering 2D (or multi-dim) data, similar to the 1D version.
    """
    # 1) Load your multi-dimensional data
    path_to_input_data = "../merged_component_0_component_1.npy"
    input_data = np.load(path_to_input_data)  # shape: (n_dims, n_particles, n_frames)
    n_particles, n_frames, n_dims = input_data.shape
    print(f"Loaded input data with shape (n_particles, n_frames, n_dims) = {input_data.shape}")
    input_data = input_data.transpose(2, 0, 1)
    print(f"Loaded input data adjusted with shape (n_dims, n_partciles, n_frames) = {input_data.shape}")

    # 2) Choose a range of time resolutions (delta_t / tau_window).
    all_tau_windows = np.unique(np.geomspace(2, n_frames, num=20, dtype=int))
    print("All tau windows:", all_tau_windows)

    # 3) Prepare arrays/lists to store the results
    tra = np.zeros((len(all_tau_windows), 3))  # [tau_window, #states, Env0 fraction]
    list_of_pop = []
    gains = []
    gains_on_tot_entropy = []
    tots_entropy = []
    clustered_entropies = []

    original_dir = os.getcwd()

    # Create or overwrite the file to store state info
    with open("all_states_info.txt", "w") as f:
        f.write("tau_window\tUnique_states\tEnv0\tpop_list_other_states\n")
    
    # Determine base name for the output file
    base_name = os.path.splitext(os.path.basename(path_to_input_data))[0]
    gains_file = f"{base_name}_gains.txt"
    
    # Write the header before the cycle
    with open(gains_file, "w") as g:
        g.write("delta_t\tgain\n")
    
    # 4) Loop over each tau_window
    for i, tau_window in enumerate(all_tau_windows):
        print(f"\n[INFO] Processing tau_window={tau_window} ...")
        # reshape_from_dnt produces shape = (n_points, tau_window*n_dims)
        # where n_points = n_particles*(n_frames//tau_window)
        reshaped_data_2d = helpers.reshape_from_dnt(input_data, tau_window)
        print("  reshaped_data_2d shape:", reshaped_data_2d.shape)

        input_data_2 = reshaped_data_2d.reshape((n_dims, n_particles, -1))
        print(f"Reshaped for input_data_2 = {input_data_2.shape}")

        # Now cluster with onion_multi
        state_list, labels = onion_multi(reshaped_data_2d, bins=25)  # Example: 25x25 bins
        #state_list, labels= apply_threshold_merge(state_list,labels,0.005)
        #print(f"  Found {len(state_list)} states.")

        # Re-reshape data for computing gain:
        #  onion_multi => (n_points, tau_window*n_dims)
        #  compute_multivariate_gain => (n_points, tau_window, n_dims)
        n_points = reshaped_data_2d.shape[0]

        ######################################################################
        #### This horrible reshaping is the reason this code is temporary ####
        ds_reshaped = np.empty((
            n_dims,
            n_particles * int(n_frames / tau_window),
            tau_window
        ))
        ds_reshaped[0, :, :] = helpers.reshape_from_nt(input_data_2[0, :, :], tau_window)
        ds_reshaped[1, :, :] = helpers.reshape_from_nt(input_data_2[1, :, :], tau_window)
        ds_reshaped = np.transpose(ds_reshaped, (1, 2, 0))
        ######################################################################

        reshaped_for_gain = reshaped_data_2d.reshape((n_points, tau_window, n_dims))
        #print(f"Reshaped for gain = {reshaped_for_gain.shape}")

        # Compute the info gain
        # Provide the same [25,25] bins (or however many dims) to match onion_multi's binning approach
        info_gain, rel_info_gain, total_entropy, clust_entropy = compute_multivariate_gain(
            ds_reshaped,
            labels,
            n_bins=[25,25]
        )
        print(f"  gain={info_gain:.4f}, relative_gain={rel_info_gain:.3%}")
        
        with open(gains_file, "a") as g:
            g.write(f"{tau_window}\t{info_gain}\n")
        
        # Store gain results
        gains.append(info_gain)
        gains_on_tot_entropy.append(rel_info_gain)
        tots_entropy.append(total_entropy)
        clustered_entropies.append(clust_entropy)

        # 5) Evaluate the population of each state
        #    Env0 fraction is unclassified portion
        pop_list = [st.perc for st in state_list]
        env0 = 1 - np.sum(pop_list)
        pop_list.insert(0, env0)  # Insert env0 as first item
        list_of_pop.append(pop_list)

        # Fill 'tra' row
        tra[i][0] = tau_window
        tra[i][1] = len(state_list)
        tra[i][2] = env0

        # Also append info to the .txt file
        other_states_str = " ".join(str(p) for p in pop_list[1:])
        with open("all_states_info.txt", "a") as f:
            f.write(f"{tau_window}\t{len(state_list)}\t{env0}\t{other_states_str}\n")

        # 6) Optionally generate onion plots at certain tau_windows (like i in [2, 6, 12])
        if i in [0,1,2, 6, 12]:
            folder_name = f"tau_{tau_window}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            os.chdir(folder_name)

            # Some tropea_clustering multi-dim plots:
            #plot_output_multi("Fig1.png", input_data_2, state_list, labels, tau_window)
            plot_output_multi_v2("Fig1.png", input_data_2, state_list, labels, tau_window,1)
            plot_one_trj_multi("Fig2.png", 0, tau_window, input_data_2, labels)
            plot_medoids_multi("Fig3.png", tau_window, input_data_2, labels)
            #plot_state_populations("Fig4.png", n_frames//tau_window, tau_window, labels)
            color_trj_from_xyz("/home/dom/Scrivania/PhD/Python_codes/becchi_clustering/wave_deriv/trajectory_281_frames.xyz",labels,n_particles,tau_window)
            #plot_sankey("Fig5.png", labels, n_particles, [50, 100, 150])

            os.chdir(original_dir)

    # 7) After the loop, produce summary plots
    #    e.g. the time_res_analysis that shows #states (left axis) and Env0 fraction (right axis)
    plot_time_res_analysis("Fig6.png", tra)
    #plot_pop_fractions("Fig7.png", list_of_pop, tra)

    # 8) Plot your gain metrics across all tau_windows
    #    a) single line with Gains (like "combined_plot.png")
    plot_time_res_and_entropy_metrics(
        "combined_plot.png",
        all_tau_windows,
        np.array(gains),
        np.array(gains_on_tot_entropy),
        np.array(tots_entropy),
        np.array(clustered_entropies),
    )

    #    b) Gains vs Env0 fraction
    plot_gain_vs_env0(
        "gains_vs_env0.png",
        all_tau_windows,
        np.array(gains),
        tra[:, 2]  # env0 fraction
    )

    #    c) Gains vs #States
    plot_gain_vs_nstates(
        "gain_vs_nstates.png",
        all_tau_windows,
        np.array(gains),
        tra[:, 1]  # number of states
    )

    #    d) Two separate plots: Gains vs Env0, Gains vs #States
    plot_env0_and_nstates_vs_gain(
        title="analysis_results",
        env0_values=tra[:, 2],
        gains=gains,
        nstates=tra[:, 1]
    )

    print("[INFO] Finished all computations and plots.")

if __name__ == "__main__":
    main()
