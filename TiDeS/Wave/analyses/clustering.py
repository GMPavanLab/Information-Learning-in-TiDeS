"""Example script for running dynsight.onion.onion_uni."""
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist
from dynsight import onion
from matplotlib.ticker import MaxNLocator

def compute_data_entropy(
    data: npt.NDArray[np.float64],
    data_range: tuple[float, float],
    n_bins: int,
) -> float:
    """Compute the entropy of a data distribution.

    It is normalized so that a uniform distribution has unitary entropy.

    Parameters:
        data:
            The dataset for which the entropy is to be computed.

        data_range:
            A tuple (min, max) specifying the range over which the data
            histogram must be computed.

        n_bins:
            The number of bins with which the data histogram must be computed.

    Returns:
        float:
            entropy:
                The value of the normalized Shannon entropy of the dataset.

    Example:

        .. testcode:: shannon1-test

            import numpy as np
            from dynsight.analysis import compute_data_entropy

            np.random.seed(1234)
            data = np.random.rand(100, 100)
            data_range = (float(np.min(data)), float(np.max(data)))

            data_entropy = compute_data_entropy(
                data,
                data_range,
                n_bins=40,
            )

        .. testcode:: shannon1-test
            :hide:

            assert np.isclose(data_entropy, 0.9993954419344714)
    """
    if data.size == 0:
        msg = "data is empty"
        raise ValueError(msg)
    counts, _ = np.histogram(
        data,
        bins=n_bins,
        range=data_range,
    )
    probs = counts / np.sum(counts)  # Data probabilities are needed
    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0.0])
    entropy /= np.log2(n_bins)
    return entropy

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


def compute_entropy_gain(
    data: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    n_bins: int = 20,
) -> tuple[float, float, float, float]:
    """Compute the relative information gained by the clustering.

    Parameters:
        data:
            The dataset over which the clustering is performed.

        labels:
            The clustering labels. Has the same shape as "data".

        n_bins:
            The number of bins with which the data histogram must be computed.
            Default is 20.

    Returns:
        tuple[float, float, float, float]
            * The absolute information gain :math:`H_0 - H_{clust}`
            * The relative information gain :math:`(H_0 - H_{clust}) / H_0`
            * The Shannon entropy of the initial data :math:`H_0`
            * The shannon entropy of the clustered data :math:`H_{clust}`

    Example:

        .. testcode:: shannon2-test

            import numpy as np
            from dynsight.analysis import compute_entropy_gain

            np.random.seed(1234)
            data = np.random.rand(100, 100)
            labels = np.random.randint(-1, 2, size=(100, 100))

            _, entropy_gain, *_ = compute_entropy_gain(
                data,
                labels,
                n_bins=40,
            )

        .. testcode:: shannon2-test
            :hide:

            assert np.isclose(entropy_gain, 0.0010065005804883983)
    """
    if data.shape[0] != labels.shape[0]:
        msg = (
            f"data ({data.shape}) and labels ({labels.shape}) "
            "must have same shape[0]"
        )
        raise RuntimeError(msg)

    data_range = (float(np.min(data)), float(np.max(data)))

    # Compute the entropy of the raw data
    total_entropy = compute_data_entropy(
        data,
        data_range,
        n_bins,
    )

    # Compute the fraction and the entropy of the single clusters
    n_clusters = np.unique(labels).size
    frac, entr = np.zeros(n_clusters), np.zeros(n_clusters)
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        frac[i] = np.sum(mask) / labels.size
        entr[i] = compute_data_entropy(
            data[mask],
            data_range,
            n_bins,
        )

    # Compute the entropy of the clustered data
    clustered_entropy = np.dot(frac, entr)
    info_gain = total_entropy - clustered_entropy

    return (
        info_gain,
        info_gain / total_entropy,
        total_entropy,
        clustered_entropy,
    )

import matplotlib.pyplot as plt

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

def main() -> None:
    # Set the path to where the example files are located
    path_to_input_data = "../component_0.npy"

    # Load the input data - it's an array of shape (n_particles, n_frames)
    input_data = np.load(path_to_input_data)[:,:]
    n_particles, n_frames = input_data.shape
    
    # Choose time resolutions
    all_delta_t = np.unique(np.geomspace(2, n_frames, num=20, dtype=int))

    tra = np.zeros((len(all_delta_t), 3))
    list_of_pop = []
    gains = []
    gains_on_tot_entropy = []
    tots_entropy = []
    clustered_entropies = []
    original_dir = os.getcwd()

    with open("all_states_info.txt", "w") as f:
        f.write("delta_t\tUnique_states\tEnv0\tpop_list_other_states\n")
    
    # Determine base name for the output file
    base_name = os.path.splitext(os.path.basename(path_to_input_data))[0]
    gains_file = f"{base_name}_gains.txt"
    
    # Write the header before the cycle
    with open(gains_file, "w") as g:
        g.write("delta_t\tgain\n")
    
    	

    for i, delta_t in enumerate(all_delta_t):
        print(delta_t)
        reshaped_data = onion.helpers.reshape_from_nt(input_data, delta_t)
        #print(np.shape(reshaped_data))
        
        state_list, labels = onion.onion_uni(reshaped_data)

        state_list, labels= apply_threshold_merge(state_list,labels,0.01)
        #state_list, labels= apply_threshold_merge(state_list,labels,0.004)

        gain, gain_on_tot_entropy, tot_entropy, clustered_entropy = compute_entropy_gain(
            reshaped_data, labels, n_bins=20
        )

        #print("state list:", (np.shape(state_list)))
        #print("labels:", (np.shape(labels)))

        print("gain:", gain)
        
        with open(gains_file, "a") as g:
            g.write(f"{delta_t}\t{gain}\n")
        
        gains.append(gain)
        gains_on_tot_entropy.append(gain_on_tot_entropy)
        tots_entropy.append(tot_entropy)
        clustered_entropies.append(clustered_entropy)

        pop_list = [state.perc for state in state_list]
        pop_list.insert(0, 1 - np.sum(np.array(pop_list)))  # Add ENV0 fraction
        list_of_pop.append(pop_list)

        tra[i][0] = delta_t
        tra[i][1] = len(state_list)
        tra[i][2] = pop_list[0]

        env0 = pop_list[0]
        other_states = pop_list[1:]
        other_states_str = " ".join(str(p) for p in other_states)

        # Append the new line to the .txt file
        with open("all_states_info.txt", "a") as f:
            f.write(f"{delta_t}\t{len(state_list)}\t{env0}\t{other_states_str}\n")


        # Example of creating plots at specific iterations
        if i in [0,2, 6, 12]:
            folder_name = f"delta_t_{delta_t}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            
            # Change to the new folder
            os.chdir(folder_name)
            
            # Generate all onion plots inside this folder
            onion.plot.plot_output_uni(
                "Fig1.png", reshaped_data, n_particles, state_list
            )
            onion.plot.plot_one_trj_uni(
                "Fig2.png", 1234, reshaped_data, n_particles, labels
            )
            onion.plot.plot_medoids_uni(
                "Fig3.png", reshaped_data, labels
            )
            onion.plot.plot_state_populations(
                "Fig4.png", n_particles, delta_t, labels
            )
            
            #onion.plot.plot_sankey("Fig5.png", labels, n_particles, [10, 20, 30, 40])

            onion.plot.color_trj_from_xyz(
                "/home/dom/Scrivania/PhD/Python_codes/becchi_clustering/wave_deriv/trajectory_281_frames.xyz", labels, n_particles, delta_t
            )

            # Go back to the original directory before continuing
            os.chdir(original_dir)

    # Plot the four entropy metrics in one figure
    plot_time_res_and_entropy_metrics(
        "combined_plot.png",
        all_delta_t,
        np.array(gains),
        np.array(gains_on_tot_entropy),
        np.array(tots_entropy),
        np.array(clustered_entropies),
    )
    
    plot_gain_vs_env0(
    "gains_vs_env0.png",
    all_delta_t,
    np.array(gains),
    tra[:, 2]  # all Env0 fractions
    )

    plot_gain_vs_nstates(
    "gain_vs_nstates.png",
    all_delta_t,
    gains,
    tra[:, 1]  # number of states
    )
     
    plot_env0_and_nstates_vs_gain(
    title="analysis_results",
    env0_values=tra[:, 2] ,
    gains=gains,
    nstates= tra[:, 1]
    )

    # Original onion plotting functions
    onion.plot.plot_time_res_analysis("Fig6.png", tra)
    onion.plot.plot_pop_fractions("Fig7.png", list_of_pop, tra)


if __name__ == "__main__":
    main()
