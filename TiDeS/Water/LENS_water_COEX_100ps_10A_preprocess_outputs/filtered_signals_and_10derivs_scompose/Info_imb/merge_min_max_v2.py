import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# Load data
min_data = np.load("imbalance_greedy_select_data.npz", allow_pickle=True)
max_data = np.load("imbalance_greedy_select_max_data.npz", allow_pickle=True)

min_imbalances = min_data["min_imbalances"]
max_imbalances = max_data["max_imbalances"]
steps = np.arange(len(min_imbalances))  # Start from 0

selected = min_data["selected"]
selected_max = max_data["selected_max"]

print("Component stacked at each step for MIN imbalance:")
for i, idx in enumerate(selected):
    print(f"  Step {i+1}: component {idx}")

print("Component stacked at each step for MAX imbalance:")
for i, idx in enumerate(selected_max):
    print(f"  Step {i+1}: component {idx}")

# Plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(steps, min_imbalances, '-', color='royalblue')
ax.plot(steps, max_imbalances, '-', color='crimson')
ax.scatter(steps, min_imbalances, color='royalblue', label="MIN I IMB")
ax.scatter(steps, max_imbalances, color='crimson', label="MAX I IMB")
ax.fill_between(steps, min_imbalances, max_imbalances, color='gray', alpha=0.22)

# Set ticks
ax.set_xticks(steps)


# Axis labels and limits
ax.set_xlabel("K", fontsize=16)
ax.set_ylabel(r"$I_{\mathrm{IMB}}(D^{10} \rightarrow D_K)$", fontsize=16)
ax.set_xlim(-1, 12)
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.set_yticks([y for y in ax.get_yticks() if y >= 0])  # remove ticks below 0
ax.set_ylim(-0.3, max(np.max(min_imbalances), np.max(max_imbalances)) * 1.05)


# Style
ax.tick_params(axis='both', labelsize=16)
#ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(fontsize=14)

plt.tight_layout()
plt.savefig("greedy_min_max_imbalance_progression_v2.png", dpi=300)
plt.show()

