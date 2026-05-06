from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Save outputs in the same folder as this script
OUT_DIR = Path(__file__).resolve().parent

# Dataset names
datasets = ["WikiText-103", "TinyStories", "pile-10k", "small-C4"]

# Baseline GPT values from Table 2 and Table 3
baseline_ppl = np.array([24.6534, 5.6196, 75.9008, 197.0284])
baseline_mem = np.array([8055.76, 8055.76, 8055.76, 8055.76])

# Q-Interference values from Table 2 and Table 3
q_ppl = np.array([24.1718, 5.6941, 116.9275, 246.9012])
q_mem = np.array([4227.14, 4227.14, 4227.14, 4227.14])

# Relative change in percent
ppl_change = 100.0 * (q_ppl - baseline_ppl) / baseline_ppl
mem_change = 100.0 * (q_mem - baseline_mem) / baseline_mem

# Matrix for the heatmap
data = np.vstack([ppl_change, mem_change])
row_labels = ["Test perplexity", "Peak memory"]

# Custom diverging palette
# Negative values are better because lower PPL and lower memory are both better.
cmap = LinearSegmentedColormap.from_list(
    "relative_change_cmap",
    ["#2A9D8F", "#F7F7F7", "#E76F51"]
)
norm = TwoSlopeNorm(vmin=-55, vcenter=0, vmax=55)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(6.7, 2.25), dpi=300)

im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

# Axis labels
ax.set_xticks(np.arange(len(datasets)))
ax.set_xticklabels(datasets)
ax.set_yticks(np.arange(len(row_labels)))
ax.set_yticklabels(row_labels)

# Move x labels to top for compact appendix style
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

# Cell borders
ax.set_xticks(np.arange(-0.5, len(datasets), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
ax.tick_params(which="minor", bottom=False, left=False)

# Annotate each cell
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        value = data[i, j]
        label = f"{value:+.1f}%"
        text_color = "white" if abs(value) > 28 else "black"
        ax.text(j, i, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=text_color)

# Clean border
for spine in ax.spines.values():
    spine.set_visible(False)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
cbar.set_label("Relative change vs. Baseline GPT (%)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Small note below the matrix
fig.text(
    0.5,
    -0.02,
    "Negative values indicate improvement because lower perplexity and lower memory are better.",
    ha="center",
    va="top",
    fontsize=8,
)

fig.tight_layout()

pdf_path = OUT_DIR / "relative_change_matrix.pdf"
png_path = OUT_DIR / "relative_change_matrix.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

print("\nRelative changes:")
for d, p, m in zip(datasets, ppl_change, mem_change):
    print(f"{d}: PPL {p:+.2f}%, Memory {m:+.2f}%")