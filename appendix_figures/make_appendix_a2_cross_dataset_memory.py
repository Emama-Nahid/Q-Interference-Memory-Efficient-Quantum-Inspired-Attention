from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Save outputs in the same folder as this script
OUT_DIR = Path(__file__).resolve().parent

# Data from the cross-dataset comparison table
datasets = ["TinyStories", "pile-10k", "small-C4"]
models = [
    "Baseline GPT",
    "Naive interference",
    "Quantum-Paper baseline",
    "Q-Interference (ours)",
]

# Peak GPU memory in MB
memory = np.array([
    [8055.76, 12138.34, 7199.26, 4227.14],
    [8055.76, 12138.34, 7199.26, 4227.14],
    [8055.76, 12138.34, 7199.26, 4227.14],
])

# Professional muted palette, different from the first figure
colors = ["#4C566A", "#BF616A", "#B48EAD", "#5E8C61"]
hatches = ["", "//", "\\\\", ".."]

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

x = np.arange(len(datasets))
bar_width = 0.19

fig, ax = plt.subplots(figsize=(7.0, 3.0), dpi=300)

for i, model in enumerate(models):
    offset = (i - 1.5) * bar_width
    bars = ax.bar(
        x + offset,
        memory[:, i],
        width=bar_width,
        label=model,
        color=colors[i],
        edgecolor="black",
        linewidth=0.4,
        hatch=hatches[i],
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 260,
            f"{height / 1000:.1f}k",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=0,
        )

ax.set_ylabel("Peak GPU memory (MB)")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 13500)
ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
ax.set_axisbelow(True)

# Clean NeurIPS-style appearance
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend above the plot to avoid covering bars
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.22),
    ncol=2,
    frameon=False,
)

fig.tight_layout()

pdf_path = OUT_DIR / "cross_dataset_peak_memory.pdf"
png_path = OUT_DIR / "cross_dataset_peak_memory.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")