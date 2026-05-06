from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Save outputs in the same folder as this script
OUT_DIR = Path(__file__).resolve().parent

# Data from Table 2 and Table 3
# Each row is one model on one dataset
rows = [
    # WikiText-103 controlled result
    ("WikiText-103", "Baseline GPT", 24.6534, 8055.76),
    ("WikiText-103", "Quantum-Paper baseline", 60.5367, 7199.26),
    ("WikiText-103", "Q-Interference (ours)", 24.1718, 4227.14),

    # TinyStories
    ("TinyStories", "Baseline GPT", 5.6196, 8055.76),
    ("TinyStories", "Naive interference", 10.4037, 12138.34),
    ("TinyStories", "Quantum-Paper baseline", 15.0016, 7199.26),
    ("TinyStories", "Q-Interference (ours)", 5.6941, 4227.14),

    # pile-10k
    ("pile-10k", "Baseline GPT", 75.9008, 8055.76),
    ("pile-10k", "Naive interference", 165.4355, 12138.34),
    ("pile-10k", "Quantum-Paper baseline", 168.0210, 7199.26),
    ("pile-10k", "Q-Interference (ours)", 116.9275, 4227.14),

    # small-C4
    ("small-C4", "Baseline GPT", 197.0284, 8055.76),
    ("small-C4", "Naive interference", 268.3050, 12138.34),
    ("small-C4", "Quantum-Paper baseline", 356.7006, 7199.26),
    ("small-C4", "Q-Interference (ours)", 246.9012, 4227.14),
]

# Different palette from the previous memory figure
model_colors = {
    "Baseline GPT": "#2F4B7C",
    "Naive interference": "#D95F02",
    "Quantum-Paper baseline": "#7570B3",
    "Q-Interference (ours)": "#1B9E77",
}

dataset_markers = {
    "WikiText-103": "o",
    "TinyStories": "s",
    "pile-10k": "^",
    "small-C4": "D",
}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.0, 3.7), dpi=300)

for dataset, model, ppl, mem in rows:
    ax.scatter(
        mem,
        ppl,
        s=70,
        color=model_colors[model],
        marker=dataset_markers[dataset],
        edgecolor="black",
        linewidth=0.45,
        alpha=0.92,
        zorder=3,
    )

# Use log scale because datasets have very different perplexity ranges
ax.set_yscale("log")

ax.set_xlabel("Peak GPU memory (MB)")
ax.set_ylabel("Test perplexity, log scale")
ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.65)
ax.set_axisbelow(True)

# Add light guidance text
ax.annotate(
    "Lower memory\nand lower perplexity",
    xy=(4300, 5.8),
    xytext=(6100, 8.5),
    arrowprops=dict(arrowstyle="->", linewidth=0.8),
    fontsize=8,
    ha="left",
    va="center",
)

# Clean NeurIPS-style appearance
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend for models
model_handles = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor=color,
        markeredgecolor="black",
        markersize=7,
        label=model,
    )
    for model, color in model_colors.items()
]

# Legend for datasets
dataset_handles = [
    Line2D(
        [0], [0],
        marker=marker,
        color="black",
        markerfacecolor="white",
        markeredgecolor="black",
        linestyle="None",
        markersize=7,
        label=dataset,
    )
    for dataset, marker in dataset_markers.items()
]

legend1 = ax.legend(
    handles=model_handles,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.00),
    frameon=False,
    title="Model",
    title_fontsize=8,
)

legend2 = ax.legend(
    handles=dataset_handles,
    loc="lower left",
    bbox_to_anchor=(1.02, 0.02),
    frameon=False,
    title="Dataset",
    title_fontsize=8,
)

ax.add_artist(legend1)

fig.tight_layout()

pdf_path = OUT_DIR / "quality_memory_tradeoff.pdf"
png_path = OUT_DIR / "quality_memory_tradeoff.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")