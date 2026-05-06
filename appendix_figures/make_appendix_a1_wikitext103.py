from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Save outputs in the same folder as this script
OUT_DIR = Path(__file__).resolve().parent

# Data from the main WikiText-103 controlled comparison table
models = ["Baseline GPT", "Quantum-Paper\nbaseline", "Q-Interference\n(ours)"]
test_ppl = np.array([24.6534, 60.5367, 24.1718])
peak_mem = np.array([8055.76, 7199.26, 4227.14])

x = np.arange(len(models))

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

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), dpi=300)

# Panel A: Test perplexity
axes[0].bar(x, test_ppl, width=0.62)
axes[0].set_ylabel("Test perplexity")
axes[0].set_title("(a) Language modeling quality")
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
axes[0].set_axisbelow(True)
axes[0].set_ylim(0, max(test_ppl) * 1.18)

for i, v in enumerate(test_ppl):
    axes[0].text(i, v + max(test_ppl) * 0.025, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)

# Panel B: Peak GPU memory
axes[1].bar(x, peak_mem, width=0.62)
axes[1].set_ylabel("Peak GPU memory (MB)")
axes[1].set_title("(b) Training memory")
axes[1].set_xticks(x)
axes[1].set_xticklabels(models)
axes[1].grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
axes[1].set_axisbelow(True)
axes[1].set_ylim(0, max(peak_mem) * 1.18)

for i, v in enumerate(peak_mem):
    axes[1].text(i, v + max(peak_mem) * 0.025, f"{v:.0f}",
                 ha="center", va="bottom", fontsize=8)

# Clean NeurIPS-style figure appearance
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(w_pad=1.7)

pdf_path = OUT_DIR / "wikitext103_controlled_comparison.pdf"
png_path = OUT_DIR / "wikitext103_controlled_comparison.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")