from pathlib import Path
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent

# Controlled WikiText-103 results
ppl = {
    "Baseline GPT": 24.6534,
    "Quantum-Paper baseline": 60.5367,
    "Q-Interference": 24.1718,
}

mem = {
    "Baseline GPT": 8055.76,
    "Quantum-Paper baseline": 7199.26,
    "Q-Interference": 4227.14,
}

colors = {
    "Baseline GPT": "#4C566A",
    "Quantum-Paper baseline": "#B48EAD",
    "Q-Interference": "#1B9E77",
}

markers = {
    "Baseline GPT": "o",
    "Quantum-Paper baseline": "s",
    "Q-Interference": "D",
}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def draw_clean_ruler(ax, values, xlim, xticks, xlabel, title, label_specs):
    ax.set_xlim(*xlim)
    ax.set_ylim(-1.05, 1.05)

    # Main ruler line
    ax.hlines(0, xlim[0], xlim[1], color="black", linewidth=1.1, zorder=1)

    # Points
    for model, value in values.items():
        ax.scatter(
            value,
            0,
            s=110 if model == "Q-Interference" else 85,
            marker=markers[model],
            color=colors[model],
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )

    # Manual labels placed in fixed lanes to avoid collision
    for spec in label_specs:
        model = spec["model"]
        value = values[model]
        text_x = spec["text_x"]
        text_y = spec["text_y"]
        ha = spec["ha"]

        ax.annotate(
            spec["label"],
            xy=(value, 0),
            xytext=(text_x, text_y),
            ha=ha,
            va="center",
            fontsize=8.5,
            fontweight="bold" if model == "Q-Interference" else "normal",
            color=colors[model] if model == "Q-Interference" else "#222222",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor=colors[model],
                linewidth=0.85,
            ),
            arrowprops=dict(
                arrowstyle="-",
                linewidth=0.85,
                color=colors[model],
                shrinkA=2,
                shrinkB=5,
            ),
            zorder=5,
        )

    # Direction cue placed far above the ruler
    cue_x0 = xlim[0] + 0.03 * (xlim[1] - xlim[0])
    cue_x1 = xlim[0] + 0.17 * (xlim[1] - xlim[0])
    cue_y = 0.82
    ax.annotate(
        "",
        xy=(cue_x0, cue_y),
        xytext=(cue_x1, cue_y),
        arrowprops=dict(arrowstyle="->", linewidth=0.9, color="#555555"),
        annotation_clip=False,
    )
    ax.text(
        cue_x1 + 0.01 * (xlim[1] - xlim[0]),
        cue_y,
        "lower is better",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#555555",
    )

    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_xticks(xticks)
    ax.set_yticks([])

    ax.grid(axis="x", linestyle=":", linewidth=0.75, alpha=0.60)
    ax.set_axisbelow(True)

    # Clean axes
    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 2))


fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.65), dpi=300)
fig.subplots_adjust(hspace=0.78, left=0.10, right=0.98, top=0.90, bottom=0.15)

# Panel A labels
ppl_label_specs = [
    {
        "model": "Q-Interference",
        "label": "Q-Interference\n24.17",
        "text_x": 22.0,
        "text_y": 0.50,
        "ha": "left",
    },
    {
        "model": "Baseline GPT",
        "label": "Baseline GPT\n24.65",
        "text_x": 31.0,
        "text_y": -0.58,
        "ha": "left",
    },
    {
        "model": "Quantum-Paper baseline",
        "label": "Quantum-Paper baseline\n60.54",
        "text_x": 58.5,
        "text_y": 0.50,
        "ha": "right",
    },
]

draw_clean_ruler(
    axes[0],
    ppl,
    xlim=(18, 64),
    xticks=[20, 30, 40, 50, 60],
    xlabel="Test perplexity",
    title="(a) Test perplexity on WikiText-103",
    label_specs=ppl_label_specs,
)

# Panel B labels
mem_label_specs = [
    {
        "model": "Q-Interference",
        "label": "Q-Interference\n4227 MB",
        "text_x": 4550,
        "text_y": 0.50,
        "ha": "left",
    },
    {
        "model": "Quantum-Paper baseline",
        "label": "Quantum-Paper baseline\n7199 MB",
        "text_x": 6900,
        "text_y": -0.58,
        "ha": "right",
    },
    {
        "model": "Baseline GPT",
        "label": "Baseline GPT\n8056 MB",
        "text_x": 7850,
        "text_y": 0.50,
        "ha": "right",
    },
]

draw_clean_ruler(
    axes[1],
    mem,
    xlim=(3900, 8400),
    xticks=[4000, 5000, 6000, 7000, 8000],
    xlabel="Peak GPU memory (MB)",
    title="(b) Peak training memory on WikiText-103",
    label_specs=mem_label_specs,
)

fig.text(
    0.5,
    0.045,
    "Each marker is one internal model. Labels show exact values from the controlled WikiText-103 result table.",
    ha="center",
    va="center",
    fontsize=8.3,
    color="#444444",
)

pdf_path = OUT_DIR / "wikitext103_controlled_ruler_clean.pdf"
png_path = OUT_DIR / "wikitext103_controlled_ruler_clean.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")