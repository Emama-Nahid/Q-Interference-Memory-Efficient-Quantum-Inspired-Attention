from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parent

datasets = ["WikiText-103", "TinyStories", "pile-10k", "small-C4"]

models = ["GPT-Neo-125M", "OPT-125M", "Q-Interference"]
short = {
    "GPT-Neo-125M": "Neo",
    "OPT-125M": "OPT",
    "Q-Interference": "Q-Int",
}

colors = {
    "GPT-Neo-125M": "#0072B2",
    "OPT-125M": "#D55E00",
    "Q-Interference": "#009E73",
}

test_ppl = {
    "WikiText-103": {"GPT-Neo-125M": 19.4386, "OPT-125M": 20.4278, "Q-Interference": 24.1718},
    "TinyStories": {"GPT-Neo-125M": 4.5713, "OPT-125M": 4.1443, "Q-Interference": 5.6941},
    "pile-10k": {"GPT-Neo-125M": 10.9116, "OPT-125M": 16.2376, "Q-Interference": 116.9275},
    "small-C4": {"GPT-Neo-125M": 34.6512, "OPT-125M": 32.4405, "Q-Interference": 246.9012},
}

peak_mem = {
    "WikiText-103": {"GPT-Neo-125M": 4097.91, "OPT-125M": 4213.64, "Q-Interference": 4227.14},
    "TinyStories": {"GPT-Neo-125M": 9934.54, "OPT-125M": 6219.52, "Q-Interference": 4227.14},
    "pile-10k": {"GPT-Neo-125M": 9934.54, "OPT-125M": 6219.52, "Q-Interference": 4227.14},
    "small-C4": {"GPT-Neo-125M": 9931.54, "OPT-125M": 6219.52, "Q-Interference": 4227.14},
}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Column layout
x_dataset = 0.03
x_quality = 0.24
x_memory = 0.62
w_block = 0.33

# Header
ax.text(x_dataset, 0.93, "Dataset", weight="bold", fontsize=9, va="center")
ax.text(x_quality, 0.93, "Test perplexity rank", weight="bold", fontsize=9, va="center")
ax.text(x_memory, 0.93, "Peak memory rank", weight="bold", fontsize=9, va="center")
ax.text(x_quality, 0.885, "lower is better", fontsize=7.5, color="0.35", va="center")
ax.text(x_memory, 0.885, "lower is better", fontsize=7.5, color="0.35", va="center")

def draw_rank_pills(ax, x0, y0, ranking, metric_type):
    pill_w = 0.102
    pill_h = 0.055
    gap = 0.006

    for idx, (model, value) in enumerate(ranking):
        x = x0 + idx * (pill_w + gap)
        edge = colors[model]

        # Light fill only for Q-Interference to make our model easy to track.
        face = "#E8F5F0" if model == "Q-Interference" else "#FFFFFF"

        patch = FancyBboxPatch(
            (x, y0 - pill_h / 2),
            pill_w,
            pill_h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)

        if metric_type == "ppl":
            val_text = f"{value:.2f}"
        else:
            val_text = f"{value/1000:.1f}k"

        ax.text(
            x + pill_w / 2,
            y0,
            f"{idx + 1} {short[model]}\n{val_text}",
            ha="center",
            va="center",
            fontsize=7.4,
            linespacing=0.95,
        )

row_ys = [0.78, 0.61, 0.44, 0.27]

for row_id, (dataset, y) in enumerate(zip(datasets, row_ys)):
    # Alternating row background
    if row_id % 2 == 0:
        ax.add_patch(Rectangle((0.015, y - 0.07), 0.96, 0.13, facecolor="#F7F7F7", edgecolor="none"))

    ax.text(x_dataset, y, dataset, fontsize=8.5, va="center")

    q_rank = sorted(test_ppl[dataset].items(), key=lambda x: x[1])
    m_rank = sorted(peak_mem[dataset].items(), key=lambda x: x[1])

    draw_rank_pills(ax, x_quality, y, q_rank, "ppl")
    draw_rank_pills(ax, x_memory, y, m_rank, "mem")

# Section divider lines
ax.plot([0.015, 0.975], [0.86, 0.86], color="0.25", linewidth=0.8)
ax.plot([0.015, 0.975], [0.18, 0.18], color="0.25", linewidth=0.8)

# Legend
legend_y = 0.09
legend_xs = [0.25, 0.43, 0.61]
for x, model in zip(legend_xs, models):
    ax.add_patch(FancyBboxPatch(
        (x, legend_y - 0.018),
        0.028,
        0.036,
        boxstyle="round,pad=0.004,rounding_size=0.006",
        linewidth=1.0,
        edgecolor=colors[model],
        facecolor="#E8F5F0" if model == "Q-Interference" else "#FFFFFF",
    ))
    ax.text(x + 0.038, legend_y, model, fontsize=8, va="center", ha="left")

ax.text(
    0.5,
    0.035,
    "Ranks are computed separately within each dataset and metric. External models are contextual references.",
    ha="center",
    va="center",
    fontsize=7.5,
    color="0.35",
)

fig.tight_layout()

pdf_path = OUT_DIR / "external_reference_scorecard.pdf"
png_path = OUT_DIR / "external_reference_scorecard.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight")

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")