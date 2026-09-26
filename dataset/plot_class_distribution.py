from __future__ import annotations

from collections import Counter
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent
OUTPUT = DATASET_DIR / "class_distribution.svg"


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text(x: float, y: float, value: object, size: int = 24, *, anchor: str = "start",
         weight: int = 400, fill: str = "#172033") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{esc(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str, *, rx: float = 0,
         stroke: str = "none", stroke_width: float = 0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.1f}"/>'
    )


def mix(low: tuple[int, int, int], high: tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    rgb = tuple(round(a + (b - a) * t) for a, b in zip(low, high))
    return "#" + "".join(f"{value:02x}" for value in rgb)


def read_counts() -> tuple[list[Path], dict[str, Counter[int]]]:
    files = sorted(DATASET_DIR.glob("batch*.dat"), key=lambda p: int(p.stem[5:]))
    counts: dict[str, Counter[int]] = {}
    for path in files:
        counter: Counter[int] = Counter()
        with path.open(encoding="ascii") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                label = int(line.split(maxsplit=1)[0])
                counter[label] += 1
        counts[path.stem] = counter
    return files, counts


files, batch_counts = read_counts()
classes = sorted({label for counter in batch_counts.values() for label in counter})
overall: Counter[int] = Counter()
for counter in batch_counts.values():
    overall.update(counter)

total = sum(overall.values())
majority = max(classes, key=overall.get)
minority = min(classes, key=overall.get)
imbalance_ratio = overall[majority] / overall[minority]

W, H = 1800, 1800
svg: list[str] = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    rect(0, 0, W, H, "#f6f8fc"),
    text(80, 72, "Class Distribution — 10 Batches", 38, weight=700),
    text(80, 110, "13,910 samples • 6 classes • 128 features", 22, fill="#65708a"),
]

# Overall distribution card and chart.
svg.append(rect(55, 145, 1160, 550, "#ffffff", rx=24, stroke="#e1e6f0", stroke_width=2))
svg.append(text(90, 195, "Overall distribution", 28, weight=700))

plot_x, plot_y, plot_w, plot_h = 120, 245, 1030, 350
max_count = max(overall.values())
for tick in range(0, 3501, 500):
    y = plot_y + plot_h - tick / 3500 * plot_h
    svg.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#e8ecf4" stroke-width="2"/>')
    svg.append(text(plot_x - 18, y + 7, f"{tick:,}", 17, anchor="end", fill="#77829a"))

bar_gap = 32
bar_w = (plot_w - bar_gap * (len(classes) - 1)) / len(classes)
for idx, label in enumerate(classes):
    count = overall[label]
    height = count / 3500 * plot_h
    x = plot_x + idx * (bar_w + bar_gap)
    y = plot_y + plot_h - height
    color = "#ef6a6a" if label == majority else "#f0ad4e" if label == minority else "#4f7de8"
    svg.append(rect(x, y, bar_w, height, color, rx=10))
    svg.append(text(x + bar_w / 2, y - 36, f"{count:,}", 20, anchor="middle", weight=700))
    svg.append(text(x + bar_w / 2, y - 12, f"{count / total:.1%}", 17, anchor="middle", fill="#65708a"))
    svg.append(text(x + bar_w / 2, plot_y + plot_h + 34, f"Class {label}", 19, anchor="middle", weight=600))

# Summary card.
svg.append(rect(1245, 145, 500, 550, "#ffffff", rx=24, stroke="#e1e6f0", stroke_width=2))
svg.append(text(1290, 200, "Imbalance summary", 28, weight=700))
svg.append(rect(1290, 232, 410, 84, "#fff3dc", rx=16))
svg.append(text(1315, 270, "MODERATE IMBALANCE", 24, weight=700, fill="#9a5b00"))
svg.append(text(1315, 297, "overall (max/min heuristic)", 17, fill="#9a6f30"))

summary_rows = [
    ("Majority class", f"Class {majority} — {overall[majority]:,} ({overall[majority] / total:.1%})"),
    ("Minority class", f"Class {minority} — {overall[minority]:,} ({overall[minority] / total:.1%})"),
    ("Max/min ratio", f"{imbalance_ratio:.2f} : 1"),
    ("Missing by batch", "Class 6 absent in batches 3–5"),
]
for i, (label, value) in enumerate(summary_rows):
    y = 365 + i * 57
    svg.append(text(1290, y, label, 18, fill="#77829a"))
    svg.append(text(1698, y, value, 18, anchor="end", weight=600))

# Per-batch heatmap.
svg.append(rect(55, 730, 1690, 985, "#ffffff", rx=24, stroke="#e1e6f0", stroke_width=2))
svg.append(text(90, 790, "Distribution within each batch", 28, weight=700))
svg.append(text(90, 827, "Each cell shows count and share of that batch", 18, fill="#65708a"))

grid_x, grid_y = 285, 890
cell_w, cell_h = 225, 58
row_gap = 8
for col, label in enumerate(classes):
    svg.append(text(grid_x + col * cell_w + cell_w / 2, grid_y - 17, f"Class {label}", 19, anchor="middle", weight=600))

low, high = (235, 243, 255), (40, 93, 188)
for row, path in enumerate(files):
    batch = path.stem
    counter = batch_counts[batch]
    batch_total = sum(counter.values())
    y = grid_y + row * (cell_h + row_gap)
    svg.append(text(grid_x - 25, y + 25, batch.replace("batch", "Batch "), 18, anchor="end", weight=600))
    for col, label in enumerate(classes):
        count = counter[label]
        share = count / batch_total
        # Scale so that 30% within a six-class problem is visually saturated.
        color = mix(low, high, share / 0.30)
        x = grid_x + col * cell_w
        svg.append(rect(x, y, cell_w - 8, cell_h, color, rx=6))
        label_color = "#ffffff" if share >= 0.20 else "#24324a"
        svg.append(text(x + (cell_w - 8) / 2, y + 36, f"{count:,}  ({share:.1%})", 16, anchor="middle", weight=600, fill=label_color))

legend_y = 1635
svg.append(text(95, legend_y, "Cell color = within-batch share", 17, fill="#65708a"))
for i in range(100):
    svg.append(rect(360 + i * 5, legend_y - 17, 5.5, 18, mix(low, high, i / 99)))
svg.append(text(350, legend_y, "0%", 15, anchor="end", fill="#65708a"))
svg.append(text(875, legend_y, "30%+", 15, fill="#65708a"))
svg.append(text(1700, legend_y, "Source: batch1.dat–batch10.dat", 16, anchor="end", fill="#8a94aa"))

svg.append("</svg>")
OUTPUT.write_text("\n".join(svg), encoding="utf-8")
print(OUTPUT)
