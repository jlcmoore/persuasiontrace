"""Generate a fake persuasion histogram-style plot.

This script creates a 5x4 inch bar-style histogram with fixed x-axis category
labels and random positive values in the range [0.2, 0.4].
"""

import random

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def main() -> None:
    """Create and save a fake persuasion chart.

    Returns:
        None. Saves a PNG file to analysis/fake_llm_persuasion_histogram.png.
    """
    categories = [
        "Single\nturn",
        "Multi\nturn",
        "Personal\ntopic",
        "General\ntopic",
        "Text",
        "Audio",
    ]
    values = [random.uniform(0.2, 0.4) for _ in categories]
    color = "b"

    figure = Figure(figsize=(4.5, 3))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(categories, values, color=color, edgecolor="black", linewidth=1.0)
    axis.set_ylabel("How much\nMORE Persuasive\nAre LLMs?")
    axis.set_yticks([])
    axis.set_ylim(0.0, 0.45)

    figure.tight_layout()
    figure.savefig("analysis/fake_llm_persuasion_histogram.png", dpi=200)


if __name__ == "__main__":
    main()
