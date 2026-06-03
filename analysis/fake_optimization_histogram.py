"""Generate a small fake optimization progress chart."""

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def main() -> None:
    """Create and save a small fake optimization chart.

    Returns:
        None. Saves a PNG file to analysis/fake_optimization_histogram.png.
    """
    categories = ["qwen", "+sft", "+grpo"]
    values = [0.24, 0.31, 0.38]

    figure = Figure(figsize=(3, 2.2))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(categories, values, color="b", edgecolor="black", linewidth=1.0)
    axis.set_title("You can optimize\nagainst it!")
    axis.set_ylabel("")
    axis.set_yticks([])
    axis.set_ylim(0.0, 0.45)

    figure.tight_layout()
    figure.savefig("analysis/fake_optimization_histogram.png", dpi=200)


if __name__ == "__main__":
    main()
