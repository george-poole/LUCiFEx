from collections.abc import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .utils import optional_ax, set_legend, set_axes


@optional_ax
def plot_bar(
    fig: Figure,
    ax: Axes, 
    y_data: Iterable[float | Iterable[float]],
    x_ticks: Iterable[str] = None,
    y_label: str | None = None,
    x_label: str | None = None,
    title: str | None = None,
    widths: float | Iterable[float] = 1.0,
    pad: float = 0.5,
    colors: str | Iterable[str | Iterable[str]] = "lightgrey",
    edge_colors: str | Iterable[str] = "black",
    legend_labels: Iterable[str | float | int] | None = None,
    legend_title: str | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    set_axes(ax, x_label=x_label, y_label=y_label, title=title)

    y_data = [[y] if not isinstance(y, Iterable) else y for y in y_data]

    if not isinstance(widths, Iterable):
        widths = [widths] * len(y_data)
    if isinstance(colors, str):
        colors = [colors] * len(y_data)
    if isinstance(edge_colors, str):
        edge_colors = [edge_colors] * len(y_data)

    widths_block = [w * len(y) for w, y in zip(widths, y_data)]

    centres = [0.0] * len(y_data)
    for i in range(1, len(y_data)):
        centres[i] = (
            centres[i - 1] + pad + 0.5 * (widths_block[i] + widths_block[i - 1])
        )
    ax.set_xticks(centres, x_ticks)

    for y, xc, w, c, ec in zip(
        y_data, centres, widths, colors, edge_colors, strict=True
    ):
        r = (len(y) - 1) / 2
        x = [xc + i * w for i in np.arange(-r, r + 1)]
        ax.bar(x, y, width=w, color=c, edgecolor=ec)

    if legend_labels:
        all_colors = []
        for c in colors:
            if isinstance(c, str):
                all_colors.append(c)
            else:
                all_colors.extend(c)
        all_colors = list(dict.fromkeys(all_colors))
        handles = [Patch(facecolor=i) for i in all_colors]
        set_legend(ax, legend_labels, legend_title, handles)

    return fig, ax
