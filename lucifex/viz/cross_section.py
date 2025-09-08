from typing import Literal

import numpy as np
from dolfinx.fem import Function
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..utils import grid, as_index

from .colormap import plot_colormap
from .line import plot_line
from .utils import optional_fig_ax


@optional_fig_ax
def plot_cross_section(
    fig: Figure,
    ax: Axes, 
    f: Function,
    axis: str | Literal[0, 1, 2],
    value: float | None = None,
    label: str | None = None,
    fraction: bool = True,
    axis_names: tuple[str, ...] = ("x", "y", "z"),
    use_cache: bool = True,
) -> None:
    
    if fraction:
        f_fraction = value
        if f_fraction < 0 or f_fraction > 1:
            raise ValueError("Fraction must be in interval [0, 1]")
        f_value = None
    else:
        f_fraction = None
        f_value = value

    if not isinstance(axis, int):
        axis = axis_names.index(axis)

    if label is None:
        label = f.name

    x = grid(use_cache=True)(f.function_space.mesh)
    f_grid = grid(use_cache=use_cache)(f)
    dim = f.function_space.mesh.geometry.dim
    if dim == 2:
        xy_names = axis_names[:2]
        _plot_cross_section_line(
            fig, ax, f_grid, x, f_fraction, f_value, axis, label, xy_names
        )
    elif dim == 3:
        _plot_cross_section_colormap(
            fig, ax, f_grid, x, f_fraction, f_value, axis, label, axis_names,
        )
    else:
        raise ValueError


def _plot_cross_section_line(
    fig: Figure,
    ax: Axes, 
    f_grid,
    xy,
    y_fraction: float | None,
    y_value: float | int | None,
    y_index: Literal[0, 1],
    y_label: str,
    xy_names: tuple[str, str],
) -> None:
    y_axis = xy[y_index]
    x_axis = xy[(y_index + 1) % len(xy_names)]

    if y_value is not None:
        yaxis_index = as_index(y_axis, y_value)
    else:
        assert y_fraction is not None
        if np.isclose(y_fraction, 1):
            yaxis_index = -1
        else:
            yaxis_index = int(y_fraction * len(y_axis))
    y_value = y_axis[yaxis_index]

    if y_index == 0:
        y_label, x_label = xy_names
        cross_sec = f_grid[yaxis_index, :]
    elif y_index == 1:
        x_label, y_label = xy_names
        cross_sec = f_grid[:, yaxis_index]
    else:
        raise ValueError

    y_label = f"{y_label}({y_label} = {y_value:.2f})"
    plot_line(fig, ax, (x_axis, cross_sec), x_label=x_label, y_label=y_label)


def _plot_cross_section_colormap(
    fig: Figure,
    ax: Axes,
    f_grid,
    xyz: tuple,
    z_fraction: float | None,
    z_value: float | int | None,
    z_index: Literal[0, 1, 2],
    f_label: str,
    xyz_labels: tuple[str, str, str],
) -> None:
    z_axis = xyz[z_index]
    x_axis = xyz[(z_index + 1) % len(xyz_labels)]
    y_axis = xyz[(z_index + 2) % len(xyz_labels)]

    if z_value is not None:
        zaxis_index = as_index(z_axis, z_value)
    else:
        assert z_fraction is not None
        zaxis_index = int(z_fraction * len(z_axis))
    z_value_on_grid = z_axis[zaxis_index]

    if z_index == 0:
        z_label, x_label, y_label = xyz_labels
        cross_sec = f_grid[zaxis_index, :, :]
    elif z_index == 1:
        y_label, z_label, x_label = xyz_labels
        cross_sec = f_grid[:, zaxis_index, :]
    elif z_index == 2:
        x_label, y_label, z_label = xyz_labels
        cross_sec = f_grid[:, :, zaxis_index]
    else:
        raise ValueError

    title = f"{f_label}({z_label} = {z_value_on_grid:.2f})"
    plot_colormap(fig, ax, (cross_sec, x_axis, y_axis), title=title, x_label=x_label, y_label=y_label)


# def plot_cross_sections(
#     f_series: Iterable[Function],
#     t_series: Iterable[float],
#     field_name: str | None = None,
#     *,
#     x_fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
#     y_fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
#     xyz_names: tuple[str, str] = ("x", "y", "z"),
#     cycler: Literal["black", "color"] = "color",
#     legend_labels: Iterable[str | float | int] | None = None,
#     legend_title: str | None = None,
# ) -> tuple[Figure, Axes]:
#     if len(f_series) != len(t_series):
#         raise ValueError("Lengths of `f_series` and `t_series` must be the same")
#     if len(x_fractions) != len(y_fractions):
#         raise ValueError("Lengths of `x_fractions` and `y_fractions` must be the same")

#     n_rows = len(x_fractions)
#     n_cols = 2

#     fig, ax = subplots(n_rows, n_cols, layout="constrained")
#     prop_cycle = LINE_CYCLER[cycler]

#     # y-cross-section plots in the first column
#     for i, yf in enumerate(y_fractions):
#         axes: Axes = ax[i, 0]
#         axes.set_prop_cycle(prop_cycle)
#         for f in f_series:
#             plot_cross_section("y", f, None, y_fraction=yf, fig_ax=(fig, axes))
#         axes.autoscale(enable=True, axis="x", tight=True)
#         f_label = f"{field_name}({xyz_names[1]} = {yf:.2f}L_y)"
#         axes.set_ylabel(f_label)
#         if i != n_rows - 1:
#             axes.set_xlabel(None)
#             axes.set_xticks([])
#         else:
#             axes.set_xlabel(xyz_names[0])

#     # x-cross-section plots in the second column
#     for i, xf in enumerate(x_fractions):
#         axes: Axes = ax[i, 1]
#         axes.set_prop_cycle(prop_cycle)
#         for f in f_series:
#             plot_cross_section("x", f, field_name, x_fraction=xf, fig_ax=(fig, axes))
#         axes.autoscale(enable=True, axis="x", tight=True)
#         f_label = f"{field_name}({xyz_names[0]} = {xf:.2f}L_x)"
#         axes.set_ylabel(f_label)
#         if i != n_rows - 1:
#             axes.set_xlabel(None)
#             axes.set_xticks([])
#         else:
#             axes.set_xlabel(xyz_names[1])

#     if legend_labels:
#         set_legend(ax[-1, -1], legend_labels, legend_title)

#     return fig, ax

