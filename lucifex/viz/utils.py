from inspect import signature
from typing import Callable, Iterable, ParamSpec, Concatenate, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# defaults for `matplotlib`
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
LW = 0.75
MS = 6.0
COLORS = ["black", "blue", "limegreen", "red", "darkorange", "fuchsia"]


P = ParamSpec('P')
def optional_fig(
    plot_func: Callable[Concatenate[Figure, P], None],
) -> Callable[P, tuple[Figure, Axes]] | Callable[Concatenate[Figure, Axes, P], None]:
    return _optional_fig_and_or_ax(plot_func)


P = ParamSpec('P')
def optional_ax(
    plot_func: Callable[Concatenate[Axes, P], None],
) -> Callable[P, tuple[Figure, Axes]] | Callable[Concatenate[Figure, Axes, P], None]:
    return _optional_fig_and_or_ax(plot_func)


P = ParamSpec('P')
def optional_fig_ax(
    plot_func: Callable[Concatenate[Figure, Axes, P], None],
) -> Callable[P, tuple[Figure, Axes]] | Callable[Concatenate[Figure, Axes, P], None]:
    return _optional_fig_and_or_ax(plot_func)


P = ParamSpec('P')
def _optional_fig_and_or_ax(
    plot_func: Callable[Concatenate[Figure, Axes, P], None] | Callable[Concatenate[Figure, P], None] | Callable[Concatenate[Axes, P], None],
):  
    if tuple(signature(plot_func).parameters.values())[0].annotation is Axes:
        ax_only = True
    else:
        ax_only = False

    # @wraps(func)
    def _inner(*args, **kwargs):
        if isinstance(args[0], Figure) and isinstance(args[1], Axes):
            fig, ax, *_args = args
            # routine mutating existing `Figure, Axes` objects and returning `None`
            if ax_only:
                return plot_func(ax, *_args, **kwargs)
            else:
                return plot_func(fig, ax, *_args, **kwargs)
        else:
            # function creating and returning new `Figure, Axes` objects
            fig, ax = plt.subplots()
            if ax_only:
                plot_func(ax, *args, **kwargs)
            else:
                plot_func(fig, ax, *args, **kwargs)
            return fig, ax

    return _inner


def set_axes(
    ax: Axes,
    x_lims: tuple[float, float] | Iterable[float] | None = None,
    y_lims: tuple[float, float] | Iterable[float] | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    aspect: float | Literal['auto', 'equal'] | None = None,
    tex: bool = False,
) -> None:
    # creating or overwriting existing axis labels and title
    if tex:
        tex_optional = lambda s: texify(s)
    else:
        tex_optional = lambda s: s
        
    if x_lims is not None:
        ax.set_xlim((np.min(x_lims), np.max(x_lims)))
    if y_lims is not None:
        ax.set_ylim((np.min(y_lims), np.max(y_lims)))
    if x_label is not None:
        ax.set_xlabel(tex_optional(x_label))
    if y_label is not None:
        ax.set_ylabel(tex_optional(y_label))
    if title is not None:
        ax.set_title(tex_optional(title))
    if aspect is not None:
        ax.set_aspect(aspect)


def set_legend(
    ax: Axes,
    labels: Iterable[str | float | int | None],
    title: str | None = None,
    handles=None,
    loc="upper left",
    bbox_to_anchor=(1, 1),
    frameon=False,
    tex: bool = True,
) -> None:

    if tex:
        tex_optional = lambda s: texify(s)
    else:
        tex_optional = lambda s: s
    
    labels = [tex_optional(i) if isinstance(i, str) else str(i) for i in labels]
    if handles is None:
        args = (labels, )
    else:
        args = (handles, labels)

    ax.legend(
        *args,
        title=tex_optional(title),
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=frameon,
    )


TEX_SYMBOLS = (
        # lower case Greek
        "alpha",
        "beta",
        "gamma",
        "delta",
        'epsilon',
        'zeta',
        'eta',
        'theta',
        'iota',
        'kappa',
        'lambda',
        'mu',
        'nu',
        'xi',
        'pi',
        'rho',
        'sigma',
        'tau',
        'upsilon',
        'phi',
        'chi',
        'psi',
        'omega',
        # upper case Greek
        'Gamma',
        'Delta',
        'Theta',
        'Lambda',
        'Xi',
        'Pi',
        'Sigma',
        'Upsilon',
        'Phi',
        'Psi',
        'Omega',
        # mathematical operations
        "times",
        "cdot",
    )


def texify(
    label: str | float | int | None,
    escape: str = "/",
) -> str:
    """Figure titles and axis labels are rendered in
    TeX by prepending and appending `$`. Mathematical symbols
    listed in `MATH_SYMBOLS` are prepended with a backslash.
    Otherwise, write e.g. `\\symbol` to provide the backslash.
    For plain text, write e.g. `\\text{...}`."""

    if isinstance(label, (float, int)):
        return f"${label}$"

    if label == "" or label is None:
        return ""

    if label[0] == "$" and label[-1] == "$":
        return label
    
    for char in (escape, '$'):
        if label[0] == char and label[-1] == char:
            return label[1:-1]

    for i in TEX_SYMBOLS:
        if (i in label) and (f"\\{i}" not in label):
            label = label.replace(i, f"\\{i}")

    label = f"${label}$"
    return label


def detexify(
    label: str | None,
    strip_wspace: bool = True,
) -> str:
    if label == "" or label is None:
        return ""
    if label[0] == '$' and label[-1] == '$':
        return detexify(label[1:-1])
    
    for i in TEX_SYMBOLS:
        if f'\\{i}' in label:
            label = label.replace(f'\\{i}', 'i')

    if strip_wspace:
        label = label.replace(' ', '')

    return label
    

def exponential_notation(
    number: float | int,
    n_digits: int = 3,
    ignore: Iterable[int] = (-1, 0, 1, 2),
    tex: bool = False,
) -> str:
    """Returns a TeX string representation of a number in the format `a × 10ⁿ`"""
    if np.isclose(number, 0):
        exponent = 0
        coeff = 0
    else:
        exponent = int(np.floor(np.log10(abs(number))))
        coeff = round(number / float(10**exponent), n_digits)

    if exponent in ignore:
        # exponents for which numbers are not converted to the format `a × 10ⁿ`
        # e.g. for (-1, 0 , 1, 2)
        # 0.123 -> 0.123
        # 1.23 -> 1.230
        # 12.3 -> 12.300
        # 123.0 -> 123.000
        label = f"{coeff * 10**exponent:.{n_digits}f}"
    else:
        # e.g. 1.230 × 10ⁿ
        label = f"{coeff:.{n_digits}f}\\times 10^{{{exponent:d}}}"

    if tex:
        label = f"${label}$"

    return label


def create_mosaic_figure(
    n_rows: int,
    n_cols: int,
    suptitle: str | None = None,
    figscale: float = 1.0,
    indexing: Literal['xy', 'ij', 'ji'] = 'xy',
    tex: bool = False,
    **kwargs,
) -> tuple[Figure, np.ndarray]:

    fig: Figure
    fig, axs = plt.subplots(
        n_rows, 
        n_cols,
        figsize=figscale * np.multiply((n_cols, n_rows), plt.rcParams["figure.figsize"]), 
        layout='compressed',
        **kwargs,
    )

    if suptitle:
        if tex:
            suptitle = texify(suptitle)
        ax: Axes = axs.flat[n_cols - 1]
        ax.text(1.0, 1.25, texify(suptitle), horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

    if indexing == 'xy':
        axs = axs.T[:, ::-1]
    elif indexing == 'ij':
        axs = axs.T

    return fig, axs


# def combine_figures(
#     figs: Iterable[str | Figure | tuple[Figure, Axes]],
#     *,
#     n_rows: int | None = None,
#     n_cols: int | None = None,
#     close: bool = True,
#     **write_kwargs,
# ) -> Image:
    
#     n_fig = len(figs)

#     if n_cols is None:
#         n_cols = min(3, n_fig)
#     if n_rows is None:
#         n_rows = max(n_fig // n_cols, int(np.ceil(n_fig / n_cols)))
    
#     images = []
#     for i, fig in enumerate(figs):
#         if isinstance(fig, str):
#             _fig_path = fig
#             _temp = False
#         else:
#             if isinstance(fig, tuple):
#                 fig = fig[0]
#             _fig_ext = 'jpg'
#             _fig_path = file_path_ext(os.getcwd(), f'_{combine_figures.__name__}_{i}', _fig_ext)
#             write(fig, _fig_path, close=close, pickle=False, file_ext=_fig_ext, **write_kwargs)
#             _temp = True
#         _im = Image.open(_fig_path).convert('RGB')
#         images.append(_im)
#         if _temp:
#             os.remove(_fig_path)

#     widths, heights = zip(*(i.size for i in images))
#     row_widths = [sum(widths[i: i + n_cols]) for i in range(n_fig - n_cols)] #FIXME
#     row_heights = [max(heights[i: i + n_cols]) for i in range(n_fig - n_cols)]

#     image = Image.new('RGB', (max(row_widths), sum(row_heights)))

#     x_offset, y_offset = 0, 0
#     for i, im in enumerate(images):
#         image.paste(im, (x_offset, y_offset))
#         x_offset += im.size[0]
#         y_offset += im.size[1]
#         # if i % n_cols == 0:
#         #     x_offset = 0

#     return image



# def save_figure(
#     fig: Figure,
#     file_name: str,
#     dir_path: str = ".",
#     file_ext: str = "pdf",
#     close: bool = False,
#     bbox_inches: str | None = "tight",
#     dpi: int = 150,
#     **savefig_kwargs,
# ) -> None:
#     os.makedirs(dir_path, exist_ok=True)
#     if file_name == '':
#         file_name = fig.get_url()
#         if file_name is None:
#             raise ValueError('Must provide a non-empty file name string.')

#     fig_path = file_path_ext(dir_path, file_name, file_ext)
#     fig.savefig(fig_path, bbox_inches=bbox_inches, dpi=dpi, **savefig_kwargs)
#     if close:
#         plt.close(fig)


# def save_animation(
#     anim: FuncAnimation,
#     file_name: str,
#     dir_path: str = ".",
#     file_ext: str = "mp4",
#     writer: str = "ffmpeg",
#     fps: int | None = None,
#     dpi: int = 150,
#     bitrate: int | None = None,
#     **saveanim_kwargs,
# ) -> None:
#     os.makedirs(dir_path, exist_ok=True)
#     anim_path = file_path_ext(dir_path, file_name, file_ext)
#     anim.save(anim_path, writer, fps, dpi, bitrate=bitrate, **saveanim_kwargs)
