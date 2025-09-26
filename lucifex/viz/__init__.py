""" Module for creating line, colormap and scatter plots """

from .anim import create_animation
from .bar import plot_bar
from .colormap import plot_colormap, plot_contours
from .line import plot_line, plot_twin_lines, plot_stacked_lines
from .mesh import plot_mesh
from .scatter import plot_scatter
from .utils import exponential_notation, tex, create_mosaic_figure, set_axes
from .vector import plot_quiver, plot_streamlines
