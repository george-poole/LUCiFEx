""" Module for creating line, colormap and scatter plots """

from .anim import animate_colormap
from .bar import plot_bar
from .colormap import plot_colormap, plot_contours
from .line import plot_line, plot_twins, plot_lines_stack
from .mesh import plot_mesh
from .scatter import plot_xy_scatter, plot_xyz_scatter
from .utils import exponential_notation, texify, create_mosaic_figure, set_axes
from .vector import plot_quiver, plot_streamlines
