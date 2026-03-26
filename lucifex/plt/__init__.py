from .anim import create_animation
from .bar import plot_bar
from .colormap import plot_colormap, plot_contours, plot_colormap_multifigure
from .line import plot_line, plot_twin_lines, plot_stacked_lines, plot_line_multifigure
from .mesh import plot_mesh
from .scatter import plot_scatter, scatter_size
from .sparsity import plot_sparsity
from .vector import plot_quiver, plot_streamlines
from .ipynb import save_figure, display_animation, display_figure, get_ipynb_file_name, set_ipynb_variable
from .utils import configure_matplotlib, create_multifigure, set_axes, create_cycler


configure_matplotlib(
    ('text', dict(usetex=True)), 
    ('font', dict(family="serif")),
)
