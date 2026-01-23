import os
from typing import Callable, ParamSpec, TypeVar, Concatenate
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from functools import wraps

from IPython.display import Video

from ..io.write import write
from ..utils import replicate_callable


def get_ipynb_file_name(
    env: str = 'IPYNB_FILE_NAME',
    key: str = '__vsc_ipynb_file__',
    ext: bool = False,
) -> str:
    
    ipynb_file_path = os.environ.get(env)
    if ipynb_file_path is None:
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None:
                _globals = ip.user_global_ns 
        except Exception:
            _globals = globals()
        ipynb_file_path = _globals[key]

    ipynb_file_path = os.path.basename(ipynb_file_path)
    if ext:
        return ipynb_file_path
    else:
        return os.path.splitext(ipynb_file_path)[0]
    

P = ParamSpec('P')
Q = ParamSpec('Q')
R = TypeVar('R')
def _save_figure(
    func: Callable[Concatenate[Figure, P], R] 
    | Callable[Concatenate[FuncAnimation, Q], R],
):
    def _(
        file_name: str,
        dir_path: str | None = './figures',
        prefix_ipynb: bool = True,
        get_path: bool = False,
    ) -> Callable[Concatenate[Figure, P], R | str] | Callable[Concatenate[FuncAnimation, Q], R | str]:

        if prefix_ipynb:
            file_name = f'{get_ipynb_file_name()}_{file_name}'
        if dir_path is not None:
            file_name = f'{dir_path}/{file_name}'

        @wraps(func)
        def __(obj, **kwargs):
            if isinstance(obj, Figure):
                _kwargs = dict(
                    file_ext=('pdf', 'png'), 
                    close=False,
                    pickle=False,
                )
                kwargs.update(_kwargs)
            write(obj, file_name, **kwargs)
            if get_path:
                return file_name
        return __
    return _


@replicate_callable(_save_figure(write))
def save_figure():
    """
    For interactive use in a `.ipynb` file. \\
    See `lucifex.io.write` for the general purpose `write` function. \\
    Default argument values from `write` are overridden and 
    File name dynamically created from notebook can be returned with `get_path=True`.
    """
    pass


def display_animation(
    anim_path: str,
    embed: bool = True, 
    loop: bool = True,
    width: int = 600,
    ext: str = '.mp4',
    **kwargs,
):
    if anim_path[:-4] != len(ext):
        anim_path = f'{anim_path}{ext}'
    if loop:
        _kwargs = dict(html_attributes="controls loop")
        kwargs.update(_kwargs)
    return Video(anim_path, embed=embed, width=width, **kwargs)