import os
from typing import Callable, ParamSpec, TypeVar, Concatenate, Any
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from functools import wraps

from IPython.display import Image, Video, display

from ..io.write import write
from ..utils.py_utils import replicate_callable


T = TypeVar('T')
def set_ipynb_variable(
    env_key: str,
    default: T,
    silent: bool = False,
    as_type: Callable[[str], T] | None = None,
) -> T:
    """
    Optionally set a variable's value from the terminal with `export env_key=value`, 
    otherwise set to `default`.
    """
    _default = object()
    value = os.environ.get(env_key, _default)
    if value is _default:
        return default
    
    if as_type is None:
        as_type = eval

    if not silent:
        print(f"Environment variable `{env_key}={value}`")  

    value = as_type(value)
    if not isinstance(value, type(default)):
        raise TypeError

    return value
    

def get_ipynb_file_name(
    env_key: str = 'IPYNB_FILE_PATH',
    dict_key: str = '__vsc_ipynb_file__',
    ext: bool = False,
) -> str:
    
    ipynb_file_path = os.environ.get(env_key)
    if ipynb_file_path is None:
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None:
                _globals = ip.user_global_ns 
        except Exception:
            _globals = globals()
        ipynb_file_path = _globals[dict_key]

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
        prefix: bool | str = True,
        return_path: bool = False,
        mkdirs: bool = True,
        sep: str = '__',
        thumbnail: str | bool = False,
        **overwrite_kwargs: Any,
    ) -> Callable[Concatenate[Figure, P], R | str] | Callable[Concatenate[FuncAnimation, Q], R | str]:

        ipynb_name = get_ipynb_file_name()
        if prefix:
            if isinstance(prefix, str):
                _prefix = prefix
            else:
                _prefix = ipynb_name
            file_name = sep.join((_prefix, file_name))
        if dir_path is not None:
            file_name = os.path.join(dir_path, file_name)
        if mkdirs:
            os.makedirs(dir_path, exist_ok=True)
        if thumbnail is True:
            thumbnail = './thumbnails'

        @wraps(func)
        def __(fig_or_anim, **kwargs):
            if not isinstance(fig_or_anim, (Figure, FuncAnimation)):
                raise TypeError(f'Unexpected type {type(fig_or_anim)}')
            _kwargs = overwrite_kwargs
            if isinstance(fig_or_anim, Figure):
                _kwargs.update(
                    file_ext=('pdf', 'png'), 
                    close=False,
                    pickle=False,
                )
            _kwargs.update(kwargs)
            write(fig_or_anim, file_name, **_kwargs)
            if thumbnail:
                _kwargs.update(file_ext='png')
                write(fig_or_anim, os.path.join(thumbnail, ipynb_name), **_kwargs)
            if return_path:
                return file_name
        return __
    return _


@replicate_callable(_save_figure(write))
def save_figure():
    """
    For interactive use in a `.ipynb` file. \\
    See `lucifex.io.write` for the general purpose `write` function. \\
    Default argument values from `write` are overridden. \\
    Dynamically-created file path can be returned with `return_path=True`.
    """
    pass


def display_figure(
    fig_path: str,
    ext: str | None = '.png',
    call: bool = True,
    **kwargs,
) -> Image | Any:
    if ext and fig_path[:-4] != len(ext):
        fig_path = f'{fig_path}{ext}'
    img = Image(fig_path, **kwargs)
    if call:
        return display(img)
    else:
        return img


def display_animation(
    anim_path: str,
    embed: bool = True, 
    loop: bool = True,
    width: int = 600,
    ext: str | None = '.mp4',
    call: bool = False,
    **kwargs,
) -> Video | Any:
    _kwargs = {}
    if ext and anim_path[:-4] != len(ext):
        anim_path = f'{anim_path}{ext}'
    if loop:
        _kwargs.update(html_attributes="controls loop")
    _kwargs.update(kwargs)
    vid = Video(anim_path, embed=embed, width=width, **_kwargs)
    if call:
        return display(vid)
    else:
        return vid