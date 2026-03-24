import os
from typing import Callable, ParamSpec, TypeVar, Concatenate, Any
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from functools import wraps

from IPython.display import Video

from ..io.write import write
from ..utils.py_utils import replicate_callable


T = TypeVar('T')
def set_ipynb_variable(
    env_key: str,
    default: T,
    report: bool = True,
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

    if report:
        print(f"Environment variable `{env_key}={value}`")  

    value = as_type(value)
    if not isinstance(value, type(default)):
        raise TypeError

    return value
    

def get_ipynb_file_name(
    env_key: str = 'IPYNB_FILE_NAME',
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
        thumbnail: bool = False,
        **defaults: Any,
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

        @wraps(func)
        def __(fig_or_anim, **kwargs):
            if isinstance(fig_or_anim, Figure):
                _kwargs = dict(
                    file_ext=('pdf', 'png'), 
                    close=False,
                    pickle=False,
                )
                _kwargs.update(defaults)
                _kwargs.update(kwargs)
            write(fig_or_anim, file_name, **_kwargs)
            if thumbnail:
                write(fig_or_anim, os.path.join(dir_path, ipynb_name), **_kwargs)
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
    File name dynamically created from notebook name can be returned with `return_path=True`.
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
    _kwargs = {}
    if anim_path[:-4] != len(ext):
        anim_path = f'{anim_path}{ext}'
    if loop:
        _kwargs.update(html_attributes="controls loop")
    _kwargs.update(kwargs)
    return Video(anim_path, embed=embed, width=width, **_kwargs)