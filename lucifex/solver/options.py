import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing_extensions import Self

from ..utils.py_utils import classproperty, MultipleDispatchTypeError


class Options(dict):
    @classproperty
    def default(cls) -> Self:
        return cls()


@dataclass
class OptionsPETSc(Options):
    ksp_type: str = "preonly"
    pc_type: str = "lu"
    ksp_max_it: int = 1000
    ksp_atol: float = 1e-10
    ksp_rtol: float = 1e-06


@dataclass
class OptionsJIT(Options): 
    """See also `dolfinx.jit.DOLFINX_DEFAULT_JIT_OPTIONS` dictionary"""
    cache_dir: str = field(default_factory=lambda: os.getenv("XDG_CACHE_HOME", default=Path.home().joinpath(".cache")) / Path("fenics"))
    cffi_debug: bool = False
    cffi_extra_compile_args: list[str] = field(default_factory=lambda:["-O2", "-g0"])
    cffi_verbose: bool = False
    cffi_libraries: list | None = None
    timeout: int | float = 10


@dataclass
class OptionsFFCX(Options):  
    """See also `dolfinx.jit` module"""
    ... # TODO


def options_dict(obj: Options | dict) -> dict:
    if isinstance(obj, Options):
        return asdict(obj) | dict(obj)
    elif isinstance(obj, dict):
        return obj
    else:
        raise MultipleDispatchTypeError(obj)
