import os
from pathlib import Path
from typing import Any
from types import EllipsisType
from typing_extensions import Self

from petsc4py import PETSc
from slepc4py import SLEPc

from lucifex.utils.py_utils import FrozenDict


DEFAULT_JIT_DIR = os.path.abspath(
    os.path.join(
        __file__,
        '../../..',
        '__jit__',
    )
)


class Options(FrozenDict[str, Any]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def default(cls) -> Self:
        try:
            return cls()
        except TypeError:
            raise TypeError('Arguments must all have default values.')
        
    def __getattr__(self, key: str) -> Any:
        if key not in self._dict:
            raise AttributeError(key)
        return self._dict[key]
    

class OptionsPETSc(Options):
    ksp_type: str
    pc_type: str
    ksp_rtol: float
    ksp_atol: float 
    ksp_divtol: float 
    ksp_max_it: int

    def __init__(
        self, 
        ksp_type: str = "gmres",
        pc_type: str = "ilu",
        ksp_rtol: float = 1e-06,
        ksp_atol: float = 1e-10,
        ksp_divtol: float = 1e4,
        ksp_max_it: int = 10000,
        **kwargs: Any,
    ):
        super().__init__(
            ksp_type=ksp_type,
            pc_type=pc_type,
            ksp_rtol=ksp_rtol,
            ksp_atol=ksp_atol,
            ksp_divtol=ksp_divtol,
            ksp_max_it=ksp_max_it,
            **kwargs,
        )


class OptionsSLEPc(Options):
    eps_type: str
    eps_target: float 
    eps_nev: int
    eps_ncv: int
    eps_mpd: int
    eps_which: str
    eps_tol: float
    eps_max_it: int
    
    def __init__(
        self, 
        eps_type: str = 'krylovschur',
        eps_target: float = 0.0,
        eps_nev: int = 5,
        eps_ncv: int = PETSc.DECIDE,
        eps_mpd: int = PETSc.DECIDE,
        eps_which: str = 'largest_magnitude',
        eps_tol: float = 1e-7,
        eps_max_it: int = 100,
        **kwargs: Any,
    ):
        super().__init__(
            eps_type=eps_type,
            eps_target=eps_target,
            eps_nev=eps_nev,
            eps_ncv=eps_ncv,
            eps_mpd=eps_mpd,
            eps_which=eps_which,
            eps_tol=eps_tol,
            eps_max_it=eps_max_it,
            **kwargs,
        )
    

def set_from_options(
    solver: PETSc.KSP | SLEPc.EPS,
    options: dict | OptionsPETSc | OptionsSLEPc,
) -> None:
    prefix = solver.getOptionsPrefix()
    _options = PETSc.Options()
    _options.prefixPush(prefix)
    for k, v in options.items():
        _options[k] = v
    _options.prefixPop()
    solver.setFromOptions()


class OptionsJIT(Options): 
    """See also `dolfinx.jit.DOLFINX_DEFAULT_JIT_OPTIONS`"""
    def __init__(
        self,
        cache_dir: str | EllipsisType | None = None,
        cffi_debug: bool = False,
        cffi_extra_compile_args: list[str] | None = None,
        cffi_verbose: bool = False,
        cffi_libraries: list | None = None,
        timeout: int | float = 10,
        **kwargs: Any,
    ):
        if cache_dir is Ellipsis:
            cache_dir = DEFAULT_JIT_DIR
        if cache_dir is None:
            cache_dir = os.getenv("XDG_CACHE_HOME", default=Path.home().joinpath(".cache")) / Path("fenics")
        if cffi_extra_compile_args is None:
            cffi_extra_compile_args = ["-O2", "-g0"]
        super().__init__(
            cache_dir=cache_dir,
            cffi_debug=cffi_debug,
            cffi_extra_compile_args=cffi_extra_compile_args,
            cffi_verbose=cffi_verbose,
            cffi_libraries=cffi_libraries,
            timeout=timeout,
            **kwargs,
        )


class OptionsFFCX(Options):  
    """See also `dolfinx.jit` module"""
    pass