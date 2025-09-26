import os
from pathlib import Path
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any
from typing_extensions import Self

from petsc4py import PETSc
from slepc4py import SLEPc


class Options:

    def __post_init__(self):
        self._dynamic: dict[str, Any] = {}

    @classmethod
    def default(cls) -> Self:
        try:
            return cls()
        except TypeError:
            raise TypeError('Dataclass fields must all have default values.')

    def __getitem__(self, key):
        try:
            return self._dynamic[key]
        except KeyError:
            return getattr(self, key)

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self._dynamic[key] = value

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        return self._asdict().items()
    
    def keys(self):
        return self._asdict().keys()
    
    def values(self):
        return self._asdict().values()
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def _asdict(self) -> dict:
        if is_dataclass(self):
            return asdict(self) | self._dynamic
        else:
            return self._dynamic
    

@dataclass
class OptionsPETSc(Options):
    ksp_type: str = "gmres"
    pc_type: str = "ilu"
    ksp_rtol: float = 1e-05
    ksp_atol: float = 1e-50
    ksp_divtol: float = 10000
    ksp_max_it: int = 10000


@dataclass
class OptionsSLEPc(Options):
    eps_type: str = 'krylovschur'
    eps_target: float = 0.0
    eps_nev: int = 5
    eps_ncv: int = PETSc.DECIDE
    eps_mpd: int = PETSc.DECIDE
    eps_which: str = 'largest_magnitude'
    eps_tol: float = 1e-7
    eps_max_it: int = 100


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


@dataclass
class OptionsJIT(Options): 
    """See also `dolfinx.jit.DOLFINX_DEFAULT_JIT_OPTIONS`"""
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