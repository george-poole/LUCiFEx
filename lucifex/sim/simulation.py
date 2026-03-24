from inspect import signature, Signature, Parameter
from functools import wraps
from collections.abc import Iterable
from typing import (
    Any,
    Callable,
    ParamSpec,
    TypeAlias,
    TypeVar,
    overload,
    get_args
)
from typing_extensions import Self
from types import EllipsisType

import numpy as np
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh

from ..utils.py_utils import (
    MultiKey, MultipleDispatchTypeError, filter_kwargs, 
    Writer, Stopper,
)
from ..utils.fenicsx_utils import is_mixed_space
from ..fem import Function, Constant
from ..fdm import ExprSeries, ConstantSeries, FunctionSeries, SubFunctionSeries
from ..solver import (
    Solver, BoundaryValueProblem, InitialBoundaryValueProblem, InitialValueProblem, 
    EigenvalueProblem, Interpolation, Projection, OptionsFFCX, OptionsJIT, OptionsPETSc, OptionsSLEPc,
)
from ..io import create_dir_path, write
from ..solver import IBVP, IVP, Evaluation
from .utils import arg_name_collisions, ArgNameCollisionError


DeltaType: TypeAlias = int | float | None
AuxiliaryType: TypeAlias = ExprSeries | Expr | Function | Constant


class Simulation(
    MultiKey[
        str, 
        FunctionSeries | ConstantSeries | ExprSeries | SubFunctionSeries | Expr | Function | Constant | float | int | np.ndarray
        ]
):
    def __init__(
        self,
        solvers: Iterable[Solver] | Solver,
        t: ConstantSeries,
        dt: ConstantSeries | Constant,
        auxiliary: Iterable[AuxiliaryType | tuple[str, AuxiliaryType | float | int | np.ndarray]] = (),
        stoppers: Iterable[Stopper] = (),
        *,
        parameters: dict[str, Any] | None = None,
        store_delta: DeltaType | tuple[DeltaType, DeltaType] | dict[str | Iterable[str], DeltaType] | EllipsisType = Ellipsis,
        write_delta: DeltaType | tuple[DeltaType, DeltaType] | dict[str | Iterable[str], DeltaType] = None,
        write_file: str | tuple[str | None, str | None] | dict[str | Iterable[str], str] | None = None,
        parameter_file: str | None = None,
        checkpoint_file: str | None = None,
        timing_file: str | None = None,
        **create_dir_path_kws,
    ):
        if isinstance(solvers, Solver):
            solvers = [solvers]
        self.solvers = list(solvers)
        self.t = t 
        self.dt = dt
        self._auxiliary = list(auxiliary)
        self.stoppers = list(stoppers)
        self._parameters = parameters
        self.store_delta = store_delta
        self.write_delta = write_delta
        self.write_file = write_file
        self.parameter_file = parameter_file
        self.checkpoint_file = checkpoint_file
        self.timing_file = timing_file
        if create_dir_path_kws:
            self._dir_path = create_dir_path(parameters, **create_dir_path_kws)
        else:
            self._dir_path = None
        self._timings = None
    
    def _getitem(
        self, 
        key: str,
    ):
        return self.namespace[key]
    
    @property
    def solutions(self) -> list[FunctionSeries | ConstantSeries]:
        """
        `len(solutions) ≤ len(solvers)` because a solution may be solved for by more 
        than one solver (e.g. in splitting or linearization schemes)
        """
        _solutions = []

        for s in self.solvers:
            if not s.solution_series in _solutions:
                _solutions.append(s.solution_series)
            if s.correction_series is not None:
                _solutions.append(s.correction_series)

        return _solutions
        
    @property
    def auxiliary(self):
        return self._auxiliary
    
    @property
    def namespace(self) -> dict[str, FunctionSeries | ConstantSeries | ExprSeries | Function | Constant | Expr | Any]:
        d =  {self.t.name: self.t, self.dt.name: self.dt}
        d.update({s.name: s for s in self.solutions})
        d.update({f.name: f for f in self._auxiliary if not isinstance(f, tuple)})
        d.update({f[0]: f[1] for f in self._auxiliary if isinstance(f, tuple)})
        return d
    
    @property
    def meshes(self) -> list[Mesh]:
        return [i.mesh for i in self.solutions]
    
    @property
    def mesh(self) -> Mesh | None:
        if len(set(self.meshes)) == 1:
            return self.meshes[0]
        else:
            return None
        
    @property
    def timings(self) -> dict[str, list[float]]:
        return self._timings
    
    def set_timings(
        self,
        timings: dict[str, list[float]],
        copy: bool = False
    ):
        if copy:
            self._timings = timings.copy()
        else:
            self._timings = timings

    _T = TypeVar('_T')
    def _solution_mapping(
        self,
        delta: _T | tuple[_T, _T] | dict[str | Iterable[str], _T],
        elementary_types: Iterable[_T],
    ) -> dict[str, _T]:
        if isinstance(delta, elementary_types):
            return {s.name: delta for s in self.solutions}
        
        if isinstance(delta, tuple) and len(delta) == 2:
            delta_function, delta_constant = delta
            function_dict = {s.name: delta_function for s in self.solutions if isinstance(s, FunctionSeries)}
            constant_dict = {s.name: delta_constant for s in self.solutions if isinstance(s, ConstantSeries)}
            return function_dict | constant_dict
        
        if isinstance(delta, dict):
            d = {}
            for k, v in delta.items():
                if isinstance(k, str):
                    d.update({k: v})
                elif isinstance(Iterable):
                    d.update({i: v for i in k})
                else:
                    raise TypeError
            return d
        
        raise MultipleDispatchTypeError(delta)

    @property
    def store_delta(self) -> dict[str, int | float | None]:
        return self._store_delta
    
    @store_delta.setter
    def store_delta(self, delta_value):
        if delta_value is Ellipsis:
            self._store_delta = {s.name: s.store for s in self.solutions}
        else:
            self._store_delta = self._solution_mapping(delta_value, get_args(DeltaType))
            for name, store in self._store_delta.items():
                    self[name].store = store

    @property
    def write_delta(self) -> dict[str, int | float | None]:
        return self._write_delta
    
    @write_delta.setter
    def write_delta(self, delta_value):
        self._write_delta = self._solution_mapping(delta_value, get_args(DeltaType))
    
    @property
    def write_file(self) -> dict[str, str | None]:
        return self._write_file
    
    @write_file.setter
    def write_file(self, value):
        self._write_file = self._solution_mapping(value, (str, type(None)))

    @property
    def parameters(self) -> dict[str, Any]:
        if self._parameters is None:
            return {}
        return self._parameters

    @property
    def dir_path(self) -> str | None:
        return self._dir_path

    @property
    def writers(self) -> list[Writer]:

        def _routine(u: FunctionSeries, file_name: str):
            mixed = is_mixed_space(u.function_space, strict=True)
            return lambda t: (
                write(u[0], file_name, self.dir_path, t, 'a', u.mesh.comm, mixed=mixed)
            )

        _writers = []
        _write_delta = {k: v for k, v in self.write_delta.items() if v is not None}
        for name, step in _write_delta.items():
            file_name = self.write_file.get(name)
            if file_name is None:
                file_name = name
            writer = Writer(_routine(self[name], file_name), step, name)
            _writers.append(writer)

        return _writers
    
    def solver_index(
        self, 
        name: str,
        default: int | None = None,
    ) -> int | tuple[int, ...]:
        indices = []
        for i, s in enumerate(self.solvers):
            if s.solution_series.name == name:
                indices.append(i)

        if len(indices) == 0:
            if default is None:
                raise ValueError(f"{name} not found in simulation's solvers.")
            else:
                return default
        elif len(indices) == 1:
            return indices[0]
        else:
            return tuple(indices)
        
    def get_solver(
        self,
        name: str,
    ) -> Solver:
        return self.solvers[self.solver_index(name)]

    @property
    def dt_solver(self) -> Evaluation | None:
        try:
            i = self.solver_index(self.dt.name)
            return self.solvers[i]
        except ValueError:
            return None
        
    @property
    def pre_solvers(self) -> list[Solver]:
        """
        Solvers prior to and including the solver for `dt`, if there is one. \\
        These are solved before the future time `tⁿ⁺¹` is evaluated.
        """
        return self.solvers[:self.solver_index(self.dt.name, -1) + 1]

    @property
    def post_solvers(self) -> list[Solver]:
        """
        Solvers after and not including the solver for `dt`, if there is one. \\
        These are solved after the future time `tⁿ⁺¹ is evaluated.
        """
        return self.solvers[self.solver_index(self.dt.name, -1) + 1:]
    
    def initial(self, dt_init: float | None) -> Self:

        _init = lambda solvers: [
            s.initial if isinstance(s, (IBVP, IVP)) and s.initial is not None 
            else s for s in solvers
        ]
        
        if dt_init is not None:
            dt_solver_init = Evaluation(self.dt, lambda: dt_init)
            if self.dt_solver is None:
                solvers_init = [*_init(self.pre_solvers), dt_solver_init, *_init(self.post_solvers)]
            else:
                solvers_init = [*_init(self.pre_solvers)[:-1], dt_solver_init, *_init(self.post_solvers)]
        else:
            solvers_init = _init(self.solvers)

        return Simulation(
            solvers_init,
            self.t,
            self.dt,
        )
    

def configure_simulation(
    petsc: OptionsPETSc | dict | None = None,
    slepc: OptionsSLEPc | dict | None = None,
    jit: OptionsJIT | dict | EllipsisType | None = None,
    ffcx: OptionsFFCX | dict | None = None,
    store_delta: DeltaType | tuple[DeltaType, DeltaType] | dict[str | Iterable[str], DeltaType] = None,
    write_delta: DeltaType | tuple[DeltaType, DeltaType] | dict[str | Iterable[str], DeltaType] = None,
    write_file: str | tuple[str | None, str | None] | dict[str | Iterable[str], str] | None = None,
    parameter_file: str = 'PARAMETERS',
    checkpoint_file: str = 'CHECKPOINT',
    timing_file: str = 'TIMING',
    dir_root: str = './',
    dir_params: Iterable[str] | str = (),
    dir_prefix: str | None = None,
    dir_suffix: str | None = None,
    dir_datetime: bool | slice = False,
    dir_uid: bool = False,
    dir_seps: tuple[str, str] = ('|', '__', '__'),
):
    if petsc is None:
        petsc = OptionsPETSc.default()
    if slepc is None:
        slepc = OptionsSLEPc.default()
    if jit is None:
        jit = OptionsJIT.default()
    if ffcx is None:
        ffcx = OptionsFFCX.default()
    _KWARGS_DEFAULT = locals().copy()
    
    configure_simulation_sig = signature(configure_simulation)
    configure_simulation_params = configure_simulation_sig.parameters
    configure_simulation_params_new = [
        Parameter(p.name, p.kind, default=_KWARGS_DEFAULT[p.name], annotation=p.annotation) 
        for p in configure_simulation_params.values()
    ]
    configure_simulation_sig_new = Signature(
        configure_simulation_params_new, 
        return_annotation=configure_simulation_sig.return_annotation,
    )

    P = ParamSpec('P')
    def _decorator(
        simulation_factory: Callable[
            P, 
            tuple[Iterable[Solver], ConstantSeries, ConstantSeries] 
            | tuple[Iterable[Solver], ConstantSeries, ConstantSeries, Iterable[ExprSeries | Function | Constant | tuple[str, Expr]]] 
            | Simulation],
    ):
        _collisions = arg_name_collisions(simulation_factory, configure_simulation)
        if _collisions:
            raise ArgNameCollisionError(_collisions)

        # TODO python3.11+ typing.get_overloads a better way?
        setattr(configure_simulation, simulation_factory.__name__, configure_simulation_sig_new)
        simulation_factory_params = signature(simulation_factory).parameters

        @overload
        def _(*args: P.args, **kwargs: P.kwargs) -> Simulation:
            ...

        @overload
        def _(
            *,
            petsc: dict | OptionsPETSc = ...,
            slepc: dict | OptionsSLEPc = ...,
            jit: dict | OptionsJIT = ...,
            store_delta: int | float | tuple[int | float | None, int | float | None] | dict[str | Iterable[str], int | float] | None = ...,
            write_delta: int | float | tuple[int | float | None, int | float | None] | dict[str | Iterable[str], int | float] | None = ...,
            write_file: str | tuple[str | None, str | None] | dict[str | Iterable[str], str] | None = ...,
            parameter_file: str = ...,
            checkpoint_file: str = ...,
            timing_file: str = ...,
            dir_root: str = ...,
            dir_params: Iterable[str] | str = ...,
            dir_prefix: str | None = ...,   
            dir_suffix: str | None = ...,
            dir_datetime: bool | slice = ...,
            dir_uid: bool = ...,
            dir_seps: tuple[str, str] = ...,
        ) -> Callable[P, Simulation]:
            ...

        @wraps(simulation_factory)
        def _(*args, **kwargs):
            if not args and kwargs and all(i in configure_simulation_sig.parameters for i in kwargs):

                kwargs_complete = _KWARGS_DEFAULT.copy()
                kwargs_complete.update(kwargs)

                def _inner(*sim_func_args, **sim_func_kwargs):
                    solver_classes = (
                        BoundaryValueProblem, 
                        InitialBoundaryValueProblem, 
                        InitialValueProblem, 
                        EigenvalueProblem,
                        Projection, 
                        Interpolation,
                    )
                    [filter_kwargs(cls.set_defaults)(**kwargs_complete) for cls in solver_classes]
                    sim_return = simulation_factory(*sim_func_args, **sim_func_kwargs)
                    [cls.set_defaults() for cls in solver_classes]
    
                    sim_parameters = {
                        k: v.default
                        for k, v in simulation_factory_params.items()
                        if v.default is not v.empty
                    }
                    sim_parameters.update({k: v for k, v in zip(simulation_factory_params, sim_func_args)})
                    sim_parameters.update(sim_func_kwargs)
                    if isinstance(sim_return, Simulation):
                        simulation_args = (
                            sim_return.solvers, 
                            sim_return.t, 
                            sim_return.dt,
                            sim_return.auxiliary,
                        )
                    else:
                        simulation_args = sim_return
                    simulation = filter_kwargs(Simulation, include=create_dir_path)(
                        *simulation_args,
                        parameters=sim_parameters,  
                        **kwargs_complete,
                    )
                    return simulation
                
                return _inner
            else:
                unrecognised = [i for i in kwargs if not i in simulation_factory_params]
                if len(unrecognised) == 1:
                    raise TypeError(f'Unrecognised keyword argument: {unrecognised[0]}.')
                if len(unrecognised) > 1:
                    raise TypeError(f'Unrecognised keyword arguments: {tuple(unrecognised)}.')
                return _(
                    petsc=petsc, 
                    slepc=slepc,
                    jit=jit, 
                    ffcx=ffcx, 
                    store_delta=store_delta, 
                    write_delta=write_delta, 
                    dir_root=dir_root, 
                    dir_params=dir_params,
                    dir_prefix=dir_prefix, 
                    dir_suffix=dir_suffix, 
                    dir_datetime=dir_datetime, 
                    dir_uid=dir_uid,
                    dir_seps=dir_seps,
                    write_file=write_file, 
                    parameter_file=parameter_file, 
                    checkpoint_file=checkpoint_file, 
                    timing_file=timing_file,
                )(*args, **kwargs)
        
        return _
    
    return _decorator


