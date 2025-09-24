from inspect import signature, Signature, Parameter
import functools
from collections.abc import Iterable
from typing import (
    Callable,
    TypeVar,
    ParamSpec,
    overload,
)
from types import EllipsisType

from ufl.core.expr import Expr
from dolfinx.fem import Function, Constant

from ..utils import MultipleDispatchTypeError, filter_kwargs
from ..fdm.series import ExprSeries, ConstantSeries, FunctionSeries
from ..solver.options import OptionsFFCX, OptionsJIT, OptionsPETSc
from ..solver import (Solver, BoundaryValueProblem, InitialBoundaryValueProblem, InitialValueProblem,
                   ProjectionProblem, InterpolationProblem)
from ..io import create_path, write
from ..utils.deferred import Writer, Stopper
from .utils import signature_name_collision


T = TypeVar('T', int | None, str | None, float | None)
class Simulation:
    def __init__(
        self,
        solvers: Iterable[Solver],
        t: ConstantSeries,
        dt: ConstantSeries | Constant,
        quantities: Iterable[ExprSeries | Function | Constant] = (),
        stoppers: Iterable[Stopper] = (),
        *,
        dir_path: str | None = None,
        store_step: int | float | tuple[int | float | None, int | float | None] | dict[str | Iterable[str], int | float] | EllipsisType | None = ...,
        write_step: int | float | tuple[int | float | None, int | float | None] | dict[str | Iterable[str], int | float] | None = None,
        write_file: str | tuple[str | None, str | None] | dict[str | Iterable[str], str] | None = None,
        parameter_file: str | None = None,
        checkpoint_file: str | None = None,
        texec_file: str | None = None,
    ):
        self.solvers = list(solvers)
        self.t = t 
        self.dt = dt
        self.quantities = list(quantities)
        self.stoppers = list(stoppers)
        self.dir_path = dir_path
        self.parameter_file = parameter_file
        self.checkpoint_file = checkpoint_file
        self.texec_file = texec_file
        self.store_step = store_step
        self.write_step = write_step
        self.write_file = write_file

    @overload
    def __getitem__(
        self, 
        key: str,
    ) -> FunctionSeries | ConstantSeries:
        ...
    
    @overload
    def __getitem__(
        self, 
        key: tuple[str, ...],
    ) -> list[FunctionSeries | ConstantSeries]:
        ...
    
    def __getitem__(
        self, 
        key: str | tuple[str, ...],
    ):
        if isinstance(key, tuple):
            return [self[i] for i in key]
        elif isinstance(key, str):
            try:
                return self.namespace[key]
            except KeyError:
                raise KeyError(f"{key} not found in simulation's namespace")
        else:
            raise TypeError
        
    def __iter__(self):
        for i in (self.solvers, self.t, self.dt, self.quantities):
            yield i

    def index(self, name: str) -> int:
        for i, s in enumerate(self.solvers):
            if s.series.name == name:
                return i
        raise ValueError
    
    def _map_to_solver_series(
        self,
        obj: dict[str | Iterable[str], T] | tuple[T, T] | T,
    ) -> dict[str, T]:
        if obj is None or isinstance(obj, T.__constraints__):
            return {i.series.name: obj for i in self.solvers}
            
        if isinstance(obj, dict):
            d = {}
            for k, v in obj.items():
                if isinstance(k, str):
                    d.update({k: v})
                elif isinstance(Iterable):
                    d.update({i: v for i in k})
                else:
                    raise TypeError
            return d
        
        if isinstance(obj, tuple):
            assert len(obj) == 2
            obj_function, obj_constant = obj
            function_dict = {i.series.name: obj_function for i in self.solvers if isinstance(i.series, FunctionSeries)}
            constant_dict = {i.series.name: obj_constant for i in self.solvers if isinstance(i.series, ConstantSeries)}
            return function_dict | constant_dict
        
        raise MultipleDispatchTypeError(obj)
    
    @property
    def namespace(self) -> dict[str, FunctionSeries | ConstantSeries | ExprSeries | Constant | Function | Expr]:
        d =  {self.t.name: self.t, self.dt.name: self.dt}
        d.update({s.series.name: s.series for s in self.solvers})
        d.update({f.name: f for f in self.quantities})
        return d
    
    @property
    def store_step(self) -> dict[str, int | float | None]:
        return self._store_step
    
    @store_step.setter
    def store_step(self, value):
        if value is Ellipsis:
            self._store_step = {s.series.name: s.series.store for s in self.solvers}
        else:
            self._store_step = self._map_to_solver_series(value)
            for name, store in self._store_step.items():
                    self[name].store = store

    @property
    def write_step(self) -> dict[str, int | float | None]:
        return self._write_step
    
    @write_step.setter
    def write_step(self, value):
        self._write_step = self._map_to_solver_series(value)
    
    @property
    def write_file(self) -> dict[str, str | None]:
        return self._write_file
    
    @write_file.setter
    def write_file(self, value):
        self._write_file = self._map_to_solver_series(value)

    @property
    def writers(self) -> list[Writer]:

        def writer_routine(u: FunctionSeries, file_name: str):
            return lambda t: write(u[0], file_name, self.dir_path, t, 'a', u.mesh.comm)

        _writers = []
        _write_step = {k: v for k, v in self.write_step.items() if v is not None}
        for name, step in _write_step.items():
            file_name = self.write_file.get(name)
            if file_name is None:
                file_name = name
            writer = Writer(writer_routine(self[name], file_name), step, name)
            _writers.append(writer)

        return _writers
    

def configure_simulation(
    petsc: OptionsPETSc | dict | None = None,
    jit: OptionsJIT | dict | None = None,
    ffcx: OptionsFFCX | dict | None = None,
    store_step: int | float | tuple[int | float | None, int | float | None] | dict[str | Iterable[str], int | float] | None = None,
    write_step: int | float | tuple[int | float | None, int | float | None] | dict[str | Iterable[str], int | float] | None = None,
    write_file: str | tuple[str | None, str | None] | dict[str | Iterable[str], str] | None = None,
    parameter_file: str = 'PARAMETERS',
    checkpoint_file: str = 'CHECKPOINT',
    texec_file: str = 'TEXEC',
    dir_base: str = './',
    dir_params: Iterable[str] | str = '',
    dir_label: str | None = None,
    dir_timestamp: bool = False,
):
    if petsc is None:
        petsc = OptionsPETSc.default
    if jit is None:
        jit = OptionsJIT.default
    if ffcx is None:
        ffcx = OptionsFFCX.default
    kwargs_default = locals().copy()
    
    configure_simulation_sig = signature(configure_simulation)
    configure_simulation_params = configure_simulation_sig.parameters
    configure_simulation_params_new = [Parameter(p.name, p.kind, default=kwargs_default[p.name], annotation=p.annotation) for p in configure_simulation_params.values()]
    configure_simulation_sig_new = Signature(configure_simulation_params_new, return_annotation=configure_simulation_sig.return_annotation)

    P = ParamSpec('P')
    def _decorator(
        simulation_func: Callable[
            P, 
            tuple[Iterable[Solver], ConstantSeries, ConstantSeries] 
            | tuple[Iterable[Solver], ConstantSeries, ConstantSeries, Iterable] 
            | Simulation],
    ):
        
        assert not signature_name_collision(simulation_func, configure_simulation)

        # TODO python 3.11+ typing.get_overloads a better solution?
        setattr(configure_simulation, simulation_func.__name__, configure_simulation_sig_new)
        simulation_func_params = signature(simulation_func).parameters

        @overload
        def _(
            *,
            petsc: dict | OptionsPETSc = ...,
            jit: dict | OptionsJIT = ...,
            store_step: int | float | tuple[int | float | None, int | float | None] | dict[str | Iterable[str], int | float] | None = ...,
            write_step: int | float | tuple[int | float | None, int | float | None] | dict[str | Iterable[str], int | float] | None = ...,
            write_file: str | tuple[str | None, str | None] | dict[str | Iterable[str], str] | None = ...,
            parameter_file: str = ...,
            checkpoint_file: str = ...,
            texec_file: str = ...,
            dir_base: str = ...,
            dir_params: Iterable[str] | str = ...,
            dir_label: str | None = ...,
            dir_timestamp: bool = ...,
        ) -> Callable[P, Simulation]:
            ...

        @overload
        def _(*args: P.args, **kwargs: P.kwargs) -> Simulation:
            ...

        @functools.wraps(simulation_func)
        def _(*args, **kwargs):
            if not args and kwargs and all(i in configure_simulation_sig.parameters for i in kwargs):

                kwargs_complete = kwargs_default.copy()
                kwargs_complete.update(kwargs)

                def _inner(*sim_func_args, **sim_func_kwargs):
                    classes = (BoundaryValueProblem, InitialBoundaryValueProblem, InitialValueProblem, 
                               ProjectionProblem, InterpolationProblem)
                    [filter_kwargs(cls.set_defaults)(**kwargs_complete) for cls in classes]
                    simulation_func_return = simulation_func(*sim_func_args, **sim_func_kwargs)
                    [cls.set_defaults() for cls in classes]
    
                    sim_parameters = {
                        k: v.default
                        for k, v in simulation_func_params.items()
                        if v.default is not v.empty
                    }
                    sim_parameters.update({k: v for k, v in zip(simulation_func_params, sim_func_args)})
                    sim_parameters.update(sim_func_kwargs)
                    dir_path = filter_kwargs(create_path)(sim_parameters, **kwargs_complete)
                    simulation = filter_kwargs(Simulation)(*simulation_func_return, dir_path=dir_path, **kwargs_complete)
                    
                    if simulation.writers:
                        write(
                            {'simulation': simulation_func.__name__} | sim_parameters, 
                            simulation.parameter_file, 
                            dir_path,
                            mode='w',
                        )

                    return simulation
                
                return _inner
            else:
                assert all(i in simulation_func_params for i in kwargs)
                return _(petsc=petsc, jit=jit, ffcx=ffcx, store_step=store_step, 
                                write_step=write_step, dir_base=dir_base, dir_params=dir_params,
                                dir_label=dir_label, dir_timestamp=dir_timestamp, 
                        write_file=write_file, parameter_file=parameter_file, checkpoint_file=checkpoint_file, texec_file=texec_file)(*args, **kwargs)
        
        return _
    
    return _decorator


