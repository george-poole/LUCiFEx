import argparse 
from collections.abc import Iterable
from types import GenericAlias, UnionType
from typing import Callable, Any, Hashable, TypeAlias, Mapping
import time
from inspect import signature, Parameter, Signature

from typing import Callable
from collections.abc import Iterable

from ..fem import Constant
from ..fdm import ConstantSeries
from ..solver import Solver, IBVP, IVP
from ..io import write, write_checkpoint, read_checkpoint, reset_directory
from ..utils.py_utils import Writer, Stopper, log_timing, LazyEvaluator
from .controllers import (
    StopperFactory, WriterFactory,
    StopperMetaFactory, WriterMetaFactory,
    as_stopper, as_writer, is_controller_factory,
)
from .simulation import configure_simulation, Simulation
from .utils import write_timing, arg_name_collisions, ArgNameCollisionError


T: TypeAlias = ConstantSeries
DT: TypeAlias = ConstantSeries | Constant
def run(
    simulation: Simulation | tuple[Solver | Iterable[Solver], T, DT],
    n_stop: int | None = None,
    t_stop: float | None = None,
    dt_init: float | None = None,
    n_init: int | None = None,
    resume: bool = False, 
    overwrite: bool | None = None,
    timing: bool | dict[Hashable, list[float]] = False,
    prehook: Callable[[Simulation], None] | None = None,
    posthook: Callable[[Simulation], None] | None = None,
    stoppers: Iterable[Stopper | LazyEvaluator[bool] | StopperFactory] = (),
    writers: Iterable[Writer | LazyEvaluator[None] | WriterFactory] = (),
    show_progress: bool = False,
) -> None:    
    if not isinstance(simulation, Simulation):
        simulation = Simulation(*simulation)
    
    if (n_stop, t_stop).count(None) == 2:
        raise ValueError('Must provide at least one of `n_stop` and `t_stop`')
    
    _stoppers: list[Stopper] = [as_stopper(s, simulation) for s in stoppers]
    _stoppers.extend(simulation.stoppers)
    _writers: list[Writer] = [as_writer(w, simulation) for w in writers]
    _writers.extend(simulation.writers)

    n_init_min = max(
        (s.n_init for s in simulation.solvers if isinstance(s, (IBVP, IVP))), 
        default=0,
    )
    if n_init is not None:
        if n_init < n_init_min:
            raise ValueError("`n_init` must be greater than or equal to the highest finite difference discretization order.")
    else:
        n_init = n_init_min

    t = simulation.t
    _dt = simulation.dt if isinstance(simulation.dt, Constant) else simulation.dt[0]
    simulation_init = simulation.initial(dt_init)

    series_checkpointed = [t, *[s.solution_series for s in simulation.solvers if s.solution_series.ics is not None]]
    if resume:
        n_init = 0
        read_checkpoint(series_checkpointed, simulation.dir_path, simulation.checkpoint_file)
    if overwrite and _writers:
        reset_directory(simulation.dir_path, ('*.h5', '*.xdmf'))
    if _writers and simulation.dir_path:
        write(simulation.parameters, simulation.parameter_file, simulation.dir_path, mode='a', file_ext='txt')
        run_parameters = dict(
            n_stop=n_stop, 
            t_stop=t_stop, 
            dt_init=dt_init, 
            n_init=n_init, 
            resume=resume,
            write_delta={k: v for k, v in simulation.write_delta.items() if v is not None},
        )
        write(simulation.auxiliary, simulation.auxiliary_file, simulation.dir_path, mode='a', file_ext='txt')
        write(run_parameters, simulation.run_file, simulation.dir_path, mode='a', file_ext='txt')
        simulation.auxiliary

    _n = 0
    _time_stoppers: list[Stopper] = []
    if n_stop is not None:
        _time_stoppers.append(Stopper(lambda: not _n < n_stop))
    if t_stop is not None:
        _time_stoppers.append(Stopper(lambda: not t[0].value < t_stop))

    _timings = {} if timing is True else timing if isinstance(timing, dict) else None
    if isinstance(_timings, dict):
        for s in set((*simulation.solvers, *simulation_init.solvers)):
            s.solve = log_timing(s.solve, _timings, f'{s.solution_series.name}_{s.solve.__name__}')
        for w in _writers:
            w.write = log_timing(w.write, _timings, f'{w.name}_{w.write.__name__}')
        _timings[run.__name__] = []

    if prehook is not None:
        prehook(simulation)

    prog = None
    if show_progress and n_stop:
        from tqdm.notebook import tqdm
        prog = tqdm(total=n_stop)

    while all(not s.stop(t[0]) for s in _time_stoppers):
        if _n < n_init:
            _simulation = simulation_init
        else:
            _simulation = simulation
        if _timings:
            t_run_start = time.perf_counter()
        [s.solve() for s in _simulation.pre_solvers]
        t.update(t[0].value + _dt.value, future=True)
        [s.solve() for s in _simulation.post_solvers]
        [w.write(t[0]) for w in _writers]
        if any(s.stop(t[0]) for s in _stoppers):
            break
        [s.forward(t[0]) for s in _simulation.solutions]
        t.forward(t[0])
        _n += 1
        if _timings:
            t_run_stop = time.perf_counter()
            _timings[run.__name__].append(t_run_stop - t_run_start)
        if prog is not None:
            prog.update()

    if _timings:
        simulation.set_timings(_timings)

    if _writers:
        write_checkpoint(series_checkpointed, t, simulation.checkpoint_file, simulation.dir_path)
        if _timings:
            write_timing(_timings, simulation.dir_path, simulation.timing_file)

    if posthook is not None:
        posthook(simulation)

from typing import Concatenate, ParamSpec

_ = ParamSpec('_')
def run_from_cli(
    simulation_factory: Callable[..., Simulation] | Callable[..., Callable[..., Simulation]],
    prehook: Callable[[Simulation], None] | Callable[Concatenate[Simulation, _], None] | None = None,
    posthook: Callable[[Simulation], None] | Callable[Concatenate[Simulation, _], None] | None = None,
    stoppers: Iterable[StopperFactory | StopperMetaFactory] = (),
    writers: Iterable[WriterFactory | WriterMetaFactory] = (),
    eval_locals: dict[str, Any] | None = None,
    description: str = 'Run a simulation from the command line',
) -> None:
    """
    Note that if single quotations are used to enclose a Python code snippet entered at the
    command line interface, any strings within that code snippet must be enclosed by 
    double quotations, or vice versa.
    
    e.g. `--parameter_name '{1: "one"}'` or `--parameter_name "{1: 'one'}"`
    """
    
    if eval_locals is None:
        eval_locals = locals_from_lucifex()

    stoppers_from_sim: list[StopperFactory] =  [s for s in stoppers if is_controller_factory(s)]
    stoppers_from_cli: list[Callable[..., StopperFactory]] = [s for s in stoppers if not is_controller_factory(s)]
    writers_from_sim: list[WriterFactory] =  [w for w in writers if is_controller_factory(w)]
    writers_from_cli: list[Callable[..., WriterFactory]]= [w for w in writers if not is_controller_factory(w)]

    _collisions = arg_name_collisions(simulation_factory, run, *stoppers_from_cli, *writers_from_cli)
    if _collisions:
        raise ArgNameCollisionError(_collisions)

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    if hasattr(configure_simulation, simulation_factory.__name__):
        sig_config: Signature = getattr(configure_simulation, simulation_factory.__name__)
        params_config = sig_config.parameters
    else:
        params_config = signature(configure_simulation).parameters
    params_simulation = signature(simulation_factory).parameters
    ARG_INDEX_N_STOP = 1
    ARG_INDEX_TIMING = 7
    ARG_SLC = slice(ARG_INDEX_N_STOP, ARG_INDEX_TIMING + 1)
    params_run = {k: v for k, v in list(signature(run).parameters.items())[ARG_SLC]}
    params_stoppers = [signature(s).parameters for s in stoppers_from_cli]
    params_writers = [signature(w).parameters for w in writers_from_cli]
    if prehook is not None:
        params_prehook = {k: v for k, v in list(signature(prehook).parameters.items())[1:]}
    else:
        params_prehook = {}
    if posthook is not None:
        params_posthook = {k: v for k, v in list(signature(posthook).parameters.items())[1:]}
    else:
        params_posthook = {}

    params_cli: dict[str, Parameter] = {}
    for p in (
        params_config, 
        params_simulation, 
        params_run, 
        *params_stoppers, 
        *params_writers, 
        params_prehook,
        params_posthook,
    ):
        params_cli.update(p)

    for name, prm in params_cli.items():
        annt = prm.annotation
        dflt = prm.default
        if annt is None:
            if dflt is None:
                raise RuntimeError('A parameter without type annotation must have a default.')
            annt = type(dflt)
        parser.add_argument(
            f'--{name}', 
            type=cli_type_conversion(dflt, eval_locals),
            default=dflt, 
            help=cli_type_name(annt),
        )

    kwargs = vars(parser.parse_args())
    get_subset = lambda p: {k: v for k, v in kwargs.items() if k in p}
    kwargs_config = get_subset(params_config)
    kwargs_simulation = get_subset(params_simulation)
    kwargs_run = get_subset(params_run)
    kwargs_stop = [get_subset(i) for i in params_stoppers]
    kwargs_write = [get_subset(i) for i in params_writers]
    kwargs_prehook = get_subset(params_prehook)
    kwargs_posthook = get_subset(params_posthook)

    if kwargs_config:
        simulation = simulation_factory(**kwargs_config)(**kwargs_simulation)
    else:
        simulation = simulation_factory(**kwargs_simulation)

    stoppers: list[StopperFactory] = [s(**k) for s, k in zip(stoppers_from_cli, kwargs_stop)]
    stoppers.extend([s(simulation) for s in stoppers_from_sim])
    writers: list[WriterFactory] = [s(**k) for s, k in zip(writers_from_cli, kwargs_write)]
    writers.extend([w(simulation) for w in writers_from_sim])

    if prehook is not None:
        _prehook = lambda sim: prehook(sim, **kwargs_prehook)
    else:
        _prehook = None

    if posthook is not None:
        _posthook = lambda sim: posthook(sim, **kwargs_posthook)
    else:
        _posthook = None

    run(
        simulation, 
        **kwargs_run, 
        stoppers=stoppers, 
        writers=writers,
        prehook=_prehook,
        posthook=_posthook,
    )

def cli_type_conversion(
    default: Any,
    eval_locals: dict[str, Any],
    ):

    def _inner(source: str):
        if source == default:
            return source
        else:
            return eval(source, globals(), eval_locals)
        
    return _inner


def cli_type_name(
    type_annotation: Any,
) -> str:
    if isinstance(type_annotation, (UnionType, GenericAlias)):
        return str(type_annotation)
    else:
        return type_annotation.__name__
    

def locals_from_lucifex(
    *,
    return_as: Callable[[dict], Mapping] = dict,
    **extra: Any,
) -> Mapping[str, Any]:
    import lucifex
    classes = [
        lucifex.solver.BoundaryConditions,
        lucifex.solver.OptionsFFCX,
        lucifex.solver.OptionsPETSc,
        lucifex.solver.OptionsSLEPc,
        lucifex.solver.OptionsJIT,
        lucifex.fem.SpatialPerturbation,
        lucifex.fem.DofsPerturbation,
        lucifex.utils.fenicsx_utils.BoundaryType,
        lucifex.utils.fenicsx_utils.CellType,
    ]
    fds = [
        lucifex.fdm.DT,
        lucifex.fdm.CN,
        lucifex.fdm.AB1,
        lucifex.fdm.AB2,
        lucifex.fdm.AB,
        lucifex.fdm.AM,
        lucifex.fdm.AM1,
        lucifex.fdm.AM2,
    ]
    return return_as(
        {
            **{cls.__name__: cls for cls in classes},
            **{repr(fd): fd for fd in fds},
            **extra,
        }
    )
    