import argparse 
from collections.abc import Iterable
from types import GenericAlias, UnionType
from typing import Callable, Any
from inspect import signature, Parameter, Signature

from typing import Callable
from collections.abc import Iterable

from ..fem import Constant
from ..fdm import ConstantSeries
from ..solver import Solver, Evaluation, IBVP, IVP
from ..io import write, write_checkpoint, read_checkpoint, reset_directory
from ..utils import log_texec

from ..utils.deferred import Writer, Stopper
from .controller import (
    CreateStopper, CreateWriter, 
    as_stopper, as_writer, has_simulation_arg,
)
from .simulation import configure_simulation, Simulation
from .utils import write_texec, arg_name_collisions, ArgNameCollisionError


def run(
    simulation: Simulation | tuple[Iterable[Solver], ConstantSeries, ConstantSeries | Constant],
    n_stop: int | None = None,
    t_stop: float | None = None,
    dt_init: float | None = None,
    n_init: int | None = None,
    resume: bool = False, 
    overwrite: bool | None = None,
    texec: bool | dict = False,
    stoppers: Iterable[Stopper | Callable[[], bool] | CreateStopper] = (),
    writers: Iterable[Writer | Callable[[], None] | CreateWriter] = (),
) -> None:    
    if isinstance(simulation, tuple):
        simulation = Simulation(*simulation)

    t = simulation.t
    dt = simulation.dt
    
    if (n_stop, t_stop).count(None) == 2:
        raise ValueError('Must provide at least one of `n_stop` and `t_stop`')
    
    _stoppers: list[Stopper] = [as_stopper(s, simulation) for s in stoppers]
    _stoppers.extend(simulation.stoppers)
    _writers: list[Writer] = [as_writer(w, simulation) for w in writers]
    _writers.extend(simulation.writers)

    n_init_min = max(s.n_init for s in simulation.solvers if isinstance(s, (IBVP, IVP)))
    if n_init is not None:
        if n_init < n_init_min:
            raise ValueError("`n_init` must be greater than or equal to the highest finite difference discretization order.")
    else:
        n_init = n_init_min

    if dt_init is not None:
        dt_solver_init = Evaluation(dt, lambda: dt_init)
    else:
        if isinstance(dt, Constant):
            dt_solver_init = Evaluation(dt, lambda: dt.value)
        else:
            dt_solver_init = simulation.solvers[simulation.index(dt.name)]

    solvers_init = [
        s.init if isinstance(s, (IBVP, IVP)) and s.init is not None 
        else dt_solver_init if s.series is dt 
        else s for s in simulation.solvers
    ]

    series_ics = [t, *[s.series for s in simulation.solvers if s.series.ics is not None]]
    if resume:
        n_init = 0
        read_checkpoint(series_ics, simulation.dir_path, simulation.checkpoint_file)
    if overwrite and _writers:
        reset_directory(simulation.dir_path, ('*.h5', '*.xdmf'))
    if _writers and simulation.dir_path:
        parameters = dict(
            n_stop=n_stop, 
            t_stop=t_stop, 
            dt_init=dt_init, 
            n_init=n_init, 
            resume=resume,
        )
        parameters.update(write_delta={k: v for k, v in simulation.write_delta.items() if v is not None})
        write(parameters, simulation.parameter_file, simulation.dir_path, mode='a')

    if isinstance(dt, ConstantSeries):
        _dt = dt[0]
    else:
        _dt = dt

    _n = 0
    _time_stoppers: list[Stopper] = []
    if n_stop is not None:
        _time_stoppers.append(Stopper(lambda: not _n < n_stop))
    if t_stop is not None:
        _time_stoppers.append(Stopper(lambda: not t[0].value < t_stop))

    _texec = {} if texec is True else texec if isinstance(texec, dict) else None
    if isinstance(_texec, dict):
        for s in set((*simulation.solvers, *solvers_init)):
            s.solve = log_texec(s.solve, _texec, f'{s.series.name}_{s.solve.__name__}')
        for w in _writers:
            w.write = log_texec(w.write, _texec, f'{w.name}_{w.write.__name__}')

    while all(not s.stop(t[0]) for s in _time_stoppers):
        if _n < n_init:
            _solvers = solvers_init
        else:
            _solvers = simulation.solvers
        [s.solve() for s in _solvers]
        t.update(t[0].value + _dt.value, future=True)
        [w.write(t[0]) for w in _writers]
        if any(s.stop(t[0]) for s in _stoppers):
            break
        [s.forward(t[0]) for s in simulation.series]
        t.forward(t[0])
        _n += 1

    if _writers:
        write_checkpoint(series_ics, t, simulation.checkpoint_file, simulation.dir_path)
        if _texec:
            write_texec(_texec, simulation.dir_path, simulation.texec_file)


def run_from_cli(
    simulation_func: Callable[..., Simulation] | Callable[..., Callable[..., Simulation]],
    posthook: Callable[[Simulation], None] | None = None,
    stoppers: Iterable[CreateStopper | Callable[..., CreateStopper]] = (),
    writers: Iterable[CreateWriter | Callable[..., CreateWriter]] = (),
    eval_locals: Iterable[object | tuple[str, object]] | None = None,
    description: str = 'Run a simulation from the command line',
) -> None:
    """Note that if single quotations are used to enclose a Python code snippet entered at the
    command line interface, any strings within that code snippet must be enclosed by double quotations, or vice versa.
    
    e.g. `--parameter_name '{1: "one"}'` or `--parameter_name "{1: 'one'}"`
    """

    if eval_locals is None:
        eval_locals = []
    if isinstance(eval_locals, list):
        import lucifex
        eval_locals.extend((
            lucifex.solver.BoundaryConditions,
            lucifex.solver.OptionsFFCX,
            lucifex.solver.OptionsPETSc,
            lucifex.solver.OptionsJIT,
            lucifex.fdm.CN,
            lucifex.fdm.AB1,
            lucifex.fdm.AB2,
            lucifex.fdm.AB,
            lucifex.fdm.AM,
            lucifex.utils.SpatialPerturbation,
            lucifex.utils.DofsPerturbation,
            )
        )

    stoppers_from_sim: list[CreateStopper] =  [s for s in stoppers if has_simulation_arg(s)]
    stoppers_from_cli: list[Callable[..., CreateStopper]] = [s for s in stoppers if not has_simulation_arg(s)]
    writers_from_sim: list[CreateWriter] =  [w for w in writers if has_simulation_arg(w)]
    writers_from_cli: list[Callable[..., CreateWriter]]= [w for w in writers if not has_simulation_arg(w)]

    _collisions = arg_name_collisions(simulation_func, run, *stoppers_from_cli, *writers_from_cli)
    if _collisions:
        raise ArgNameCollisionError(_collisions)

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    if hasattr(configure_simulation, simulation_func.__name__):
        sig_config: Signature = getattr(configure_simulation, simulation_func.__name__)
        params_config = sig_config.parameters
    else:
        params_config = signature(configure_simulation).parameters
    params_simulation = signature(simulation_func).parameters
    params_run = {k: v for k, v in list(signature(run).parameters.items())[1:-2]}
    params_stoppers = [signature(s).parameters for s in stoppers_from_cli]
    params_writers = [signature(w).parameters for w in writers_from_cli]
    if posthook is not None:
        params_hook = {k: v for k, v in list(signature(posthook).parameters.items())[1:]}
    else:
        params_hook = {}

    params_cli: dict[str, Parameter] = {}
    for p in (
        params_config, 
        params_simulation, 
        params_run, 
        *params_stoppers, 
        *params_writers, 
        params_hook,
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
    kwargs_hook = get_subset(params_hook)

    if kwargs_config:
        simulation = simulation_func(**kwargs_config)(**kwargs_simulation)
    else:
        simulation = simulation_func(**kwargs_simulation)

    stoppers: list[CreateStopper] = [s(**k) for s, k in zip(stoppers_from_cli, kwargs_stop)]
    stoppers.extend([s(simulation) for s in stoppers_from_sim])
    writers: list[CreateWriter] = [s(**k) for s, k in zip(writers_from_cli, kwargs_write)]
    writers.extend([w(simulation) for w in writers_from_sim])

    run(simulation, **kwargs_run, stoppers=stoppers, writers=writers)

    if posthook is not None:
        posthook(simulation, **kwargs_hook)


def cli_type_conversion(
    default: Any,
    eval_locals: Iterable[object | tuple[str, object]],
    ):
    eval_locals = [(i.__name__, i) if not isinstance(i, tuple) else i for i in eval_locals]
    def _inner(source: str):
        if source == default:
            return source
        else:
            return eval(source, globals(), dict(eval_locals))
    return _inner


def cli_type_name(
    type_annotation: Any,
) -> str:
    if isinstance(type_annotation, (UnionType, GenericAlias)):
        return str(type_annotation)
    else:
        return type_annotation.__name__
    
    