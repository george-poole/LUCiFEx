import argparse 
from collections.abc import Iterable
from types import GenericAlias, UnionType
from typing import Callable, Any
from inspect import signature, Parameter, Signature

from typing import Callable
from collections.abc import Iterable

from dolfinx.fem import Constant

from ..fdm.series import ConstantSeries
from ..solver import Solver, EvaluationProblem, IBVP, IVP
from ..io.write import write
from ..io.checkpoint import write_checkpoint, read_checkpoint, reset_directory
from ..utils import log_texec

from ..utils.deferred import Writer, Stopper
from .deferred import StopperFromSimulation, WriterFromSimulation
from .simulation import create_simulation, Simulation, is_simulation_callable
from .utils import write_texec, signature_name_collision


def integrate(
    simulation: Simulation | tuple[Iterable[Solver], ConstantSeries, ConstantSeries | Constant],
    n_stop: int | None = None,
    t_stop: float | None = None,
    dt_init: float | None = None,
    n_init: int | None = None,
    resume: bool = False, 
    overwrite: bool | None = None,
    texec: bool | dict = False,
    stoppers: Iterable[Stopper | Callable[[], bool] | StopperFromSimulation] = (),
    writers: Iterable[Writer | Callable[[], None] | WriterFromSimulation] = (),
) -> None:
    """
    Mutates `simulation`
    """
    
    if isinstance(simulation, tuple):
        simulation = Simulation(*simulation)

    t = simulation.t
    dt = simulation.dt
    
    if (n_stop, t_stop).count(None) == 2:
        raise ValueError('Must provide at least one of `n_stop` and `t_stop`')
    
    _stoppers: list[Stopper] = [s if isinstance(s, Stopper) else s(simulation) if is_simulation_callable(s) else Stopper(s) for s in stoppers]
    _stoppers.extend(simulation.stoppers)
    _writers: list[Writer] = [w if isinstance(w, Writer) else w(simulation) if is_simulation_callable(w) else Writer(w) for w in writers]
    _writers.extend(simulation.writers)

    n_init_solvers = max(i.n_init for i in simulation.solvers if isinstance(i, (IBVP, IVP)))
    if n_init is not None:
        if not n_init > n_init_solvers:
            raise ValueError("Overridden `n_init` value must be greater than simulation's finite difference discretization order")
    else:
        n_init = n_init_solvers

    if dt_init is not None:
        dt_solver_init = EvaluationProblem(dt, lambda: dt_init)
    else:
        if isinstance(dt, Constant):
            dt_solver_init = EvaluationProblem(dt, lambda: dt.value)
        else:
            dt_solver_init = simulation.solvers[simulation.index(dt.name)]

    solvers_init = [
        i.init if isinstance(i, (IBVP, IVP)) and i.init is not None else dt_solver_init if i.series is dt else i
        for i in simulation.solvers
    ]

    series_ics = [t, *[i.series for i in simulation.solvers if i.series.ics is not None]]
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
        parameters.update(write_step={k: v for k, v in simulation.write_step.items() if v is not None})
        write(parameters, simulation.parameter_file, simulation.dir_path, mode='a')

    if isinstance(dt, ConstantSeries):
        _dt = dt[0]
    else:
        _dt = dt

    _n = 0
    if n_stop is not None:
        _stoppers.append(Stopper(lambda: not _n < n_stop))
    if t_stop is not None:
        _stoppers.append(Stopper(lambda: not t[0].value < t_stop))

    _texec = {} if texec is True else texec if isinstance(texec, dict) else None
    if isinstance(_texec, dict):
        for s in set((*simulation.solvers, *solvers_init)):
            s.solve = log_texec(s.solve, _texec, f'{s.series.name}_{s.solve.__name__}')
        for w in _writers:
            w.write = log_texec(w.write, _texec, f'{w.name}_{w.write.__name__}')

    while all(not s.stop(t[0]) for s in _stoppers):
        if _n < n_init:
            _solvers = solvers_init
        else:
            _solvers = simulation.solvers
        [s.solve() for s in _solvers]
        t.update(t[0].value + _dt.value, future=True)
        [w.write(t[0]) for w in _writers]
        [s.forward(t[0]) for s in _solvers]
        t.forward(t[0])
        _n += 1

    if _writers:
        write_checkpoint(series_ics, t, simulation.checkpoint_file, simulation.dir_path)
        if _texec:
            write_texec(_texec, simulation.dir_path, simulation.texec_file)


def integrate_from_cli(
    simulation_func: Callable[..., Simulation] | Callable[..., Callable[..., Simulation]],
    description: str = 'Run a simulation from the command line',
    stoppers: Iterable[StopperFromSimulation | Callable[..., StopperFromSimulation]] = (),
    writers: Iterable[WriterFromSimulation | Callable[..., WriterFromSimulation]] = (),
    eval_locals: Iterable[object | tuple[str, object]] | None = None,
) -> Simulation:
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
            lucifex.utils.Perturbation,
            )
        )

    stoppers_from_sim: list[StopperFromSimulation] =  [s for s in stoppers if is_simulation_callable(s)]
    stoppers_from_cli: list[Callable[..., StopperFromSimulation]] = [s for s in stoppers if not is_simulation_callable(s)]
    writers_from_sim: list[WriterFromSimulation] =  [w for w in writers if is_simulation_callable(w)]
    writers_from_cli: list[Callable[..., WriterFromSimulation]]= [w for w in writers if not is_simulation_callable(w)]

    if signature_name_collision(simulation_func, integrate, *stoppers_from_cli, *writers_from_cli):
        raise TypeError('Parameter names must be unique to each callable.')

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    if hasattr(create_simulation, simulation_func.__name__):
        sig_simulation: Signature = getattr(create_simulation, simulation_func.__name__)
        params_simulation = sig_simulation.parameters
    else:
        params_simulation = signature(create_simulation).parameters
    params_solvers = signature(simulation_func).parameters
    params_integrate = {k: v for k, v in list(signature(integrate).parameters.items())[1:-2]}
    params_stoppers = [signature(s).parameters for s in stoppers_from_cli]
    params_writers = [signature(w).parameters for w in writers_from_cli]

    params_cli: dict[str, Parameter] = {}
    for p in (params_simulation, params_solvers, params_integrate, *params_stoppers, *params_writers):
        params_cli.update(p)

    for name, prm in params_cli.items():
        annt = prm.annotation
        dflt = prm.default
        if annt is None:
            if dflt is None:
                raise RuntimeError('A parameter without type annotation must have a default.')
            annt = type(dflt)
        parser.add_argument(f'--{name}', type=cli_type_conversion(dflt, eval_locals), default=dflt, help=cli_type_name(annt))

    kwargs = vars(parser.parse_args())
    get_subset = lambda p: {k: v for k, v in kwargs.items() if k in p}
    simulation_kwargs = get_subset(params_simulation)
    solvers_kwargs = get_subset(params_solvers)
    integrate_kwargs = get_subset(params_integrate)
    stop_kwargs = [get_subset(i) for i in params_stoppers]
    write_kwargs = [get_subset(i) for i in params_writers]

    if simulation_kwargs:
        simulation = simulation_func(**simulation_kwargs)(**solvers_kwargs)
    else:
        simulation = simulation_func(**solvers_kwargs)

    stoppers: list[StopperFromSimulation] = [s(**k) for s, k in zip(stoppers_from_cli, stop_kwargs)]
    stoppers.extend([s(simulation) for s in stoppers_from_sim])
    writers: list[WriterFromSimulation] = [s(**k) for s, k in zip(writers_from_cli, write_kwargs)]
    writers.extend([w(simulation) for w in writers_from_sim])

    integrate(simulation, **integrate_kwargs, stoppers=stoppers, writers=writers)

    return simulation


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
    
    