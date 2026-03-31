from typing import Any, Generic, TypeVar, Callable, Hashable
from typing_extensions import Self
from collections.abc import Iterable
from abc import abstractmethod

from dolfinx.mesh import Mesh

from ..utils.py_utils import MultiKey, FrozenDict
from ..fdm import FunctionSeries, ConstantSeries, GridFunctionSeries, TriFunctionSeries, NPyConstantSeries
from ..io.load import (
    load_txt_dict, load_mesh, 
    load_function_series, load_constant_series, load_npy_constant_series,
    load_grid_function_series, load_tri_function_series,
)
from .run import locals_from_lucifex


T = TypeVar('T')
class SimulationFromEXT(
    MultiKey[
        str,
        T | Any,
    ],
    Generic[T],
):
    def __init__(
        self,
        dir_path: str,
        function_series: Iterable[str],
        constant_series: Iterable[str],
        write_file: str | dict[str, str] | None = None,
        parameter_file: str = 'PARAMETERS',
        auxiliary_file: str = 'AUXILIARY',
        run_file: str = 'RUN',
        checkpoint_file: str = 'CHECKPOINT',
        timing_file: str = 'TIMING',
        lazy: bool = True,
        use_cache: bool = True,
        load_args: dict[str, Any | tuple[Any, ...]] | None = None,
    ):
        if write_file is None:
            write_file = {
                **dict(zip(function_series, function_series)),
                **dict(zip(constant_series, constant_series))
            }
        if isinstance(write_file, str):
            write_file = {
                **dict(zip(function_series, [write_file] * len(function_series))),
                **dict(zip(constant_series, [write_file] * len(constant_series)))
            }
    
        self._dir_path = dir_path
        self._function_series = list(function_series)
        self._constant_series = list(constant_series)
        self._write_file = write_file
        self._parameter_file = parameter_file
        self._auxiliary_file = auxiliary_file
        self._run_file = run_file
        self._checkpoint_file = checkpoint_file
        self._timing_file = timing_file
        self._loaded = {}

        _load_args = {k: () for k in write_file}
        if load_args is not None:
            load_args = {k: (v, ) if not isinstance(v, tuple) else v for k, v in load_args.items()}
            _load_args.update(load_args)

        self._load_args = _load_args
        self.use_cache = use_cache

        if not lazy:
            self.load_all()

    @classmethod
    def from_dir_paths(
        cls: type['Self'],
        dir_paths: Iterable[str],
        function_series: Iterable[str],
        constant_series: Iterable[str],
        parameters: dict[str, Any] | None = None,
        write_file: str | dict[str, str] | None = None,
        parameter_file: str = 'PARAMETERS',
        auxiliary_file: str = 'AUXILIARY',
        run_file: str = 'RUN',
        checkpoint_file: str = 'CHECKPOINT',
        timing_file: str = 'TIMING',
        *,
        lazy: bool = True,
        use_cache: bool = True,
        load_args: dict[str, Any | tuple[Any, ...]] | None = None,
    ) -> list[Self]:
        return cls._from_dir_paths(
            dir_paths,
            parameters,
            function_series,
            constant_series,
            write_file=write_file,
            parameter_file=parameter_file,
            auxiliary_file=auxiliary_file,
            run_file=run_file,
            checkpoint_file=checkpoint_file,
            timing_file=timing_file,
            lazy=lazy,
            use_cache=use_cache,
            load_args=load_args,
        )
    
    @classmethod
    def _from_dir_paths(
        cls: type['Self'],
        dir_paths: Iterable[str],
        parameters: dict[str, Any],
        *args,
        **kwargs,
    ) -> list[Self]:
        if parameters is None:
            parameters = {}

        _lazy = kwargs.get('lazy', True)
        _kwargs = kwargs.copy()
        _kwargs.update(lazy=True)

        sims_from_ext = []
        for dp in dir_paths:
            sim = cls(
                dp,
                *args,
                **_kwargs,
            )
            if all(sim[k] == v for k, v in parameters.items()):
                if not _lazy:
                    sim.load_all()
                sims_from_ext.append(sim)

        return sims_from_ext

    @classmethod
    def dict_from_dir_paths(
        cls: type['Self'],
        key: str | tuple[str, ...],
        dir_paths: Iterable[str],
        function_series: Iterable[str],
        constant_series: Iterable[str],
        parameters: dict[str, Any] | None = None,
        write_file: str | dict[str, str] | None = None,
        parameter_file: str = 'PARAMETERS',
        auxiliary_file: str = 'AUXILIARY',
        run_file: str = 'RUN',
        checkpoint_file: str = 'CHECKPOINT',
        timing_file: str = 'TIMING',
        *,
        lazy: bool = True,
        use_cache: bool = True,
        load_args: dict[str, Any | tuple[Any, ...]] | None = None,
        sorting: Callable[[dict], dict] | None = None,
    ) -> dict[Hashable | tuple[Hashable, ...], Self]:

        return cls._dict_from_dir_paths(
            key,
            dir_paths,
            parameters,
            sorting,
            function_series,
            constant_series,
            write_file,
            parameter_file,
            auxiliary_file,
            run_file,
            checkpoint_file,
            timing_file,
            lazy=lazy,
            use_cache=use_cache,
            load_args=load_args,
        )

    @classmethod
    def _dict_from_dir_paths(
        cls: type['Self'],
        key: str | tuple[str, ...],
        dir_paths: Iterable[str],
        parameters: dict[str, Any] | None = None,
        sorting: Callable[[dict], dict] | None = None,
        *args,
        **kwargs,
    ) -> dict[Hashable | tuple[Hashable, ...], Self]:
        if parameters:
            if isinstance(key, str):
                _keys = (key, )
            else:
                _keys = key
            for _k in _keys:
                if _k in parameters.keys():
                    raise ValueError(
                        f"Cannot use values of '{_k}' as a key if it is a fixed parameter."
                    )

        sims_from_ext = cls._from_dir_paths(
            dir_paths, 
            parameters,
            *args,
            **kwargs,
        )

        if sorting is None:
            sorting = lambda d: d

        return sorting({sim[key]: sim for sim in sims_from_ext})

    @abstractmethod
    def _load_function_series(
        self
    ) -> T:
        ...

    @abstractmethod
    def _load_constant_series(
        self,
        name,
    ) -> T:
        ...

    def load(
        self,
        name: str,
        reload: bool = False,
    ) -> None:
        if not reload and name in self._loaded:
            return
        if name in self._function_series:
            self._loaded[name] = self._load_function_series(name)
        elif name in self._constant_series:
            self._loaded[name] = self._load_constant_series(name)
        else:
            for file_name in (
                self._parameter_file,
                self._auxiliary_file,
                self._run_file,
            ):
                ld_dict = load_txt_dict(use_cache=self.use_cache)(
                    self._dir_path, 
                    file_name, 
                    locals_from_lucifex(return_as=FrozenDict),
                )
                if name in ld_dict:
                    self._loaded[name] = ld_dict[name]
                    return
            raise KeyError(name)

    def load_all(
        self,
        reload: bool = False,
    ) -> None:
        [self.load(i, reload) for i in self.write_file]

    def _getitem(
        self, 
        key,
    ):
        try:
            return self._loaded[key]
        except KeyError:
            self.load(key)
            return self._loaded[key]
        
    @property
    def namespace(self) -> dict[str, FunctionSeries | ConstantSeries | Any]:
        return self._loaded

    @property
    def dir_path(self) -> str:
        return self._dir_path

    @property
    def write_file(self):
        return self._write_file
    
    def include_function_series(
        self,
        name: str,
        write_file: str | None = None,
        *load_args: Any,
        lazy: bool = True,
    ) -> None:
        self._include_series(
            name, write_file, self._function_series, *load_args, lazy=lazy,
        )

    def include_constant_series(
        self,
        name: str,
        write_file: str | None = None,
        *load_args: Any,
        lazy: bool = True,
    ) -> None:
        self._include_series(
            name, write_file, self._constant_series, *load_args, lazy=lazy,
        )
    
    def _include_series(
        self,
        name: str,
        write_file: str | None,
        lst: list[str],
        *load_args,
        lazy: bool = True,
    ) -> None:
        if write_file is None:
            write_file = name
        lst.append(write_file)
        self._write_file[name] = write_file
        if load_args:
            self._load_args[name] = load_args
        else:
            self._load_args[name] = ()
        if not lazy:
            self.load(name)


class SimulationFromXDMF(
    SimulationFromEXT[FunctionSeries | ConstantSeries],
):
    def __init__(
        self,
        dir_path: str,
        mesh: Mesh | tuple[str, str],
        function_series: Iterable[str],
        constant_series: Iterable[str],
        write_file: str | dict[str, str] | None = None,
        parameter_file: str = 'PARAMETERS',
        auxiliary_file: str = 'AUXILIARY',
        run_file: str = 'RUN',
        checkpoint_file: str = 'CHECKPOINT',
        timing_file: str = 'TIMING',
        lazy: bool = True,
        use_cache: bool = True,
        load_args: dict[str, Any | tuple[Any, ...]] | None = None,
    ):
        super().__init__(
            dir_path,
            function_series,
            constant_series,
            write_file,
            parameter_file,
            auxiliary_file,
            run_file,
            checkpoint_file,
            timing_file,
            lazy,
            use_cache,
            load_args,
        )
        if not isinstance(mesh, Mesh):
            mesh_name, mesh_file = mesh
            mesh = load_mesh(mesh_name, dir_path, mesh_file)
        self._mesh = mesh

    @classmethod
    def from_dir_paths(
        cls: type['Self'],
        dir_paths: Iterable[str],
        mesh: Mesh | tuple[str, str],
        function_series: Iterable[str],
        constant_series: Iterable[str],
        parameters: dict[str, Any] | None = None,
        write_file: str | dict[str, str] | None = None,
        parameter_file: str = 'PARAMETERS',
        auxiliary_file: str = 'AUXILIARY',
        run_file: str = 'RUN',
        checkpoint_file: str = 'CHECKPOINT',
        timing_file: str = 'TIMING',
        *,
        lazy: bool = True,
        use_cache: bool = True,
        load_args: dict[str, Any | tuple[Any, ...]] | None = None,
    ) -> list[Self]:
        return cls._from_dir_paths(
            dir_paths,
            parameters,
            mesh,
            function_series,
            constant_series,
            write_file=write_file,
            parameter_file=parameter_file,
            auxiliary_file=auxiliary_file,
            run_file=run_file,
            checkpoint_file=checkpoint_file,
            timing_file=timing_file,
            lazy=lazy,
            use_cache=use_cache,
            load_args=load_args,
        )

    @classmethod
    def dict_from_dir_paths(
        cls: type['Self'],
        key: str | tuple[str, ...],
        dir_paths: Iterable[str],
        mesh: Mesh | tuple[str, str],
        function_series: Iterable[str],
        constant_series: Iterable[str],
        parameters: dict[str, Any] | None = None,
        write_file: str | dict[str, str] | None = None,
        parameter_file: str = 'PARAMETERS',
        auxiliary_file: str = 'AUXILIARY',
        run_file: str = 'RUN',
        checkpoint_file: str = 'CHECKPOINT',
        timing_file: str = 'TIMING',
        *,
        lazy: bool = True,
        use_cache: bool = True,
        load_args: dict[str, Any | tuple[Any, ...]] | None = None,
        sorting: Callable[[dict], dict] | None = None,
    ) -> dict[str | tuple[str, ...], Self]:

        return cls._dict_from_dir_paths(
            key,
            dir_paths,
            parameters,
            sorting,
            mesh,
            function_series,
            constant_series,
            write_file,
            parameter_file,
            auxiliary_file,
            run_file,
            checkpoint_file,
            timing_file,
            lazy=lazy,
            use_cache=use_cache,
            load_args=load_args,
        )

    @property
    def mesh(self):
        return self._mesh

    def _load_function_series(
        self,
        name,
    ):
        return load_function_series(use_cache=self.use_cache)(
                name, self._dir_path, self._write_file[name], self._mesh, *self._load_args[name],
        )
    
    def _load_constant_series(
        self,
        name,
    ):
        return load_constant_series(use_cache=self.use_cache)(
                name, self._dir_path, self._write_file[name], self._mesh, *self._load_args[name],
        )


T = TypeVar('T')
class SimulationFromNPZ(
    SimulationFromEXT[T | NPyConstantSeries],
    Generic[T],
):    
    def _load_constant_series(
        self,
        name,
    ):
        return load_npy_constant_series(use_cache=self.use_cache)(
            name, self._dir_path, self._write_file[name], *self._load_args[name]
        )


class GridSimulationFromNPZ(
    SimulationFromNPZ[GridFunctionSeries],
):
    def _load_function_series(
        self,
        name,
    ):
        return load_grid_function_series(use_cache=self.use_cache)(
            name, self._dir_path, self._write_file[name], *self._load_args[name]
        )
    

class TriSimulationFromNPZ(
    SimulationFromNPZ[TriFunctionSeries],
):
    def _load_function_series(
        self,
        name,
    ):
        return load_tri_function_series(use_cache=self.use_cache)(
            name, self._dir_path, self._write_file[name], *self._load_args[name]
        )