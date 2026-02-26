import os
import glob
from typing import Any, Generic, TypeVar
from typing_extensions import Unpack
from collections.abc import Iterable
from abc import abstractmethod

from natsort import natsorted
from dolfinx.mesh import Mesh

from ..utils.py_utils import MultiKey
from ..fdm import FunctionSeries, ConstantSeries, GridFunctionSeries, TriFunctionSeries, NumericSeries
from ..io.load import (
    load_txt_dict, load_mesh, 
    load_function_series, load_constant_series, load_numeric_series,
    load_grid_function_series, load_tri_function_series,
)
from ..io._proxy import proxy, Proxy, ObjectName, ObjectType, FileName


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
        mesh: Mesh | tuple[str, str],
        function_series: Iterable[str],
        constant_series: Iterable[str],
        write_file: str | dict[str, str] | None = None,
        parameter_file: str = 'PARAMETERS',
        checkpoint_file: str = 'CHECKPOINT',
        timing_file: str = 'TIMING',
        lazy: bool = True,
        load_args: dict[str, tuple] | None = None,
        use_cache: bool = True
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

        if not isinstance(mesh, Mesh):
            mesh_name, mesh_file = mesh
            mesh = load_mesh(mesh_name, dir_path, mesh_file)
        
        self._dir_path = dir_path
        self._function_series = list(function_series)
        self._constant_series = list(constant_series)
        self._write_file = write_file
        self._parameter_file = parameter_file
        self._checkpoint_file = checkpoint_file
        self._timing_file = timing_file
        
        self._mesh = mesh
        self._loaded = {}

        _load_args = {k: () for k in write_file}
        if load_args is not None:
            _load_args.update(load_args)

        self._load_args = _load_args
        self.use_cache = use_cache

        if not lazy:
            [self._load(i) for i in write_file]

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

    def _load(
        self,
        name: str,
        reload: bool = False,
    ) -> None:
        
        if not reload and name in self._loaded:
            return

        if name in self._function_series:
            ld = self._load_function_series(name)
        elif name in self._constant_series:
            ld = self._load_constant_series(name)
        else:
            ld = load_txt_dict(self._dir_path, self._parameter_file) #FIXME eval_locals
            raise ValueError(name)

        self._loaded[name] = ld

    def _getitem(
        self, 
        key,
    ):
        try:
            return self._loaded[key]
        except KeyError:
            self._load(key)
            return self._loaded[key]
        
    @property
    def namespace(self) -> dict[str, FunctionSeries | ConstantSeries | Any]:
        return self._loaded
    
    @property
    def write_file(self):
        return self._write_file
    
    def include_function_series(
        self,
        name: str,
        write_file: str | None,
        *load_args,
    ) -> None:
        self._include_series(
            name, write_file, self._function_series, *load_args,
        )

    def include_constant_series(
        self,
        name: str,
        write_file: str | None,
        *load_args,
    ) -> None:
        self._include_series(
            name, write_file, self._constant_series, *load_args,
        )
    
    def _include_series(
        self,
        name: str,
        write_file: str | None,
        lst: list[str],
        *load_args,
    ) -> None:
        if write_file is None:
            write_file = name
        lst.append(write_file)
        self._write_file[name] = write_file
        if load_args:
            self._load_args[name] = load_args


class SimulationFromXDMF(
    SimulationFromEXT[FunctionSeries | ConstantSeries],
):
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
    SimulationFromEXT[T | NumericSeries],
    Generic[T],
):    
    def _load_constant_series(
        self,
        name,
    ):
        return load_numeric_series(use_cache=self.use_cache)(
            name, self._dir_path, self._write_file[name],
        )


class GridSimulationFromNPZ(
    SimulationFromNPZ[GridFunctionSeries],
):
    def _load_function_series(
        self,
        name,
    ):
        return load_grid_function_series(use_cache=self.use_cache)(
            name, self._dir_path, self._write_file[name],
        )
    

class TriSimulationFromNPZ(
    SimulationFromNPZ[TriFunctionSeries],
):
    def _load_function_series(
        self,
        name,
    ):
        return load_tri_function_series(use_cache=self.use_cache)(
            name, self._dir_path, self._write_file[name],
        )













class SimulationData(
    MultiKey[
        str,
        FunctionSeries |  GridFunctionSeries | TriFunctionSeries | ConstantSeries | NumericSeries
    ]
):
    def __init__(
        self,
        dir_path: str,
        parameter_file: str | None = None,
        *metadata: tuple[ObjectName, ObjectType, FileName, Unpack[tuple]],
        # lazy: bool = True
        ):
        self._dir_path = dir_path
        self._loaded: dict[
            str, 
            FunctionSeries | ConstantSeries | GridFunctionSeries | NumericSeries | TriFunctionSeries
        ] = {}
        self._proxies: dict[str, Proxy] = {}
        self._parameter_file = parameter_file
        self.add_metadata(
            *metadata,
        )

    def add_metadata(
        self,
        *metadata,
    ) -> None:
        for md in metadata:
            name, _type, file_name, *args = md
            self._proxies[name] = proxy((name, _type, file_name, *args))

    def load(
        self, 
        name: str, 
        reload: bool = False,
    ) -> None:
        if not reload and name in self._loaded:
            return
        try:
            obj = self._proxies[name].load_arg(self._dir_path)
        except KeyError:
            raise KeyError(f"No metadata has been included for '{name}'")
        self._loaded[name] = obj

    def _getitem(
        self, 
        key: str | tuple[str, ...],
    ):
        # FIXME
        # if not isinstance(key, str):
        #     self._proxies[key[0]] = proxy(*key)
        try:
            return self._loaded[key]
        except KeyError:
            self.load(key)
            return self._loaded[key]
            
    def __setitem__(
        self, 
        name: str | tuple[str, ...], 
        value: Any | tuple[Any, ...],
    ):
        if isinstance(name, str):
            if name in self._proxies:
                raise ValueError(f"'{name}' already exists.")
            else:
                self._loaded[name] = value
        else:
            for n, v in zip(name, value):
                self[n] = v
                
    def __repr__(self):
        return f'{self.__class__.__name__}(dir_path={self.dir_path})'
    
    @property
    def dir_path(self):
        return self._dir_path
    
    @property
    def parameter_file(self):
        return self._parameter_file
    
    def __iter__(self):
        return iter(self.keys())

    def items(self):
        return self._loaded.items()
    
    def keys(self):
        return self._loaded.keys()
    
    def values(self):
        return self._loaded.values()


def find_simulations(
    root_dir_path: str,
    *metadata: tuple[ObjectName, ObjectType, FileName, Unpack[tuple]],
    include: str | Iterable[str] = '*',
    exclude: str | Iterable[str] = (),
    parameter_file: str | None = None,
) -> list[SimulationData]:

    dir_paths = set()

    include = [include] if isinstance(include, str) else include
    for pattern in include:
        dir_paths.update(glob.glob(f'{root_dir_path}/{pattern}/'))

    exclude = [exclude] if isinstance(exclude, str) else exclude
    for pattern in exclude:
        [dir_paths.discard(i) for i in glob.glob(f'{root_dir_path}/{pattern}/')]

    dir_paths = natsorted(dir_paths)

    datasets = []
    for dir_path in dir_paths:
        datasets.append(
            SimulationData(
                dir_path,
                parameter_file,
                *metadata,
            )
        )

    return datasets


def filter_by_parameters(
    datasets: Iterable[SimulationData],
    parameters: dict[str, Any] | Iterable[dict, str, Any],
    *load_parameter_dict_args,
) -> list[SimulationData]:
    if not isinstance(parameters, dict):
        filtered = set()
        for p in parameters:
            filtered.update(filter_by_parameters(datasets, file_name, p, *load_parameter_dict_args))
        return natsorted(filtered)

    filtered = []
    for d in datasets:
        file_name = d.parameter_file
        dir_path = d.dir_path
        if not os.path.isdir(dir_path):
            continue
        try:
            p = load_txt_dict(dir_path, file_name, *load_parameter_dict_args)
        except FileNotFoundError:
            continue
        if all(k in p for k in parameters):
            if all(p[k] == v for k, v in parameters.items()):
                filtered.append(d)

    return filtered


def filter_by_dirname(
    datasets: Iterable[SimulationData],
    substr: Iterable[str | dict[str, Any]] = (),  
    parameters: dict[str, Any] | None = None,
    sep: str = '=',
) -> list[SimulationData]:
    if parameters is None:
        parameters = {}

    all_substr = [*substr, *[f'{k}{sep}{v}' for k, v in parameters.items()]]
        
    filtered = []
    for d in datasets:
        dir_path = d.dir_path
        if not os.path.isdir(dir_path):
            continue
        if all(s in dir_path for s in all_substr):
            filtered.append(d)

    return filtered


class FindDirectoryError(ValueError):
    def __init__(self, n: int):
        super().__init__(f'{n} directories found.')


def find_by_parameters(
    datasets: Iterable[SimulationData],
    parameters: dict[str, Any],
    *load_parameter_dict_args,
) -> SimulationData:
    filtered = filter_by_parameters(datasets, parameters, *load_parameter_dict_args)
    n = len(filtered)
    if n == 1:
        return filtered[0]
    else:
        raise FindDirectoryError(n)


def find_by_dirname(
    datasets: Iterable[SimulationData],
    substr: Iterable[str] = (),  
    parameters: dict[str, Any] | None = None,
    sep: str = '='
) -> SimulationData:
    filtered = filter_by_dirname(datasets, substr, parameters, sep)
    n = len(filtered)
    if n == 1:
        return filtered[0]
    else:
        raise FindDirectoryError(n)
    

def find_by_id(
    root_dir_path: str ,
    dir_id: str,
    root_search: bool = False,
    recursive_search: bool = False,
) -> str:
    globbed = set(glob.glob(f'{root_dir_path}/*{dir_id}*'))
    if root_search:
        globbed.update(glob.glob(f'{root_dir_path}*{dir_id}*'))
    if recursive_search:
        globbed.update(glob.glob(f'{root_dir_path}/**/*{dir_id}*', recursive=True))

    n = len(globbed)
    if n == 1:
        root_dir_path = list(globbed)[0]
    else:
        raise FindDirectoryError(n)
        
    return root_dir_path

