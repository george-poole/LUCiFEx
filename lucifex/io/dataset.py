import os
import glob
from typing import Any, overload
from typing_extensions import Unpack
from collections.abc import Iterable

from natsort import natsorted

from ..fdm import FunctionSeries, GridSeries, ConstantSeries, NumericSeries, TriangulationSeries
from .load import load_txt_dict
from .proxy import proxy, Proxy, ObjectName, ObjectType, FileName


class DataSet:
    def __init__(
        self,
        dir_path: str,
        parameter_file: str | None = None,
        *metadata: tuple[ObjectName, ObjectType, FileName, Unpack[tuple]],
        ):
        self._dir_path = dir_path
        self._loaded: dict[
            str, 
            FunctionSeries | ConstantSeries | GridSeries | NumericSeries | TriangulationSeries
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

    @overload
    def __getitem__(
        self, 
        key: str,
    ) -> FunctionSeries |  GridSeries | TriangulationSeries | ConstantSeries | NumericSeries:
        ...

    @overload
    def __getitem__(
        self, 
        key: tuple[str, ...],
    ) -> list[FunctionSeries | GridSeries | TriangulationSeries | ConstantSeries | NumericSeries]:
        ...

    def __getitem__(
        self, 
        key: str | tuple[str, ...],
    ):
        if isinstance(key, tuple) and all(isinstance(i, (str, tuple)) for i in key):
            return [self[i] for i in key]
        else:
            if not isinstance(key, str):
                self._proxies[key[0]] = proxy(*key)
            try:
                return self._loaded[key]
            except KeyError:
                self.load(key)
                return self[key]    
            
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
            
    def __iter__(self):
        return iter(self.keys())

    def items(self):
        return self._loaded.items()
    
    def keys(self):
        return self._loaded.keys()
    
    def values(self):
        return self._loaded.values()
    
    def __repr__(self):
        return f'{self.__class__.__name__}(dir_path={self.dir_path})'
    
    @property
    def dir_path(self):
        return self._dir_path
    
    @property
    def parameter_file(self):
        return self._parameter_file
    

def find_datasets(
    root_dir_path: str,
    *metadata: tuple[ObjectName, ObjectType, FileName, Unpack[tuple]],
    include: str | Iterable[str] = '*',
    exclude: str | Iterable[str] = (),
    parameter_file: str | None = None,
) -> list[DataSet]:

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
            DataSet(
                dir_path,
                parameter_file,
                *metadata,
            )
        )

    return datasets


def filter_by_parameters(
    datasets: Iterable[DataSet],
    parameters: dict[str, Any] | Iterable[dict, str, Any],
    *load_parameter_dict_args,
) -> list[DataSet]:
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
    datasets: Iterable[DataSet],
    substr: Iterable[str | dict[str, Any]] = (),  
    parameters: dict[str, Any] | None = None,
    sep: str = '=',
) -> list[DataSet]:
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
    datasets: Iterable[DataSet],
    parameters: dict[str, Any],
    *load_parameter_dict_args,
) -> DataSet:
    filtered = filter_by_parameters(datasets, parameters, *load_parameter_dict_args)
    n = len(filtered)
    if n == 1:
        return filtered[0]
    else:
        raise FindDirectoryError(n)


def find_by_dirname(
    datasets: Iterable[DataSet],
    substr: Iterable[str] = (),  
    parameters: dict[str, Any] | None = None,
    sep: str = '='
) -> DataSet:
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

