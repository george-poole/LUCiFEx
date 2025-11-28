import os
import glob
from typing import Any
from typing_extensions import Unpack
from collections.abc import Iterable

from natsort import natsorted

from ..fdm import FunctionSeries, GridSeries, ConstantSeries, NumericSeries
from .load import load_txt_dict
from .proxy import proxy, Proxy, ObjectName, ObjectType, FileName


class DataSet:
    def __init__(
        self,
        dir_path: str,
        *,
        functions: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
        constants: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
        grids: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
        numerics: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
        parameter_file: str | None = None,
        ):
        self._dir_path = dir_path
        self._loaded: dict[
            str, 
            FunctionSeries | ConstantSeries | GridSeries | NumericSeries
        ] = {}
        self._proxies: dict[str, Proxy] = {}
        self.parameter_file = parameter_file

        for metadata in (*functions, *constants, *grids, *numerics):
            if metadata in functions:
                _type = FunctionSeries
            elif metadata in constants:
                _type = ConstantSeries
            elif metadata in grids:
                _type = GridSeries
            elif metadata in numerics:
                _type = NumericSeries
            else:
                raise ValueError
            name, file_name, *args = metadata
            self._proxies[name] = proxy((name, _type, file_name, *args))

    def __getitem__(
        self, 
        *names: str | tuple[ObjectName, ObjectType, FileName, Unpack[tuple]],
    ) -> FunctionSeries | ConstantSeries | GridSeries | NumericSeries:
        for name in names:
            if not isinstance(name, str):
                self._proxies[name[0]] = proxy(*name)
            try:
                return self._loaded[name]
            except KeyError:
                obj = self._proxies[name].load_arg(self._dir_path)
                self._loaded[name] = obj
                return self[name]    
            
    def __setitem__(self, name: str, value):
        if name in self._proxies:
            raise ValueError
        else:
            self._loaded[name] = value
            
    def __iter__(self):
        return iter(self.keys())

    def items(self):
        return self._loaded.items()
    
    def keys(self):
        return self._loaded.keys()
    
    def values(self):
        return self._loaded.values()
    
    @property
    def dir_path(self):
        return self._dir_path


def find_datasets(
    root_dir_path: str,
    include: str | Iterable[str] = '*',
    exclude: str | Iterable[str] = (),
    functions: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    constants: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    grids: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
    numerics: Iterable[tuple[ObjectName, FileName, Unpack[tuple]]] = (),
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
                functions=functions,
                constants=constants,
                grids=grids,
                numerics=numerics,
                parameter_file=parameter_file, 
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

