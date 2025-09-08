import os
import sys
import glob
from dataclasses import dataclass
from typing import Any
from collections.abc import Iterable

from natsort import natsorted

from .load import load_txt_dict


@dataclass
class Dataset:
    root: str
    directories: list[str]


def find_dataset(
    data_root: str,
    include: str | Iterable[str] = '*',
    exclude: str | Iterable[str] = (),
) -> Dataset:

    data_directories = set()

    include = [include] if isinstance(include, str) else include
    for pattern in include:
        data_directories.update(glob.glob(f'{data_root}/{pattern}/'))

    exclude = [exclude] if isinstance(exclude, str) else exclude
    for pattern in exclude:
        [data_directories.discard(i) for i in glob.glob(f'{data_root}/{pattern}/')]

    data_directories = natsorted(data_directories)

    return Dataset(data_root, data_directories)


def filter_by_parameters(
    dir_paths: Iterable[str],
    file_name: str,
    parameters: dict[str, Any] | Iterable[dict, str, Any],
    *load_parameter_dict_args,
) -> list[str]:
    if not isinstance(parameters, dict):
        filtered = set()
        for p in parameters:
            filtered.update(filter_by_parameters(dir_paths, file_name, p, *load_parameter_dict_args))
        return natsorted(filtered)

    filtered = []
    for d in dir_paths:
        if not os.path.isdir(d):
            continue
        try:
            p = load_txt_dict(d, file_name, *load_parameter_dict_args)
        except FileNotFoundError:
            continue
        if all(k in p for k in parameters):
            if all(p[k] == v for k, v in parameters.items()):
                filtered.append(d)

    return filtered


def filter_by_tags(
    dir_paths: Iterable[str],
    tags: Iterable[str] = (),  
    parameters: dict[str, Any] | None = None,
    sep: str = ' = ',
) -> list[str]:
    if parameters is None:
        parameters = {}
        
    filtered = []
    for d in dir_paths:
        if not os.path.isdir(d):
            continue
        if all(t in d for t in tags):
            if all(f'{k}{sep}{v}' in d for k, v in parameters.items()):
                filtered.append(d)

    return filtered


FindByError = lambda n: ValueError(f'{n} directories found.')


def find_by_parameters(
    dir_paths: Iterable[str],
    file_name: str,
    parameters: dict[str, Any],
    *load_parameter_dict_args,
) -> str:
    filtered = filter_by_parameters(dir_paths, file_name, parameters, *load_parameter_dict_args)
    n = len(filtered)
    if n == 1:
        return filtered[0]
    else:
        raise FindByError(n)


def find_by_tags(
    dir_paths: Iterable[str],
    tags: Iterable[str] = (),  
    parameters: dict[str, Any] | None = None,
    sep: str = ' = '
) -> str:
    filtered = filter_by_tags(dir_paths, tags, parameters, sep)
    n = len(filtered)
    if n == 1:
        return filtered[0]
    else:
        raise FindByError(n)
    

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
        raise FindByError(n)
        
    return root_dir_path

