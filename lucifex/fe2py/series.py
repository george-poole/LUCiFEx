from abc import ABC, abstractmethod
from functools import cached_property
from typing import TypeVar, Iterable, Generic, Callable, overload
from typing_extensions import Self
import operator

import numpy as np
from matplotlib.tri.triangulation import Triangulation
from dolfinx.mesh import Mesh

from ..fdm import ConstantSeries, FunctionSeries, Series, SubSeriesError
from ..utils.fenicsx_utils import ScalarVectorError, get_component_functions, is_cartesian, is_simplicial, NonCartesianQuadMeshError

from .grid import GridSeries
from .tri import TriSeries


