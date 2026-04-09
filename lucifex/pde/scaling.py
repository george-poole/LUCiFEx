from typing import Callable, ParamSpec, Generic, TypeVar, overload

from dolfinx.mesh import Mesh
from lucifex.fem import Constant
from lucifex.utils.py_utils import MultiKey
from lucifex.utils.npy_utils import AnyNumber


T = TypeVar('T') # AnyNumber | Constant
class ScalingMap(
    MultiKey[str, T],
    Generic[T],
):
    
    @overload
    def __init__(
        self: 'ScalingMap[AnyNumber]', 
        d: dict[str, AnyNumber],
        name: str | None = None,
    ):
        ...

    @overload  
    def __init__(
        self: 'ScalingMap[Constant]', 
        d: dict[str, AnyNumber],
        name: str | None = None,
        *,
        mesh: Mesh,
    ):
        ...

    def __init__(
        self, 
        d: dict[str, AnyNumber],
        name: str | None = None,
        *,
        mesh: Mesh | None = None,
    ):
        self._d = d
        self._name = name
        self._mesh = mesh

    @property
    def map(self) -> dict[str, AnyNumber]:
        return self._d
    
    @property
    def name(self) -> str | None:
        return self._name
    
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.map)}, '{self.name}')"
    
    def __call__(self, mesh: Mesh) -> 'ScalingMap[Constant]':
        return ScalingMap(self._d, self._name, mesh=mesh)

    def _getitem(
        self, 
        arg: str,
    ):
        if self._mesh is None:
            return self._d[arg]
        else:
            return Constant(self._mesh, self._d[arg], arg)
        

P = ParamSpec('P')
class ScalingOptions(Generic[P]):
    def __init__(
        self,
        keys: tuple[str, ...],
        options_factory: Callable[P, dict[str, tuple[float, ...]]],
    ):
        self._keys = keys
        self._options_factory = options_factory

    def __getitem__(
        self, 
        option: str,
    ) -> Callable[P, ScalingMap[AnyNumber]]:
        def _(*args: P.args, **kwargs: P.kwargs) -> ScalingMap[AnyNumber]:
            values = self._options_factory(*args, **kwargs)[option]
            return ScalingMap(dict(zip(self._keys, values)), option)
        return _
    