from typing import Callable, ParamSpec, Generic
from typing_extensions import Unpack

from dolfinx.mesh import Mesh
from lucifex.fem import Constant
from lucifex.utils.py_utils import MultiKey
from lucifex.utils.npy_utils import AnyNumber


# FIXME overloaded __getitem__ type hints
class ScalingMap(
    MultiKey[
        str | tuple[Mesh, str] | tuple[Mesh, Unpack[tuple[str, ...]]], 
        AnyNumber | Constant
    ]
):

    def __init__(
        self, 
        d: dict[str, AnyNumber],
        name: str | None = None
    ):
        self._d = d
        self._name = name

    @property
    def map(self) -> dict[str, AnyNumber]:
        return self._d
    
    @property
    def name(self) -> str | None:
        return self._name
    
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.map)}, '{self.name}')"

    def _getitem(
        self, 
        arg: str | tuple[Mesh, str] | tuple[Mesh, Unpack[tuple[str, ...]]],
    ):
        if isinstance(arg, str):
            return self._d[arg]
        
        match arg:
            case (mesh, name) if isinstance(mesh, Mesh):             
                return Constant(mesh, self._getitem(name), name)
            case (mesh, *names) if isinstance(mesh, Mesh):
                return tuple(self._getitem((mesh, n)) for n in names)
            case _:
                raise TypeError
        

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
    ) -> Callable[P, ScalingMap]:
        def _(*args: P.args, **kwargs: P.kwargs) -> ScalingMap:
            values = self._options_factory(*args, **kwargs)[option]
            return ScalingMap(dict(zip(self._keys, values)), option)
        return _
    