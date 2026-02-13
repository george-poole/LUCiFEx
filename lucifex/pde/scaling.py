from typing import overload, Callable, ParamSpec, Generic
from typing_extensions import Unpack

from dolfinx.mesh import Mesh
from lucifex.fem import Constant
from lucifex.utils.py_utils import MultiKey


class ScalingMap(
    MultiKey[
        str | tuple[Mesh, str] | tuple[Mesh, Unpack[tuple[str, ...]]], 
        int | float | Constant
    ]
):

    def __init__(
        self, 
        d: dict[str, float | int],
        name: str | None = None
    ):
        self._d = d
        self._name = name

    @property
    def map(self) -> dict[str, float | int]:
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
class ScalingChoice(Generic[P]):
    def __init__(
        self,
        names: tuple[str, ...],
        choices_factory: Callable[P, dict[str, tuple[float, ...]]],
    ):
        self._symbols = names
        self._choices_factory = choices_factory

    def __getitem__(
        self, 
        option: str,
    ) -> Callable[P, ScalingMap]:
        def _(*args: P.args, **kwargs: P.kwargs) -> ScalingMap:
            values = self._choices_factory(*args, **kwargs)[option]
            return ScalingMap(dict(zip(self._symbols, values)), option)
        return _
    