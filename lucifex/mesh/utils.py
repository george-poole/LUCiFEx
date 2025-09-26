from typing import Callable, ParamSpec, Concatenate, overload

from dolfinx.mesh import Mesh


P = ParamSpec('P')
def overload_mesh(
    mesh_func: Callable[Concatenate[Mesh, P], None],
):
    @overload
    def _(mesh: Mesh, /, *args: P.args, **kwargs: P.kwargs) -> None:
        ...

    @overload
    def _(copy_name: str, mesh: Mesh, /, *args: P.args, **kwargs: P.kwargs) -> Mesh:
        ...

    def _(*args, **kwargs):
        if isinstance(args[0], Mesh):
            return mesh_func(*args, **kwargs)
        elif isinstance(args[0], str) and isinstance(args[1], Mesh):
            mesh = copy_mesh(args[1], args[0])
            mesh_func(mesh, *args[2:], **kwargs)
            return mesh
        else: 
            raise TypeError(
                'Expected positional-only arguments `mesh: Mesh` or copy_name: str, mesh: Mesh',
            )
        
    return _


def copy_mesh(
    mesh: Mesh,
    name: str | None = None,
) -> Mesh:
    msh = Mesh(
        mesh.comm,
        mesh.topology,
        mesh.geometry,
        mesh.ufl_domain(),
    )
    if name is None:
        name = mesh.name
    msh.name = name
    return msh
