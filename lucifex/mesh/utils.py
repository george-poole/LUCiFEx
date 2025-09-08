from typing import Callable, ParamSpec, Concatenate, overload

from dolfinx.mesh import Mesh


P = ParamSpec('P')
def overload_mesh(
    mesh_transform: Callable[Concatenate[Mesh, P], None],
):
    @overload
    def _(mesh: Mesh, /, *args: P.args, **kwargs: P.kwargs) -> None:
        ...

    @overload
    def _(copy: bool, mesh: Mesh, /, *args: P.args, **kwargs: P.kwargs) -> Mesh:
        ...

    def _(*args, **kwargs):
        if isinstance(args[0], bool) and isinstance(args[1], Mesh):
            if args[0]:
                mesh = copy_mesh(args[1])
            else:
                mesh = args[1]
            mesh_transform(mesh, *args[2:], **kwargs)
            return mesh
        else:
            return mesh_transform(*args, **kwargs)


def copy_mesh(mesh: Mesh) -> Mesh:
    msh = Mesh(
        mesh.comm,
        mesh.topology,
        mesh.geometry,
        mesh.ufl_domain(),
    )
    msh.name = mesh.name
    return msh
