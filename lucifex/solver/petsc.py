from typing import TypeAlias, Iterable, Literal
from types import EllipsisType

import functools
import ufl
import numpy as np
from dolfinx.fem import (
    form as dolfinx_form,
    DirichletBCMetaClass,
    FormMetaClass as DolfinxFormMetaClass,
    Function,
    Constant,
    FunctionSpace,
)
from dolfinx.fem.petsc import (
    assemble_matrix as dolfinx_assemble_matrix,
    assemble_vector as dolfinx_assemble_vector,
    apply_lifting,
    set_bc,
)
from dolfinx.cpp.fem.petsc import insert_diagonal
from dolfinx.cpp.la.petsc import create_matrix as cpp_create_matrix
from dolfinx_mpc import (
    assemble_matrix as mpc_assemble_matrix,
    MultiPointConstraint,
    create_sparsity_pattern,
)
from petsc4py import PETSc

PETScMat: TypeAlias = PETSc.Mat
"""Alias to `PETSc.Mat`"""
PETScVec: TypeAlias = PETSc.Vec
"""Alias to `PETSc.Vec`"""


class FormMetaClass(DolfinxFormMetaClass):
    ufl_constants: list[Constant]
    ufl_coefficients: list[Function]


@functools.wraps(dolfinx_form)
def form(*args, **kwargs) -> FormMetaClass:
    f = dolfinx_form(*args, **kwargs)
    ufl_form: ufl.Form = args[0]
    f.ufl_constants = ufl_form.constants()
    f.ufl_coefficients = ufl_form.coefficients()
    return f


ATTR_CONSTANTS = "constants"
ATTR_COEFFICIENTS = "coefficients"
ATTR_ASSEMBLY_COUNT = "assembly_count"


def assemble_matrix(
    m: PETScMat,
    a: FormMetaClass,
    bcs: list[DirichletBCMetaClass] | None = None,
    mpc: MultiPointConstraint | None = None,
    diag: float = 1.0,
    cache: bool | EllipsisType = ...,
) -> None:
    if bcs is None:
        bcs = []

    consts_old = m.getAttr(ATTR_CONSTANTS)
    coeffs_old = m.getAttr(ATTR_COEFFICIENTS)
    consts_new = None
    coeffs_new = None
    if cache is ... or cache is True:
        # TODO better as a dict? would need need all constants and funcs uniquely named
        # consts_new = {i.name: i.value.copy() for i in a.ufl_constants}
        # coeffs_new = {i.name: i.x.array.copy() for i in a.ufl_coefficients}
        consts_new = [i.value.copy() for i in a.ufl_constants]
        coeffs_new = [i.x.array.copy() for i in a.ufl_coefficients]


    if (consts_old is None and coeffs_old is None) or not (
        cache is ...
        and np.isclose(consts_old, consts_new).all()
        # and np.isclose(coeffs_old, coeffs_new).all() # TODO handle vector-valued constants
        and all(np.isclose(i, j).all() for i, j in zip(coeffs_old, coeffs_new, strict=True))
    ):
        # first assembly or assembling again
        _assemble_matrix(m, a, bcs, mpc, diag)
        m.setAttr(ATTR_CONSTANTS, consts_new)
        m.setAttr(ATTR_COEFFICIENTS, coeffs_new)
        return
    else:
        # not assembling again
        return


def _assemble_matrix(
    m: PETScMat,
    a: FormMetaClass,
    bcs: list[DirichletBCMetaClass],
    mpc: MultiPointConstraint | None,
    diag: float,
) -> None:
    # actually assembling the matrix
    if m.isAssembled():
        m.zeroEntries()
    if mpc is None:
        dolfinx_assemble_matrix(m, a, bcs, diag)
    else:
        mpc_assemble_matrix(a, mpc, bcs, diag, A=m)
    m.assemble()
    assembly_count = m.getAttr(ATTR_ASSEMBLY_COUNT)
    if assembly_count is None:
        assembly_count = 0
    assembly_count += 1
    m.setAttr(ATTR_ASSEMBLY_COUNT, assembly_count)


def assemble_vector(
    v: PETScVec,
    l: FormMetaClass,
    bcs_a: tuple[list[DirichletBCMetaClass], FormMetaClass] = None,
) -> None:
    with v.localForm() as local:
        local.set(0)
    dolfinx_assemble_vector(v, l)
    # modifying vector to apply essential boundary conditions
    if bcs_a is not None:
        bcs, a = bcs_a
        apply_lifting(v, [a], [bcs])
        v.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        set_bc(v, bcs)


def sum_matrix(
    m_sum: PETScMat,
    m: list[PETScMat | None],
    scalings: list[float | Constant],
    bcs_fs: tuple[list[DirichletBCMetaClass], FunctionSpace] = None,
) -> None:
    m_sum.zeroEntries()

    if not m_sum.isAssembled():
        m_sum.assemble()

    for mi, si in zip(m, scalings):
        if mi is None:
            continue
        m_sum.axpy(PETSc.ScalarType(si), mi)

    if bcs_fs is not None:
        bcs, (fs_test, fs_trial) = bcs_fs
        ## TODO why do comparison of function spaces??
        if fs_test is fs_trial:
            # modifying matrix diagonal to apply essential boundary conditions
            m_sum.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
            m_sum.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
            diagonal = 1.0
            insert_diagonal(
                m_sum,
                fs_test,
                bcs,
                diagonal,
            )

    m_sum.assemble()


def sum_vector(
    v_sum: PETScVec,
    v: list[PETScVec | None],
    scalings: list[float],
    bcs_fs: tuple[list[DirichletBCMetaClass], FunctionSpace] = None,
) -> None:
    raise NotImplementedError # TODO bcs_fs
    # v_sum.zeroEntries()

    # if not v_sum.isAssembled():
    #     v_sum.assemble()

    # for vi, si in zip(v, scalings):
    #     if vi is None:
    #         continue
    #     v_sum.axpy(PETSc.ScalarType(si), vi)

    # v_sum.assemble()


def array_matrix(
    m: PETScMat,
    indices: tuple[Iterable[int], Iterable[int]] | Literal["dense"] | None = "dense",
    copy: bool = False,
) -> PETScMat | np.ndarray:
    if copy:
        _m = m.copy()
    else:
        _m = m._matrix

    if indices is None:
        return _m
    elif indices == "dense":
        _m.convert("dense")
        return _m.getDenseArray()
    else:
        return _m.getValues(*indices)
    

def array_vector(
    v: PETScVec,
    indices: int | Iterable[int] | Literal["dense"] | None = "dense",
    copy: bool = False,   
) -> PETScVec | np.ndarray:
    if copy:
        _v = v.copy()
    else:
        _v = v

    if indices is None:
        return _v
    elif indices == "dense":
        return _v.getArray()
    else:
        return _v.getValues(indices)


def create_mpc_matrix(a: FormMetaClass, mpc: MultiPointConstraint):
    pattern = create_sparsity_pattern(a, mpc)
    pattern.assemble()
    return cpp_create_matrix(mpc.function_space.mesh.comm, pattern)

