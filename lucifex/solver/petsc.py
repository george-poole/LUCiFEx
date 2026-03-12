from typing import TypeAlias, Iterable, Literal, Any
from types import EllipsisType
from functools import wraps

import ufl
import numpy as np
from petsc4py import PETSc
from dolfinx.fem import (
    form,
    DirichletBCMetaClass,
    FormMetaClass,
    FunctionSpace,
)
from dolfinx.fem.petsc import (
    assemble_matrix as dolfinx_assemble_matrix,
    assemble_vector as dolfinx_assemble_vector,
    create_matrix as dolfinx_create_matrix,
    # TODO
    # assemble_matrix_block,
    # assemble_matrix_nest,
    apply_lifting,
    set_bc,
)
from dolfinx.cpp.fem.petsc import insert_diagonal
from dolfinx.cpp.la.petsc import create_matrix as cpp_create_matrix
from dolfinx_mpc import (
    MultiPointConstraint,
    assemble_matrix as mpc_assemble_matrix,
    assemble_vector as mpc_assemble_vector,
    apply_lifting as mpc_apply_lifting,
    create_sparsity_pattern,
)

from ..utils.py_utils import ToDoError
from ..fem import Constant


PETScMat: TypeAlias = PETSc.Mat
"""Alias to `PETSc.Mat`"""
PETScVec: TypeAlias = PETSc.Vec
"""Alias to `PETSc.Vec`"""


class MetaForm(FormMetaClass):
    ufl_form: ufl.Form


@wraps(form)
def meta_form(*args, **kwargs) -> MetaForm:
    f = form(*args, **kwargs)
    ufl_form = kwargs.get('form')
    if ufl_form is None:
        ufl_form = args[0]
    f.ufl_form = ufl_form
    return f


ATTR_CONSTANTS = "constants"
ATTR_COEFFICIENTS = "coefficients"
ATTR_ASSEMBLY_COUNT = "assembly_count"


def assemble_matrix(
    m: PETScMat,
    a: MetaForm,
    bcs: list[DirichletBCMetaClass] | None = None,
    mpc: MultiPointConstraint | None = None,
    diag: float = 1.0,
    cache: bool | EllipsisType = Ellipsis,
) -> None:
    if bcs is None:
        bcs = []

    _consts: list | None = m.getAttr(ATTR_CONSTANTS)
    _coeffs: list | None = m.getAttr(ATTR_COEFFICIENTS)
    consts = None
    coeffs = None
    if cache is Ellipsis:
        consts = [i.value.copy() for i in a.ufl_form.constants()]
        coeffs = [i.x.array.copy() for i in a.ufl_form.coefficients()]
    if cache is True:
        consts = []
        coeffs = []

    if (cache is False 
        or (cache is True and (_consts is None and _coeffs is None))
        or (cache is Ellipsis and np.isclose(_consts, consts).all() and all(np.isclose(i, j).all() for i, j in zip(_coeffs, coeffs, strict=True)))
    ):
        _assemble_matrix(m, a, bcs, mpc, diag)
        m.setAttr(ATTR_CONSTANTS, consts)
        m.setAttr(ATTR_COEFFICIENTS, coeffs)
        return
    else:
        return


def _assemble_matrix(
    m: PETScMat,
    a: MetaForm,
    bcs: list[DirichletBCMetaClass],
    mpc: MultiPointConstraint | None,
    diag: float,
) -> None:
    if m.isAssembled():
        m.zeroEntries()
    if mpc is None:
        dolfinx_assemble_matrix(m, a, bcs, diag)
    else:
        mpc_assemble_matrix(a, mpc, bcs, diag, m)
    m.assemble()
    assembly_count = m.getAttr(ATTR_ASSEMBLY_COUNT)
    if assembly_count is None:
        assembly_count = 0
    assembly_count += 1
    m.setAttr(ATTR_ASSEMBLY_COUNT, assembly_count)


def assemble_vector(
    v: PETScVec,
    l: MetaForm,
    bcs_a: tuple[list[DirichletBCMetaClass], MetaForm] = None,
    mpc: MultiPointConstraint | None = None,
) -> None:
    with v.localForm() as local:
        local.set(0)
    if mpc is None:
        dolfinx_assemble_vector(v, l)
    else:
        mpc_assemble_vector(l, mpc, v)
    
    if bcs_a is not None:
        bcs, a = bcs_a
        if mpc is None:
            apply_lifting(v, [a], [bcs])
        else:
            mpc_apply_lifting(v, [a], [bcs], mpc)
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
        if fs_test is fs_trial:
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
    raise ToDoError
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
        _m = m

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
    

def create_matrix(
   a: MetaForm, 
   mpc: MultiPointConstraint | None = None,  
) -> PETScMat:
    if mpc is None:
        return dolfinx_create_matrix(a)
    else:
        pattern = create_sparsity_pattern(a, mpc)
        pattern.assemble()
        return cpp_create_matrix(mpc.function_space.mesh.comm, pattern)


class BlockedForm:
    def __init__(
        self, 
        *rows: Iterable[ufl.Form | None],
        names: Iterable[str] | None,
    ):
        self._rows = [list(r) for r in rows]
        self._names = tuple(names)

    def __getitem__(
        self, 
        key: tuple[int, int] | tuple[str, str],
    ) -> ufl.Form:
        i, j  = key

        if isinstance(i, int) and isinstance(j, int):
            return self._rows[i][j]
        
        if isinstance(i, int) and isinstance(j, str):
            assert self._names is not None
            i = self._names.index(i)
            _ = self._names.index(j)
            return self[i, j]
        
        raise TypeError(key)
    
    def __add__(self, other):
        pass # TODO
    

ScaledForm: TypeAlias = tuple[Constant | float, ufl.Form | BlockedForm]


def is_scaled_form(obj: Any) -> bool:
    return isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], (float, Constant))


def bilinear_form(
    f: ufl.Form | BlockedForm,
):
    if isinstance(f, ufl.Form):
        return ufl.lhs(f)
    else:
        ...


def linear_form(
    f: ufl.Form | BlockedForm,
):
    if isinstance(f, ufl.Form):
        return ufl.rhs(f)
    else:
        ...