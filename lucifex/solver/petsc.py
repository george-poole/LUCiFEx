import os
from collections.abc import Iterable
from typing import TypeAlias, Literal
from types import EllipsisType

import numpy as np
from petsc4py import PETSc
from ufl import Form
from dolfinx.fem import (
    form as dolfinx_form,
    DirichletBCMetaClass,
    FormMetaClass,
    FunctionSpace,
)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    create_matrix,
    create_vector,
    create_matrix_block,
    create_vector_block,
    assemble_matrix_block,
    assemble_vector_block,
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
from ..utils.fenicsx_utils import BlockForm
from ..fem import Constant


PETScMat: TypeAlias = PETSc.Mat
"""Alias to `PETSc.Mat`"""
PETScVec: TypeAlias = PETSc.Vec
"""Alias to `PETSc.Vec`"""


class MetaForm(FormMetaClass):
    """
    `dolfinx.fem.FormMetaClass` with monkey-patched `ufl_form` attribute.
    """
    ufl_form: Form


def create_metaform(
    form: Form | BlockForm | None
    | Iterable[Form | None] | Iterable[Iterable[Form | None]], 
    dtype: np.dtype = PETSc.ScalarType,
    ffcx_options: dict | None = None, 
    jit_options: dict | None = None,
) -> MetaForm | list[MetaForm] | list[list[MetaForm]]:
    
    if form is None:
        return None

    if ffcx_options is None:
        ffcx_options = {}
    if jit_options is None:
        jit_options = {}

    if isinstance(form, Form):
        metaform = dolfinx_form(form, dtype, ffcx_options, jit_options)
        metaform.ufl_form = form
        return metaform
    elif isinstance(form, BlockForm):
        return create_metaform(form.forms)
    else:
        return [create_metaform(i) for i in form]


ATTR_ASSEMBLY_COUNT = "assembly_count"
ATTR_CONSTANTS = "constants"
ATTR_COEFFICIENTS = "coefficients"


def assemble_petsc_matrix(
    m: PETScMat,
    a: MetaForm | Iterable[MetaForm],
    bcs: Iterable[DirichletBCMetaClass] | None = None,
    mpc: MultiPointConstraint | Iterable[MultiPointConstraint] | None = None,
    diag: float = 1.0,
    cache: bool | EllipsisType = False,
) -> None:
    if bcs is None:
        bcs = []

    if cache is False:
        return _assemble_petsc_matrix(m, a, bcs, mpc, diag)
    
    if cache is True:
        assembly_count: int | None = m.getAttr(ATTR_ASSEMBLY_COUNT)
        if not assembly_count:
            return _assemble_petsc_matrix(m, a, bcs, mpc, diag)
        else:
            return
        
    if cache is Ellipsis:
        _consts: list | None = m.getAttr(ATTR_CONSTANTS)
        _coeffs: list | None = m.getAttr(ATTR_COEFFICIENTS)
        consts = [i.value.copy() for i in a.ufl_form.constants()]
        coeffs = [i.x.array.copy() for i in a.ufl_form.coefficients()]
        if (np.isclose(_consts, consts).all() 
            and all(np.isclose(i, j).all() for i, j in zip(_coeffs, coeffs, strict=True))
        ):
            _assemble_petsc_matrix(m, a, bcs, mpc, diag)
            m.setAttr(ATTR_CONSTANTS, consts)
            m.setAttr(ATTR_COEFFICIENTS, coeffs)
            return 

    raise TypeError(cache)


def _assemble_petsc_matrix(
    m: PETScMat,
    a: MetaForm | Iterable[Iterable[MetaForm]],
    bcs: Iterable[DirichletBCMetaClass],
    mpc: MultiPointConstraint | Iterable[MultiPointConstraint] | None,
    diag: float,
) -> None:
    if m.isAssembled():
        m.zeroEntries()
            
    if not isinstance(a, FormMetaClass):
        if not mpc:
            assemble_matrix_block(
                m,
                a,
                bcs,
                diag,
            )
        else:
            raise ToDoError
    else:
        if not mpc:
            assemble_matrix(m, a, bcs, diag)
        else:
            mpc_assemble_matrix(a, mpc, bcs, diag, m)

    m.assemble()
    assembly_count = m.getAttr(ATTR_ASSEMBLY_COUNT)
    if assembly_count is None:
        assembly_count = 0
    assembly_count += 1
    m.setAttr(ATTR_ASSEMBLY_COUNT, assembly_count)


def assemble_petsc_vector(
    v: PETScVec,
    l: MetaForm | Iterable[MetaForm],
    bcs_a: tuple[Iterable[DirichletBCMetaClass], MetaForm] = None,
    mpc: MultiPointConstraint | None = None,
) -> None:
    with v.localForm() as local:
        local.set(0)

    if not isinstance(l, FormMetaClass):
        if bcs_a is None:
            raise TypeError('Cannot be `None` when assembling a blocked vector.')
        bcs, a = bcs_a
        if not mpc:
            assemble_vector_block(
                v, l, a, bcs,
            )
            # TODO bcs and lifting?
        else:
            raise ToDoError
    else:
        if not mpc:
            assemble_vector(v, l)
        else:
            mpc_assemble_vector(l, mpc, v)
        
        if bcs_a is not None:
            bcs, a = bcs_a
            if not mpc:
                apply_lifting(v, [a], [bcs])
            else:
                mpc_apply_lifting(v, [a], [bcs], mpc)
            v.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
            set_bc(v, bcs)

  
def create_petsc_matrix(
   a: MetaForm | Iterable[Iterable[MetaForm]], 
   mpc: MultiPointConstraint | Iterable[MultiPointConstraint] |  None = None,  
) -> PETScMat:
    if not isinstance(a, FormMetaClass):
        if not mpc:
            return create_matrix_block(a)
        else:
            raise ToDoError
    else:
        if not mpc:
            return create_matrix(a)
        else:
            pattern = create_sparsity_pattern(a, mpc)
            pattern.assemble()
            return cpp_create_matrix(mpc.function_space.mesh.comm, pattern)


def create_petsc_vector(
    l: MetaForm | Iterable[MetaForm],
) -> PETScVec:
    if not isinstance(l, FormMetaClass):
        return create_vector_block(l)
    else:
        return create_vector(l)


def sum_petsc_matrix(
    m_sum: PETScMat,
    m: Iterable[PETScMat | None],
    scalings: Iterable[float | Constant],
    bcs_fs: tuple[
        Iterable[DirichletBCMetaClass], tuple[FunctionSpace, FunctionSpace]
    ] = None,
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


def sum_petsc_vector(
    v_sum: PETScVec,
    v: Iterable[PETScVec | None],
    scalings: Iterable[float],
    bcs_fs: tuple[
        Iterable[DirichletBCMetaClass], tuple[FunctionSpace, FunctionSpace]
    ] = None,
) -> None:
    v_sum.zeroEntries()

    if not v_sum.isAssembled():
        v_sum.assemble()

    for vi, si in zip(v, scalings):
        if vi is None:
            continue
        v_sum.axpy(PETSc.ScalarType(si), vi)

    if bcs_fs is not None:
        # bcs, (fs_test, fs_trial) = bcs_fs
        # TODO manipulations?
        raise ToDoError

    v_sum.assemble()


def view_petsc_matrix(
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
    

def view_petsc_vector(
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
    

def set_petsc_threads(
    omp_num_threads: int | None = None,
    openblas_num_threads: int | None = None,
    mkl_num_threads: int | None = None,
) -> None:
    if omp_num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
    if openblas_num_threads is not None:
        os.environ["OPENBLAS_NUM_THREADS"] = str(openblas_num_threads)
    if mkl_num_threads is not None:
        os.environ["MKL_NUM_THREADS"] = str(mkl_num_threads)
