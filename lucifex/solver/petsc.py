from collections.abc import Iterable
from typing import TypeAlias, Iterable, Literal, Any, TypeVar
from typing_extensions import Self
from types import EllipsisType, NoneType
from functools import wraps, reduce

import ufl
import numpy as np
from ufl.core.expr import Expr
from petsc4py import PETSc
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
from ..fem import Constant


PETScMat: TypeAlias = PETSc.Mat
"""Alias to `PETSc.Mat`"""
PETScVec: TypeAlias = PETSc.Vec
"""Alias to `PETSc.Vec`"""


T = TypeVar('T')
Scaled: TypeAlias = tuple[Constant | float, T]

def is_scaled_type(
    obj: Any,
    tp: type | None = None
) -> bool:
    _bool = isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], (float, Constant))
    if tp is None:
        return _bool
    else:
        return _bool and isinstance(obj[1], tp)


class BlockedForm:
    def __init__(
        self, 
        *forms: Iterable[ufl.Form | None] | ufl.Form | None,
        names: Iterable[str] | None = None,
    ):
        if not forms:
            raise ValueError
        
        if all(isinstance(i, (ufl.Form, NoneType)) for i in forms):
            self._forms = list(forms)
            self._matrix_like = True
        else:
            if not all(len(r) == len(forms[0]) for r in forms):
                raise ValueError('Expected all rows to be of same length.')
            if not len(forms) == len(forms[0]):
                raise ValueError('Expected number of rows to match number of columns.')
            self._forms = [list(r) for r in forms]
            self._matrix_like = True

        if names is not None:
            if not len(names) == len(forms[0]):
                raise ValueError('Expected length of names to match number of rows or columns.')
            self._names = tuple(names)

    @property
    def forms(self) -> list[list[ufl.Form | None]] | list[ufl.Form | None]:
        return self._forms
    
    # @property
    # def transpose(self) -> Self | None:
    #     if self.is_matrix_like:
    #         return BlockedForm(
    #             *[list(r) for r in zip(self._forms)]
    #         )
    #     else:
    #         return None
    
    @property
    def is_matrix_like(self) -> bool:
        return self._matrix_like
    
    @property
    def is_vector_like(self) -> bool:
        return not self.is_matrix_like

    def __getitem__(
        self, 
        key: int | tuple[int, int] | str | tuple[str, str],
    ) -> ufl.Form:
        if isinstance(key, tuple):
            if not self.is_matrix_like:
                raise IndexError('`BlockedForm` instance is vector-like, not matrix-like.')
            i, j  = key
            if isinstance(i, int) and isinstance(j, int):
                return self._forms[i][j]
            if isinstance(i, str) and isinstance(j, str):
                assert self._names is not None
                return self[self._names.index(i), self._names.index(i)]
        
        if isinstance(key, int):
            return self._forms[key]
        
        if isinstance(key, str):
            assert self._names is not None
            return self[self._names.index(key)]    
            
        raise TypeError(key)
    
    def __add__(
        self, 
        other: Self | Literal[0],
    ) -> Self:
        if self.is_matrix_like and isinstance(other, BlockedForm) and other.is_matrix_like:
            summed = [
                [add_forms(i, j) for i, j in zip(row, row_other)] for row, row_other in zip(self.forms, other.forms)
            ]
            return BlockedForm(*summed)
        
        if self.is_vector_like and isinstance(other, BlockedForm) and not other.is_vector_like:
            summed = [
                add_forms(i, j) for i, j in zip(self.forms, other.forms)
            ]
            return BlockedForm(*summed)
        
        if isinstance(other, int) and other == 0:
            return self
        
        raise TypeError(other)
    
    def __radd__(self, other: Self) -> Self:
        return self.__add__(other)

    def __mul__(self, other: float | Expr):
        if self.is_matrix_like:
            multiplied = [
                [other * i for i in row] for row in self.forms
            ]
        else:
            multiplied = [
                other * i for i in self.forms
            ]
        return BlockedForm(*multiplied)
    
    def __rmul__(self, other: float | Expr):
        return self.__mul__(other)
    

class MetaForm(FormMetaClass):
    """
    `dolfinx.fem.FormMetaClass` with monkey-patched `ufl_form` attribute.
    """
    ufl_form: ufl.Form


@wraps(dolfinx_form)
def create_metaform(
    *args, 
    **kwargs,
) -> MetaForm | list[MetaForm] | list[list[MetaForm]]:
    _form = kwargs.get('form')
    if _form is None:
        _form = args[0]
    if isinstance(_form, ufl.Form):
        f = dolfinx_form(*args, **kwargs)
        f.ufl_form = _form
        return f
    if isinstance(_form, BlockedForm):
        return create_petsc_matrix(_form.forms)
    else:
        return [create_metaform(i) for i in _form]


ATTR_ASSEMBLY_COUNT = "assembly_count"
ATTR_CONSTANTS = "constants"
ATTR_COEFFICIENTS = "coefficients"


def assemble_petsc_matrix(
    m: PETScMat,
    a: MetaForm | Iterable[MetaForm],
    bcs: list[DirichletBCMetaClass] | None = None,
    mpc: MultiPointConstraint | None = None,
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
    a: MetaForm | list[list[MetaForm]],
    bcs: list[DirichletBCMetaClass],
    mpc: MultiPointConstraint | None,
    diag: float,
) -> None:
    if m.isAssembled():
        m.zeroEntries()
            
    if not isinstance(a, FormMetaClass):
        if mpc is None:
            assemble_matrix_block(
                m,
                a,
                bcs,
                diag,
            )
        else:
            raise ToDoError
    else:
        if mpc is None:
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
    l: MetaForm | list[MetaForm],
    bcs_a: tuple[list[DirichletBCMetaClass], MetaForm] = None,
    mpc: MultiPointConstraint | None = None,
) -> None:
    with v.localForm() as local:
        local.set(0)

    if not isinstance(l, FormMetaClass):
        if bcs_a is None:
            raise TypeError('Cannot be `None` when assembling a blocked vector.')
        bcs, a = bcs
        if mpc is None:
            assemble_vector_block(
                v, l, a, bcs,
            )
            # TODO bcs and lifting?
        else:
            raise ToDoError
    else:
        if mpc is None:
            assemble_vector(v, l)
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

  
def create_petsc_matrix(
   a: MetaForm | list[MetaForm], 
   mpc: MultiPointConstraint | None = None,  
) -> PETScMat:
    if not isinstance(a, FormMetaClass):
        if mpc is None:
            return create_matrix_block(a)
        else:
            raise ToDoError
    else:
        if mpc is None:
            return create_matrix(a)
        else:
            pattern = create_sparsity_pattern(a, mpc)
            pattern.assemble()
            return cpp_create_matrix(mpc.function_space.mesh.comm, pattern)


def create_petsc_vector(
    l: MetaForm | list[MetaForm],
) -> PETScVec:
    if not isinstance(l, FormMetaClass):
        return create_vector_block(l)
    else:
        return create_vector(l)


def sum_petsc_matrix(
    m_sum: PETScMat,
    m: list[PETScMat | None],
    scalings: list[float | Constant],
    bcs_fs: tuple[
        list[DirichletBCMetaClass], tuple[FunctionSpace, FunctionSpace]
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
    v: list[PETScVec | None],
    scalings: list[float],
    bcs_fs: tuple[
        list[DirichletBCMetaClass], tuple[FunctionSpace, FunctionSpace]
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
    

def extract_bilinear_form(
    form: ufl.Form | BlockedForm | None,
    strict: bool = False
) -> ufl.Form | BlockedForm |None:
    if form is None:
        return None
    if isinstance(form, ufl.Form):
        if len(form.arguments()) == 2:
            return ufl.lhs(form)
        if not strict:
            return None
        else:
            raise ValueError
    else:
        mat_forms = [[extract_bilinear_form(f) for f in r] for r in form.forms]
        return BlockedForm(*mat_forms)


def extract_linear_form(
    form: ufl.Form | BlockedForm | None,
    strict: bool = False
) -> ufl.Form | BlockedForm | None:
    if form is None:
        return None
    if isinstance(form, ufl.Form):
        if not ufl.rhs(form).empty():
            return ufl.rhs(form)
        if not strict:
            return None
        else:
            raise ValueError
    else:
        mat_forms = [[extract_linear_form(f) for f in r] for r in form.forms]
        vec_forms = [add_forms(i) for i in mat_forms]
        return BlockedForm(*vec_forms)


def add_forms(
    i: ufl.Form | None, 
    j: ufl.Form | None,
) -> ufl.Form | None:
    if i is None and j is None:
        return None
    if i is None:
        return j
    if j is None:
        return i
    return i + j