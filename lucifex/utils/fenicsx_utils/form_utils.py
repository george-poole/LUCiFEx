from typing import Any, Iterable, TypeVar, TypeAlias, Literal, Callable, ParamSpec
from types import NoneType
from typing_extensions import Self
from functools import reduce

import numpy as np
from dolfinx.mesh import Mesh
from dolfinx.fem import Constant, FunctionSpace
from ufl import Form, rhs, lhs, inner, Argument, Measure
from ufl.core.expr import Expr

from .expr_utils import extract_function_space


T = TypeVar('T')
Scaled: TypeAlias = tuple[Constant | float, T]
"""Type alias for a form with a scale factor"""


def is_scaled_type(
    obj: Any,
    tp: type | None = None
) -> bool:
    _bool = isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], (float, Constant))
    if tp is None:
        return _bool
    else:
        return _bool and isinstance(obj[1], tp)
    
P = ParamSpec('P')
class BlockForm:
    def __init__(
        self, 
        forms: Iterable[Form | None] | Iterable[Iterable[Form | None]],
        names: Iterable[str] | None = None,
    ):
        if not forms:
            raise ValueError
        
        if all(isinstance(i, (Form, NoneType)) for i in forms):
            self._forms = list(forms)
            self._matrix_like = False
        else:
            if not all(len(row) == len(forms[0]) for row in forms):
                raise ValueError('Expected all rows to be of same length.')
            if not len(forms) == len(forms[0]):
                raise ValueError('Expected number of rows to match number of columns.')
            self._forms = [list(row) for row in forms]
            self._matrix_like = True

        if names is not None:
            if not len(names) == len(forms[0]):
                raise ValueError('Expected length of names to match number of rows or columns.')
            self._names = tuple(names)

    @property
    def function_spaces(self) -> list[list[FunctionSpace | None]] |  list[FunctionSpace | None]:
        if self.is_matrix_like:
            return [
                [extract_function_space(f.arguments()[0]) if f is not None else None for f in row] 
                for row in self.forms
            ]
        else:
            return [
                extract_function_space(f.arguments()[0]) if f is not None else None 
                for f in self.forms
            ]

    @property
    def forms(self) -> list[list[Form | None]] | list[Form | None]:
        return self._forms
    
    @property
    def is_matrix_like(self) -> bool:
        return self._matrix_like
    
    @property
    def is_vector_like(self) -> bool:
        return not self.is_matrix_like
        
    def __repr__(self):
        return repr(self.forms)

    def __getitem__(
        self, 
        key: int | tuple[int, int] | str | tuple[str, str],
    ) -> Form | None | list[Form | None]:
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
        if self.is_matrix_like and isinstance(other, BlockForm) and other.is_matrix_like:
            summed = [
                [_add_forms(i, j) for i, j in zip(row, row_other)] 
                for row, row_other in zip(self.forms, other.forms)
            ]
            return BlockForm(summed)
        
        if self.is_vector_like and isinstance(other, BlockForm) and other.is_vector_like:
            summed = [
                _add_forms(i, j) for i, j in zip(self.forms, other.forms)
            ]
            return BlockForm(summed)
        
        if isinstance(other, int) and other == 0:
            return self
        
        raise TypeError(f'Cannot add {type(self)} and {type(other)}.')
    
    def __radd__(self, other: Self) -> Self:
        return self.__add__(other)

    def __mul__(self, other: float | Expr):
        if self.is_matrix_like:
            multiplied = [
                [_mul_form(other, i) for i in row] for row in self.forms
            ]
        else:
            multiplied = [
                _mul_form(other, i) for i in self.forms
            ]
        return BlockForm(multiplied)
    
    def __rmul__(self, other: float | Expr):
        return self.__mul__(other)
    

def _add_forms(
    i: Form | None, 
    j: Form | None,
) -> Form | None:
    if i is None and j is None:
        return None
    if i is None:
        return j
    if j is None:
        return i
    return i + j


def _mul_form(
    scale: float | Expr,
    f: Form | None,
) -> Form | None:
    if f is None:
        return None
    else:
        return scale * f  


def is_none(
    expr: Expr | Form | Any | Iterable[Form | Any] | Iterable[Iterable[Form | Any]] | BlockForm,
    iter_func: Callable[[Iterable[Form | Expr | None]], bool] = all,
    none_aliases: Iterable[Any] = (0, None),
) -> bool:
    if expr in none_aliases:
        return True
    if isinstance(expr, (Expr, Form)):
        return False
    if isinstance(expr, BlockForm):
        return is_none(expr.forms, iter_func, none_aliases)
    else:
        return iter_func(
            [is_none(i, iter_func, none_aliases) for i in expr]
        )
    

def extract_bilinear_form(
    form: Form | BlockForm | None,
    strict: bool = False
) -> Form | BlockForm |None:
    if form is None:
        return None
    if isinstance(form, Form):
        if len(form.arguments()) == 2:
            return lhs(form)
        if not strict:
            return None
        else:
            raise ValueError
    else:
        if form.is_matrix_like:
            mat_forms = [[extract_bilinear_form(f) for f in row] for row in form.forms]
        else:
            diag_forms = [extract_bilinear_form(f) for f in form.forms]
            dim = len(diag_forms)
            mat_forms = [[None] * dim] * dim
            for i in range(dim):
                mat_forms[i][i] = diag_forms[i]
        return BlockForm(mat_forms)


def extract_linear_form(
    form: Form | BlockForm | None,
    strict: bool = False
) -> Form | BlockForm | None:
    if form is None:
        return None
    if isinstance(form, Form):
        if not rhs(form).empty():
            return rhs(form)
        if not strict:
            return None
        else:
            raise ValueError
    else:
        if form.is_matrix_like:
            mat_forms = [[extract_linear_form(f) for f in row] for row in form.forms]
            vec_forms = [reduce(_add_forms, row) for row in mat_forms]
            return BlockForm(vec_forms)
        else:
            vec_forms = [extract_linear_form(f) for f  in form.forms]
            return BlockForm(vec_forms)
    

def create_zero_form(
    v: Argument,
    mesh: Mesh,
    dx: Measure,
) -> Form:
    """
    Linear form `∫ 𝐯·𝟎 dx` created as the inner product of the test function and zero.
    Convenient for adding as  as extra term if the weak formulation `a(u,v) = l(v)` 
    would otherwise lack a linear form `l(v)`.
    """
    return inner(v, Constant(mesh, np.full(v.ufl_shape, 0.0))) * dx


def extract_integrands(
    form: Form,
) -> list[Expr]:
    return [i.integrand() for i in form.integrals()]


def extract_integrand(
    form: Form,
) -> Expr:
    return sum(extract_integrands(form))

