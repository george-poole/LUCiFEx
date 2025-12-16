from typing import (
    Literal,
    Callable,
    ParamSpec,
)
from collections.abc import Iterable
from types import EllipsisType
from typing_extensions import Self
from functools import partial

import numpy as np
from ufl import Form, lhs, rhs, Measure, TestFunction, TrialFunction
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh
from dolfinx.fem import FunctionSpace #Constant, Function
from dolfinx.fem.petsc import create_vector, DirichletBCMetaClass
from dolfinx_mpc import MultiPointConstraint
from petsc4py import PETSc
from slepc4py import SLEPc

from ..utils import (
    replicate_callable, 
    create_fem_space, SpatialMarkerAlias
)
from ..fdm import FiniteDifference, FiniteDifferenceArgwise, FunctionSeries, finite_difference_order
from ..fdm.ufl_operators import inner
from ..fem import Function, Constant, Perturbation
from .bcs import BoundaryConditions, Value, SubspaceIndex
from .options import (
    OptionsPETSc, OptionsSLEPc,
    OptionsFFCX, OptionsJIT, set_from_options,
)
from .petsc import (
    meta_form,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    sum_matrix,
    sum_vector,
    array_matrix,
    array_vector,
    PETScMat,
    PETScVec,
)
from .eval import GenericSolver


P = ParamSpec("P")
class BoundaryValueProblem(GenericSolver[Function, FunctionSeries]):

    petsc_default = OptionsPETSc.default()
    jit_default = OptionsJIT.default()
    ffcx_default = OptionsFFCX.default()

    @classmethod
    def set_defaults(
        cls,
        petsc=None,
        jit=None,
        ffcx=None,
    ):
        if petsc is None:
            petsc = OptionsPETSc.default()
        if jit is None:
            jit = OptionsJIT.default()
        if ffcx is None:
            ffcx = OptionsFFCX.default()
        cls.petsc_default = petsc
        cls.jit_default = jit
        cls.ffcx_default = ffcx

    def __init__(
        self,
        solution: Function | FunctionSeries,
        forms: Form | Iterable[Form | tuple[Constant | float, Form]],
        bcs: BoundaryConditions 
        | list[DirichletBCMetaClass] 
        | tuple[list[DirichletBCMetaClass], MultiPointConstraint] 
        | None = None,
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] = False,
        assemble_termwise: tuple[bool, bool] = (False, False),
        future: bool = False,
        overwrite: bool = False,
    ) -> None:
        
        super().__init__(solution, corrector, future, overwrite)
        
        if isinstance(forms, Form):
            forms = (forms, )

        if petsc is None:
            petsc = self.petsc_default
        if jit is None:
            jit = self.jit_default
        if ffcx is None:
            ffcx = self.ffcx_default
        petsc = dict(petsc)
        jit = dict(jit)
        ffcx = dict(ffcx)

        if bcs is None:
            bcs = []
        if isinstance(bcs, list):
            self._bcs = bcs
            self._mpc = None
        elif isinstance(bcs, tuple):
            self._bcs, self._mpc = bcs
        else:
            forms_weak_bcs = bcs.create_weak_bcs(solution.function_space)
            forms = (*forms, *forms_weak_bcs)
            self._bcs = bcs.create_strong_bcs(solution.function_space)        
            self._mpc = bcs.create_periodic_bcs(solution.function_space)

        if self._mpc is not None and not self._mpc.finalized:
            self._mpc.finalize()
        
        # variational forms
        self._scalings = [i[0] if isinstance(i, tuple) else 1.0 for i in forms]
        self._forms = [i[1] if isinstance(i, tuple) else i for i in forms]
        # bilinear forms
        self._a_forms = [lhs(i) if len(i.arguments()) == 2 else None for i in self._forms]
        self._a_form_termwise: Form = sum([i for i in self._a_forms if i is not None])
        self._a_form_nontermwise = sum(
            [i * j for i, j in zip(self._scalings, self._a_forms, strict=True) if j is not None]
        )
        if self._a_form_termwise == 0 or self._a_form_nontermwise == 0:
            raise RuntimeError(f'{self.__class__.__name__} requires must have a bilinear form `a(u,v)` with test and trial functions `v, u`.')
        # linear forms
        self._l_forms = [rhs(i) if not rhs(i).empty() else None for i in self._forms]
        self._l_form_termwise: Form = sum([i for i in self._l_forms if i is not None])
        self._l_form_nontermwise = sum(
            [i * j for i, j in zip(self._scalings, self._l_forms, strict=True) if j is not None]
        )
        if self._l_form_termwise == 0 or self._l_form_nontermwise == 0:
            raise RuntimeError(f'{self.__class__.__name__} requires must have a linear form `l(v)` with test function `v`.')
        # setters
        self._create_form = partial(
            meta_form,
            jit_options=jit,
            form_compiler_options=ffcx,
        )
        self._create_matrix = partial(create_matrix, mpc=self._mpc)
        prefix = f"{self.__class__.__name__}_{id(self)}"
        self._set_solver = lambda solver: (solver.setOptionsPrefix(prefix), set_from_options(solver, petsc))
        self._set_matvec = lambda matvec: (matvec.setOptionsPrefix(prefix), matvec.setFromOptions())
        # initialization status
        self._init_matrix_termwise = False
        self._init_matrix_nontermwise = False
        self._init_vector_termwise = False
        self._init_vector_nontermwise = False
        # termwise matrix attributes
        self._a_metaforms = None
        self._a_metaform_termwise = None
        self._matrices = None
        self._matrix_termwise = None
        self._solver_termwise = None
        # nontermwise matrix attributes
        self._a_metaform_nontermwise = None
        self._matrix_nontermwise = None
        self._solver_nontermwise = None
        # termwise vector attributes
        self._l_metaforms = None
        self._l_metaform_termwise = None
        self._vectors = None
        self._vector_termwise = None
        # nontermwise vector attributes
        self._l_metaform_nontermwise = None
        self._vector_nontermwise = None
        # effective attributes
        self._matrix = None
        self._vector = None
        self._solver = None
        # defaults
        self._assemble_termwise = assemble_termwise
        self._cache_matrix = cache_matrix

    def _create_matrix_termwise(self) -> None:
        if self._init_matrix_termwise:
            return
        self._a_metaforms = [
            self._create_form(ai) if ai is not None else None for ai in self._a_forms]
        self._a_metaform_termwise = self._create_form(self._a_form_termwise)
        self._matrices = [
            self._create_matrix(ai) if ai is not None else None
            for ai in self._a_metaforms
        ]
        self._matrix_termwise = self._create_matrix(self._a_metaform_termwise)
        self._solver_termwise = PETSc.KSP().create(self.solution.function_space.mesh.comm)
        self._solver_termwise.setOperators(self._matrix_termwise)
        self._set_solver(self._solver_termwise)
        for mat in (*self._matrices, self._matrix_termwise):
            if mat is not None:
                self._set_matvec(mat)
        self._init_matrix_termwise = True

    def _create_matrix_nontermwise(self) -> None:
        if self._init_matrix_nontermwise:
            return
        self._a_metaform_nontermwise = self._create_form(self._a_form_nontermwise)
        self._matrix_nontermwise = self._create_matrix(self._a_metaform_nontermwise)
        self._solver_nontermwise = PETSc.KSP().create(self.solution.function_space.mesh.comm)
        self._solver_nontermwise.setOperators(self._matrix_nontermwise)
        self._set_solver(self._solver_nontermwise)
        self._set_matvec(self._matrix_nontermwise)
        self._init_matrix_nontermwise = True

    def _create_vector_termwise(self):
        if self._init_vector_termwise:
            return
        self._l_metaforms = [self._create_form(i) if i is not None else None for i in self._l_forms]
        self._l_metaform_termwise = self._create_form(self._l_form_termwise)
        self._vectors = [
            create_vector(i) if i is not None else None
            for i in self._l_metaforms
        ]
        self._vector_termwise = create_vector(self._l_metaform_termwise)
        for vec in (*self._vectors, self._vector_termwise):
            if vec is not None:
                self._set_matvec(vec)
        self._init_vector_termwise = True

    def _create_vector_nontermwise(self):
        if self._init_vector_nontermwise:
            return
        self._l_metaform_nontermwise = self._create_form(self._l_form_nontermwise)
        self._vector_nontermwise = create_vector(self._l_metaform_nontermwise)
        self._set_matvec(self._vector_nontermwise)    
        self._init_vector_nontermwise = True

    @classmethod
    def from_forms_func(
        cls,
        forms_func: Callable[P, Iterable[Form | tuple[Constant | float, Form]] | Form],
        bcs: BoundaryConditions 
        | list[DirichletBCMetaClass] 
        | tuple[list[DirichletBCMetaClass], MultiPointConstraint] 
        | None = None,
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] = False,
        assemble_termwise: tuple[bool, bool] = (False, False),
        future: bool = False,
        overwrite: bool = False,
        solution: Function | FunctionSeries | None = None, 
    ):
        """
        If the `solution` argument is not provided, it will be inferred
        as the zeroth argument to `forms_func`.
        """
        def _create(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Self:
            nonlocal solution
            if solution is None:
                if args:
                    solution = args[0]
                else:
                    solution = list(kwargs.values())[0]
            return cls(
                solution, 
                forms_func(*args, **kwargs), 
                bcs, 
                petsc, 
                jit, 
                ffcx, 
                corrector,
                cache_matrix, 
                assemble_termwise,
                future,
                overwrite,
            )
        return _create

    def solve(
        self,
        future: bool | None = None,
        overwrite: bool | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] | None = None,
        assemble_termwise: tuple[bool, bool] | None = None,
    ) -> None:
        """Mutates the data structures `self.solution` and `self.series` containing the solution"""
        if cache_matrix is None:
            cache_matrix = self._cache_matrix
        if assemble_termwise is None:
            assemble_termwise = self._assemble_termwise

        #  matrix_termwise, vector_termwise = assemble_termwise
        matrix_termwise, vector_termwise = assemble_termwise

        if matrix_termwise:
            self._create_matrix_termwise()
            if not isinstance(cache_matrix, Iterable):
                cache_matrix = [cache_matrix] * len(self._matrices)
            [
                assemble_matrix(m, a, self._bcs, self._mpc, cache=c)
                for m, a, c in zip(self._matrices, self._a_metaforms, cache_matrix, strict=True)
                if (m, a) != (None, None)
            ]
            sum_matrix(
                self._matrix_termwise,
                self._matrices,
                self._scalings,
                (self._bcs, self._a_metaform_termwise.function_spaces),
            )
            self._matrix = self._matrix_termwise
            self._solver = self._solver_termwise
        else:
            self._create_matrix_nontermwise()
            assemble_matrix(
                self._matrix_nontermwise,
                self._a_metaform_nontermwise,
                self._bcs,
                self._mpc,
                cache=cache_matrix,
            )
            self._matrix = self._matrix_nontermwise
            self._solver = self._solver_nontermwise

        if vector_termwise:
            self._create_vector_termwise()
            [
                assemble_vector(v, l, (self._bcs, a), self._mpc)
                for v, l, a in zip(self._vectors, self._l_metaforms, self._a_metaforms, strict=True)
                if (v, l, a) != (None, None, None)
            ]
            sum_vector(
                self._vector_termwise, self._vectors, self._scalings, (self._bcs, self._a_metaform_termwise)
            )
            self._vector = self._vector_termwise
        else:
            self._create_vector_nontermwise()
            assemble_vector(
                self._vector_nontermwise,
                self._l_metaform_nontermwise,
                (self._bcs, self._a_metaform_nontermwise),
                self._mpc,
            )
            self._vector = self._vector_nontermwise

        self._solver.solve(self._vector, self._solution.vector)
        self._solution.x.scatter_forward()
        if self._mpc:
            self._mpc.backsubstitution(self._solution.vector)

        super().solve(future, overwrite)

    def get_matrix(
        self,
        indices: tuple[Iterable[int], Iterable[int]] | Literal["dense"] | None = None,
        copy: bool = False,
    ) -> PETScMat | np.ndarray | None:
        if self._matrix is None:
            return None
        return array_matrix(self._matrix, indices, copy)

    def get_vector(
        self,
        indices: int | Iterable[int] | Literal["dense"] | None = None,
        copy: bool = False,
    ) -> PETScVec | np.ndarray | None:
        if self._vector is None:
            return None
        return array_vector(self._vector, indices, copy)
    
    def get_matrices(
        self,
        indices,
        copy
    ) -> list[PETScMat | None] | None:
        if self._matrices is None:
            return
        return [array_matrix(mat, indices, copy) if mat is not None else None for mat in self._matrices]

    def get_vectors(
        self,
        indices,
        copy
    ) -> list[PETScVec | None] | None:
        if self._vectors is None:
            return
        return [array_vector(vec, indices, copy) if vec is not None else None for vec in self._vectors]

    @property 
    def solver(self) -> PETSc.KSP | None:
        return self._solver
    
    @property
    def bcs(self) -> list[DirichletBCMetaClass]:
        return self._bcs
    
    @property
    def mpc(self) -> MultiPointConstraint | None:
        return self._mpc
    
    @property
    def bilinear_forms(self) -> list[Form]:
        return self._a_forms
    
    @property
    def linear_forms(self) -> list[Form]:
        return self._l_forms
    
    @property
    def scalings(self) -> list[float | Constant]:
        return self._scalings
    
    @property
    def cache_matrix(self) -> bool | EllipsisType | Iterable[bool | EllipsisType]:
        """
        If `True`, matrix is assembled only in the first `solve` call and is subsequently cached.
        If `False`, matrix is reassembled in each `solve` call.
        If `...`, arguments are compared to their value in the previous `solve` call, 
        and matrix is reassembled if any argument value has changed.
        """
        return self._cache_matrix
    
    @cache_matrix.setter
    def cache_matrix(self, value):
        if isinstance(value, Iterable):
            assert all(isinstance(i, (bool, EllipsisType)) for i in value)
            value = tuple(value)
        else:
            assert isinstance(value, (bool, EllipsisType))
        self._cache_matrix = value
    
    @property
    def assemble_termwise(self) -> tuple[bool, bool]:
        """Default value `(use_matrix_partition, use_vector_partition)`"""
        return self._assemble_termwise

    @assemble_termwise.setter
    def assemble_termwise(self, value):
        assert isinstance(value, tuple)
        assert len(value) == 2
        self._assemble_termwise = value
    

P = ParamSpec("P")
class InitialBoundaryValueProblem(BoundaryValueProblem):

    petsc_default = OptionsPETSc.default()
    jit_default = OptionsJIT.default()
    ffcx_default = OptionsFFCX.default()

    def __init__(
        self,
        solution: FunctionSeries,
        forms: Iterable[Form | tuple[Constant | float, Form]],
        forms_init : Iterable[Form | tuple[Constant | float, Form]] | None = None,
        n_init: int | None = None,
        ics: Function | Constant | Perturbation | Callable | float | Iterable[float] | None = None,
        bcs: BoundaryConditions | None = None,
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] = False,
        assemble_termwise: tuple[bool, bool] = (False, False),
        future: bool = True,
        overwrite: bool = False,
    ) -> None:
        if ics is not None:
            solution.initialize_from_ics(ics)
        
        super().__init__(
            solution, 
            forms, 
            bcs, 
            petsc, 
            jit, 
            ffcx, 
            corrector, 
            cache_matrix, 
            assemble_termwise, 
            future,
            overwrite,
        )

        if forms_init is not None:
            self._initial = InitialBoundaryValueProblem(
                solution, 
                forms_init, 
                bcs=bcs, 
                petsc=petsc, 
                jit=jit, 
                ffcx=ffcx, 
                corrector=corrector,
                cache_matrix=cache_matrix, 
                assemble_termwise=assemble_termwise,
            )
            assert n_init is not None and n_init > 0
            self._n_init = n_init
        else:
            self._initial = None
            assert n_init is None or n_init == 0
            self._n_init = 0

    @classmethod
    def from_forms_func(
        cls,
        forms_func: Callable[P, Iterable[Form | tuple[Constant | float, Form]]],
        ics: Function | Constant | Perturbation | None = None,
        bcs: BoundaryConditions | None = None,
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] = False,
        assemble_termwise: tuple[bool, bool] = (False, False),
        future: bool = True,
        overwrite: bool = False,
        solution: Function | FunctionSeries | None = None, 
    ):
        """
        If the `solution` argument is not provided, it will be inferred
        as the zeroth argument to `forms_func`.
        """
        def _create(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Self:
            nonlocal solution
            if solution is None:
                if args:
                    solution = args[0]
                else:
                    solution = list(kwargs.values())[0]
            forms = forms_func(*args, **kwargs)

            def _init(arg):
                if isinstance(arg, (FiniteDifference, FiniteDifferenceArgwise)):
                    if arg.initial is not None:
                        return arg.initial
                    else:
                        return arg
                else:
                    return arg

            order = finite_difference_order(*args, *kwargs.values())
            if order is None:
                raise ValueError(f'No `{FiniteDifference.__name__}` arguments from which to deduce order.')
            if order > 1:
                args_init = [_init(i) for i in args]
                kwargs_init = {k: _init(v) for k, v in kwargs.items()}
                forms_init = forms_func(*args_init, **kwargs_init)
                n_init = order - 1
            else:
                forms_init = None
                n_init = None

            return cls(
                solution, 
                forms, 
                forms_init,
                n_init, 
                ics, 
                bcs, 
                petsc, 
                jit, 
                ffcx, 
                corrector, 
                cache_matrix, 
                assemble_termwise, 
                future, 
                overwrite,
            )
        
        return _create

    @property
    def initial(self) -> Self | None:
        return self._initial

    @property
    def n_init(self) -> int:
        return self._n_init


P = ParamSpec("P")
class InitialValueProblem(InitialBoundaryValueProblem):

    petsc_default = OptionsPETSc.default()
    jit_default = OptionsJIT.default()
    ffcx_default = OptionsFFCX.default()

    def __init__(
        self,
        uxt: FunctionSeries,
        forms: Iterable[Form | tuple[Constant | float, Form]],
        forms_init : tuple[int, Iterable[Form | tuple[Constant | float, Form]]] | None = None,
        ics: Function | Constant | Perturbation | Callable | float | Iterable[float] | None = None,
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] = False,
        assemble_termwise: tuple[bool, bool] = (False, False),
        future: bool = True,
        overwrite: bool = False,
    ) -> None:
        bcs = None
        super().__init__(
            uxt,
            forms, 
            forms_init, 
            ics, 
            bcs, 
            petsc, 
            jit, 
            ffcx, 
            corrector, 
            cache_matrix, 
            assemble_termwise, 
            future,
            overwrite,
        )

    @classmethod
    def from_forms_func(
        cls,
        forms_func: Callable[P, Iterable[Form | tuple[Constant | float, Form]]],
        ics: Function | Constant | Perturbation | None = None,
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] = False,
        assemble_termwise: tuple[bool, bool] = (False, False),
        future: bool = True,
        overwrite: bool = False,
        solution: Function | FunctionSeries | None = None, 
    ):
        bcs = None
        return InitialBoundaryValueProblem.from_forms_func(
            forms_func,
            ics,
            bcs,
            petsc,
            jit,
            ffcx,
            corrector,
            cache_matrix,
            assemble_termwise,
            future,
            solution,
            overwrite,
        )


class EigenvalueProblem:

    slepc_default = OptionsSLEPc.default()
    jit_default = OptionsJIT.default()
    ffcx_default = OptionsFFCX.default()

    @classmethod
    def set_defaults(
        cls,
        slepc=None,
        jit=None,
        ffcx=None,
    ):
        if slepc is None:
            slepc = OptionsSLEPc.default()
        if jit is None:
            jit = OptionsJIT.default()
        if ffcx is None:
            ffcx = OptionsFFCX.default()
        cls.slepc_default = slepc
        cls.jit_default = jit
        cls.ffcx_default = ffcx
    
    def __init__(
        self,
        solutions: list[Function] | list[FunctionSeries] | FunctionSpace | tuple[Mesh, str, int],
        forms: tuple[Form, Form],
        bcs: BoundaryConditions 
        | list[DirichletBCMetaClass] 
        | tuple[list[DirichletBCMetaClass], MultiPointConstraint] 
        | None = None,
        slepc: OptionsSLEPc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        cache_matrix: bool | EllipsisType | tuple[bool | EllipsisType, bool | EllipsisType] = False,
        future: bool = False,
        overwrite: bool = False,
    ) -> None:
        
        if slepc is None:
            slepc = self.slepc_default
        if jit is None:
            jit = self.jit_default
        if ffcx is None:
            ffcx = self.ffcx_default
        slepc = dict(slepc)
        jit = dict(jit)
        ffcx = dict(ffcx)
        
        EPS_NEV_ATTR = 'eps_nev'
        if isinstance(solutions, list):
            nev = len(eigenfunctions)
            slepc[EPS_NEV_ATTR] = nev
            fs = solutions[0].function_space
            names = [i.name for i in solutions]
            if all(isinstance(i, Function) for i in solutions):
                eigenfunctions = solutions
                eigenseries = [FunctionSeries(fs, n) for n in names]
            elif all(isinstance(i, FunctionSeries) for i in solutions):
                eigenfunctions = [Function(fs, name=n) for n in names]
                eigenseries = solutions
            else:
                raise TypeError
        else:
            nev = slepc.get(EPS_NEV_ATTR, self.slepc_default.eps_nev)
            fs = create_fem_space(solutions)
            eigenfunctions = [Function(fs) for _ in range(nev)]
            eigenseries = [FunctionSeries(fs) for _ in range(nev)]
        
        if bcs is None:
            bcs = []
        if isinstance(bcs, list):
            self._bcs = bcs
            self._mpc = None
        elif isinstance(bcs, tuple):
            self._bcs, self._mpc = bcs
        else:
            if bcs.create_weak_bcs(fs):
                raise RuntimeError(f'{self.__class__.__name__} cannot have a linear form `l(v)` with test function `v`.')
            self._bcs = bcs.create_strong_bcs(fs)        
            self._mpc = bcs.create_periodic_bcs(fs)

        if self._mpc is not None and not self._mpc.finalized:
            self._mpc.finalize()

        _create_form = partial(
            meta_form,
            jit_options=jit,
            form_compiler_options=ffcx,
        )
        _create_matrix = partial(create_matrix, mpc=self._mpc)

        for f in forms:
            if not rhs(f).empty():
                raise RuntimeError(f'{self.__class__.__name__} cannot have a linear form `l(v)` with test function `v`.')

        form_lhs, form_rhs = forms
        self._forms = form_lhs, form_rhs
        self._metaform_lhs = _create_form(form_lhs)
        self._metaform_rhs = _create_form(form_rhs)
        self._matrix_lhs = _create_matrix(self._metaform_lhs)
        self._matrix_rhs = _create_matrix(self._metaform_rhs)

        prefix = f"{self.__class__.__name__}_{id(self)}"
        self._solver = SLEPc.EPS().create(fs.mesh.comm)
        self._solver.setOperators(self._matrix_lhs, self._matrix_rhs)
        self._solver.setOptionsPrefix(prefix)
        set_from_options(self._solver, slepc)

        self._n_requested = nev
        self._n_converged = None
        self._eigenvalues = [None] * nev
        self._eigenfunctions = eigenfunctions
        self._eigenseries = eigenseries
        self._cache_matrix = cache_matrix
        self._future = future
        self._overwrite = overwrite

    @classmethod
    def from_forms_func(
        cls,
        forms_func: Callable[P, tuple[Form, Form]],
        bcs: BoundaryConditions | None = None,
        slepc: OptionsSLEPc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        cache_matrix: bool | EllipsisType | tuple[bool | EllipsisType, bool | EllipsisType] = False,
        future: bool = False,
        overwrite: bool = False,
        solutions: list[Function] | list[FunctionSeries] | FunctionSpace | tuple[Mesh, str, int] | None = None,
    ):
        def _create(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Self:
            nonlocal solutions
            if solutions is None:
                if args:
                    solutions = args[0]
                else:
                    solutions = list(kwargs.values())[0]
                if isinstance(solutions, (Function, FunctionSeries)):
                    solutions = solutions.function_space
            return cls(
                solutions,
                forms_func(*args, **kwargs),
                bcs, 
                slepc,
                jit,
                ffcx,
                cache_matrix,
                future,
                overwrite,
            )

        return _create

    def solve(
        self,
        future: bool | None = None,
        overwrite: bool | None = None,
        cache_matrix: bool | EllipsisType | tuple[bool | EllipsisType, bool | EllipsisType] | None = None,
    ) -> None:
        if future is None:
            future = self._future
        if overwrite is None:
            overwrite = self._overwrite
        if cache_matrix is None:
            cache_matrix = self.cache_matrix
        if not isinstance(cache_matrix, tuple):
            cache_matrix = (cache_matrix, cache_matrix)
        cache_lhs, cache_rhs = cache_matrix

        assemble_matrix(self._matrix_lhs, self._metaform_lhs, self._bcs, cache=cache_lhs)
        assemble_matrix(self._matrix_rhs, self._metaform_rhs, self._bcs, cache=cache_rhs)

        self._solver.setOperators(self._matrix_lhs, self._matrix_rhs)
        self._solver.solve()

        self._n_converged = self._solver.getConverged()
        for n in range(min(self._n_converged, self._n_requested)):
            self._eigenvalues[n] = self._solver.getEigenvalue(n)
            self._solver.getEigenpair(n, self._eigenfunctions[n].vector)
            self._eigenfunctions[n].x.scatter_forward()
            self._eigenseries[n].update(self._eigenfunctions[n], future, overwrite)

    def forward(self, t: float | Constant | np.ndarray) -> None:
        for i in range(self._n_converged):
            self._eigenseries[i].forward(t)

    def get_matrices(
        self,
        indices: tuple[Iterable[int], Iterable[int]] | Literal["dense"] | None = None,
        copy: bool = False,
    ) -> tuple[PETScMat, PETScMat] | tuple[np.ndarray, np.ndarray]:
        return (
            array_matrix(self._matrix_lhs, indices, copy), 
            array_matrix(self._matrix_rhs, indices, copy), 
        )

    @property    
    def solver(self) -> SLEPc.EPS:
        return self._solver
    
    @property
    def bilinear_forms(self) -> tuple[Form, Form]:
        return self._forms
    
    @property
    def cache_matrix(self) -> bool | EllipsisType | tuple[bool | EllipsisType, bool | EllipsisType]:
        return self._cache_matrix
    
    @cache_matrix.setter
    def cache_matrix(self, value):
        if isinstance(value, Iterable):
            assert all(isinstance(i, (bool, EllipsisType)) for i in value)
            value = tuple(value)
        else:
            assert isinstance(value, (bool, EllipsisType))
        self._cache_matrix = value
         
    @property
    def n_converged(self) -> int | None:
        return self._n_converged

    @property
    def eigenvalues(self) -> list[float]: 
        return self._eigenvalues
    
    @property
    def eigenfunctions(self) -> list[Function]:
        return self._eigenfunctions

    @property
    def eigenseries(self) -> list[FunctionSeries]:
        return self._eigenseries
        

P = ParamSpec("P")
class Projection(BoundaryValueProblem):

    petsc_default = OptionsPETSc.default()
    jit_default = OptionsJIT.default()
    ffcx_default = OptionsFFCX.default()

    def __init__(
        self, 
        solution: Function | FunctionSeries,
        expression: Function | Expr,
        bcs: BoundaryConditions | Iterable[tuple[SpatialMarkerAlias, Value] | tuple[SpatialMarkerAlias, Value, SubspaceIndex]] | None = None, 
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        future: bool = False,
        overwrite: bool = False,
    ):

        v = TestFunction(solution.function_space)
        u = TrialFunction(solution.function_space)
        dx = Measure('dx')
        F_lhs = inner(v, u) * dx
        F_rhs = inner(v, expression) * dx
        forms = [F_lhs, -F_rhs]

        if isinstance(bcs, Iterable):
            bcs = BoundaryConditions(*bcs)

        super().__init__(
            solution, 
            forms, 
            bcs, 
            petsc, 
            jit, 
            ffcx, 
            corrector, 
            cache_matrix=True, 
            assemble_termwise=(False, False), 
            future=future, 
            overwrite=overwrite,
        )

    @classmethod
    def from_expr_func(
        cls,
        solution: Function | FunctionSeries, 
        expr_func: Callable[P, Function | Expr],
        bcs: BoundaryConditions | Iterable[tuple[SpatialMarkerAlias, Value] | tuple[SpatialMarkerAlias, Value, SubspaceIndex]] | None = None, 
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        future: bool = False,
        overwrite: bool = False,
    ):
        """from function"""
        def _create(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Self:
            return cls(
                solution, 
                expr_func(*args, **kwargs), 
                bcs, 
                petsc, 
                jit, 
                ffcx, 
                corrector,
                future,
                overwrite,
            )
        return _create


@replicate_callable(BoundaryValueProblem.from_forms_func)
def bvp():
    pass

@replicate_callable(InitialBoundaryValueProblem.from_forms_func)
def ibvp():
    pass

@replicate_callable(InitialValueProblem.from_forms_func)
def ivp():
    pass

@replicate_callable(EigenvalueProblem.from_forms_func)
def evp():
    pass

@replicate_callable(Projection.from_expr_func)
def projection():
    pass