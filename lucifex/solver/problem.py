from typing import (
    Literal,
    Callable,
    ParamSpec,
    Any,
)
from collections.abc import Iterable
from types import EllipsisType
from typing_extensions import Self
from functools import partial

import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc
from ufl import Form, Measure, rhs, TestFunction, TrialFunction
from ufl.core.expr import Expr
from dolfinx.mesh import Mesh
from dolfinx.fem import FunctionSpace
from dolfinx.fem.petsc import DirichletBCMetaClass
from dolfinx_mpc import MultiPointConstraint

from ..utils.fenicsx_utils import (
    create_function_space, Marker, BlockForm, Scaled,
    is_scaled_type, is_none, extract_bilinear_form, extract_linear_form,
)
from ..utils.py_utils import replicate_callable
from ..fdm import (
    FiniteDifference, FiniteDifferenceArgwise, 
    FunctionSeries, finite_difference_order,
)
from ..fdm.ufl_operators import inner
from ..fem import Function, Constant, Perturbation, is_unsolved
from .bcs import BoundaryConditions, Value, SubspaceIndex
from .options import (
    OptionsPETSc, OptionsSLEPc,
    OptionsFFCX, OptionsJIT, set_from_options,
)
from .petsc import (
    create_metaform,
    assemble_petsc_matrix,
    assemble_petsc_vector,
    create_petsc_matrix,
    create_petsc_vector,
    sum_petsc_matrix,
    sum_petsc_vector,
    view_petsc_matrix,
    view_petsc_vector,
    PETScMat,
    PETScVec,
)
from .eval import Solver
from .utils import BilinearFormError, LinearFormError, EigenvalueFormError, UnsolvedFormError


P = ParamSpec("P")
class BoundaryValueProblem(Solver[Function, FunctionSeries]):

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
        """
        `cls.set_defaults()` without any arguments will reset defaults.
        """
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
        forms: Form | Scaled[Form | BlockForm] | BlockForm 
        | Iterable[Form | Scaled[Form]] 
        | Iterable[BlockForm | Scaled[BlockForm]],
        bcs: BoundaryConditions 
        | Iterable[DirichletBCMetaClass | MultiPointConstraint] 
        | None = None,
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] = False,
        assemble_termwise: tuple[bool, bool] = (False, False),
        pc_form: Form | BlockForm | None = None,
        future: bool | int = False,
        overwrite: bool = False,
    ) -> None:
        
        super().__init__(solution, corrector, future, overwrite)
        
        if petsc is None:
            petsc = self.petsc_default
        if jit is None:
            jit = self.jit_default
        if ffcx is None:
            ffcx = self.ffcx_default
        petsc = dict(petsc)
        jit = dict(jit)
        ffcx = dict(ffcx)

        # variational forms
        if isinstance(forms, (Form, BlockForm)) or is_scaled_type(forms):
            forms = (forms, )
        if isinstance(bcs, BoundaryConditions):
            forms = (*forms, *bcs.create_weak_bcs(solution))
        self._scalings: list[float | Constant] = [i[0] if is_scaled_type(i) else 1.0 for i in forms]
        self._forms: list[Form] | list[BlockForm] = [i[1] if is_scaled_type(i) else i for i in forms]

        if all(isinstance(i, Form) or i is None for i in (*self._forms, pc_form)):
            self._blocked = False
        elif all(isinstance(i, BlockForm) or i is None for i in (*self._forms, pc_form)):
            self._blocked = True
        else:
            raise RuntimeError('Cannot have non-blocked and blocked forms together.')
        
        # bilinear forms
        self._a_forms = [extract_bilinear_form(i) for i in self._forms]
        self._a_form_termwise = sum([i for i in self._a_forms if i is not None])
        self._a_form_nontermwise = sum(
            [i * j for i, j in zip(self._scalings, self._a_forms, strict=True) if j is not None]
        )
        if is_none(self._a_form_termwise) or is_none(self._a_form_nontermwise):
            raise BilinearFormError(self.__class__, solution.name)
        
        # linear forms
        self._l_forms = [extract_linear_form(i) for i in self._forms]
        self._l_form_termwise: Form | BlockForm = sum([i for i in self._l_forms if i is not None])
        self._l_form_nontermwise = sum(
            [i * j for i, j in zip(self._scalings, self._l_forms, strict=True) if j is not None]
        )
        if is_none(self._l_form_termwise, any) or is_none(self._l_form_nontermwise, any):
            raise LinearFormError(self.__class__, solution.name)

        # set block subspaces if applicable
        if self._blocked:
            self._block_subspaces = solution.function_subspaces
            # self._block_subspaces = self._l_form_termwise.block_subspaces
        else:
            self._block_subspaces = None

        # essential boundary conditions and constraints
        if bcs is None:
            bcs = []
        if isinstance(bcs, BoundaryConditions):
            self._bcs = bcs.create_strong_bcs(solution.function_space, fs_sub=self._block_subspaces)   
            constraints = []
            mpc = bcs.create_periodic_mpc(solution.function_space, finalize=True) # TODO include dirichlet bcs here?
            if mpc is not None:
                constraints.append(mpc)
        else:
            self._bcs = [i for i in bcs if isinstance(i, DirichletBCMetaClass)]
            constraints = [i for i in bcs if isinstance(i, MultiPointConstraint)]
        for mpc in constraints:
            if not mpc.finalized:
                raise RuntimeError('Multipoint constraints must be finalized.')
        
        # blocked constraints if applicable
        if self._blocked:
            self._mpc = constraints
        else:
            if len(constraints) > 1:
                raise RuntimeError('Multiple multipoint constraints are for blocked solvers.')
            if not constraints:
                self._mpc = None
            else:
                self._mpc = constraints[0]
    
        # setters
        self._create_metaform = partial(
            create_metaform,
            jit_options=jit,
            ffcx_options=ffcx,
        )
        self._create_matrix = partial(create_petsc_matrix, mpc=self._mpc)
        prefix = f"{self.__class__.__name__}_{id(self)}"
        self._set_solver = lambda solver: (solver.setOptionsPrefix(prefix), set_from_options(solver, petsc))
        self._set_matvec = lambda matvec: (matvec.setOptionsPrefix(prefix), matvec.setFromOptions())
        # initialization status
        self._init_pc_matrix = False
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
        # preconditioner attributes
        self._pc_form = pc_form
        self._pc_metaform = None
        self._pc_matrix = None
        # working attributes
        self._matrix = None
        self._vector = None
        self._solution_vector = None
        self._solver = None
        self._dofmaps_slices = None
        # defaults
        self._assemble_termwise = assemble_termwise
        self._cache_matrix = cache_matrix
        # low-level options
        self._petsc = petsc
        self._jit = jit
        self._ffcx = ffcx

    def _create_pc_matrix(self) -> None:
        if self._init_pc_matrix:
            return
        assert self._pc_form is not None
        self._pc_metaform = self._create_metaform(self._pc_form)
        self._pc_matrix = self._create_matrix(self._pc_metaform)
        self._init_pc_matrix = True

    def _create_matrix_termwise(self) -> None:
        if self._init_matrix_termwise:
            return
        self._a_metaforms = [
            self._create_metaform(ai) if ai is not None else None for ai in self._a_forms]
        self._a_metaform_termwise = self._create_metaform(self._a_form_termwise)
        self._matrices = [
            self._create_matrix(ai) if ai is not None else None
            for ai in self._a_metaforms
        ]
        self._matrix_termwise = self._create_matrix(self._a_metaform_termwise)
        self._solver_termwise = PETSc.KSP().create(self.solution.function_space.mesh.comm)
        self._solver_termwise.setOperators(self._matrix_termwise, self._pc_matrix)
        self._set_solver(self._solver_termwise)
        for mat in (*self._matrices, self._matrix_termwise):
            if mat is not None:
                self._set_matvec(mat)
        if self._pc_matrix is not None:
            self._set_matvec(self._pc_matrix)
        self._init_matrix_termwise = True

    def _create_matrix_nontermwise(self) -> None:
        if self._init_matrix_nontermwise:
            return
        self._a_metaform_nontermwise = self._create_metaform(self._a_form_nontermwise)
        self._matrix_nontermwise = self._create_matrix(self._a_metaform_nontermwise)
        self._solver_nontermwise = PETSc.KSP().create(self.solution.function_space.mesh.comm)
        self._solver_nontermwise.setOperators(self._matrix_nontermwise, self._pc_matrix)
        self._set_solver(self._solver_nontermwise)
        self._set_matvec(self._matrix_nontermwise)
        if self._pc_matrix is not None:
            self._set_matvec(self._pc_matrix)
        self._init_matrix_nontermwise = True

    def _create_vector_termwise(self) -> None:
        if self._init_vector_termwise:
            return
        self._l_metaforms = [self._create_metaform(i) if i is not None else None for i in self._l_forms]
        self._l_metaform_termwise = self._create_metaform(self._l_form_termwise)
        self._vectors = [
            create_petsc_vector(i) if i is not None else None
            for i in self._l_metaforms
        ]
        self._vector_termwise = create_petsc_vector(self._l_metaform_termwise)
        for vec in (*self._vectors, self._vector_termwise):
            if vec is not None:
                self._set_matvec(vec)
        self._init_vector_termwise = True

    def _create_vector_nontermwise(self) -> None:
        if self._init_vector_nontermwise:
            return
        self._l_metaform_nontermwise = self._create_metaform(self._l_form_nontermwise)
        self._vector_nontermwise = create_petsc_vector(self._l_metaform_nontermwise)
        self._set_matvec(self._vector_nontermwise)    
        self._init_vector_nontermwise = True

    def _get_solution_vector(self) -> PETScVec:
        if self._solution_vector is None:
            if not self._blocked:
                self._solution_vector = self._solution.vector
            else:
                self._solution_vector =  self._matrix.createVecLeft() # TODO Right()
        return self._solution_vector
    
    @classmethod
    def from_forms_factory(
        cls,
        forms_factory: Callable[P, Iterable[Form | Scaled[Form]] | Form],
        bcs: BoundaryConditions 
        | Iterable[DirichletBCMetaClass | MultiPointConstraint] 
        | None = None,
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] = False,
        assemble_termwise: tuple[bool, bool] = (False, False),
        pc_form: Form | BlockForm | None = None,
        future: bool | int = False,
        overwrite: bool = False,
        solution: Function | FunctionSeries | None = None, 
    ):
        """
        If the `solution` argument is not provided, it will be inferred
        as the zeroth argument to `forms_factory`.
        """
        def _create(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Self:
            nonlocal solution
            if solution is None:
                solution = _deduce_from_args(0, args, kwargs)
            return cls(
                solution, 
                forms_factory(*args, **kwargs), 
                bcs, 
                petsc, 
                jit, 
                ffcx, 
                corrector,
                cache_matrix, 
                assemble_termwise,
                pc_form,
                future,
                overwrite,
            )
        return _create

    def solve(
        self,
        future: bool | int | None = None,
        overwrite: bool | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] | None = None,
        assemble_termwise: tuple[bool, bool] | None = None,
        cache_pc: bool = True,
        assert_solved: bool = False,
    ) -> None:
        """
        Mutates `self.solution` and `self.series`
        """
        if assert_solved:
            if any(is_unsolved(i) for i in self.forms):
                raise UnsolvedFormError(self.__class__, self.solution.name)
        if cache_matrix is None:
            cache_matrix = self._cache_matrix
        if assemble_termwise is None:
            assemble_termwise = self._assemble_termwise

        if self._pc_form is not None:
            self._create_pc_matrix()
            assemble_petsc_matrix(
                self._pc_matrix, 
                self._pc_metaform, 
                self._bcs, 
                self._mpc, 
                cache=cache_pc,
            )

        matrix_termwise, vector_termwise = assemble_termwise

        if matrix_termwise:
            self._create_matrix_termwise()
            if not isinstance(cache_matrix, Iterable):
                cache_matrix = [cache_matrix] * len(self._matrices)
            [
                assemble_petsc_matrix(m, a, self._bcs, self._mpc, cache=c)
                for m, a, c in zip(self._matrices, self._a_metaforms, cache_matrix, strict=True)
                if (m, a) != (None, None)
            ]
            sum_petsc_matrix(
                self._matrix_termwise,
                self._matrices,
                self._scalings,
                (self._bcs, self._a_metaform_termwise.function_spaces),
            )
            self._matrix = self._matrix_termwise
            self._solver = self._solver_termwise
        else:
            self._create_matrix_nontermwise()
            assemble_petsc_matrix(
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
                assemble_petsc_vector(v, l, (self._bcs, a), self._mpc)
                for v, l, a in zip(self._vectors, self._l_metaforms, self._a_metaforms, strict=True)
                if (v, l, a) != (None, None, None)
            ]
            sum_petsc_vector(
                self._vector_termwise, 
                self._vectors, 
                self._scalings, 
                (self._bcs, self._l_metaform_termwise.function_spaces),
            )
            self._vector = self._vector_termwise
        else:
            self._create_vector_nontermwise()
            assemble_petsc_vector(
                self._vector_nontermwise,
                self._l_metaform_nontermwise,
                (self._bcs, self._a_metaform_nontermwise),
                self._mpc,
            )
            self._vector = self._vector_nontermwise

        vec = self._get_solution_vector()
        self._solver.solve(self._vector, vec)

        if self._blocked:   
            dofmaps, slices = self._get_block_dofmaps_slices()     
            for dfmp, slc in zip(dofmaps, slices, strict=True):
                self._solution.x.array[dfmp] = vec[slc]
        
        self._solution.x.scatter_forward()
        
        if self._mpc:
            if self._blocked:
                [mpc.backsubstitution(vec) for mpc in self._mpc]
            else:
                self._mpc.backsubstitution(vec)

        super().solve(future, overwrite)

    def _get_block_dofmaps_slices(
        self,
    ) -> tuple[list[np.ndarray], list[slice]]:
        if self._dofmaps_slices is None:
            fs = self._solution.function_space
            n_sub = fs.num_sub_spaces

            dofmaps: list[np.ndarray] = []
            offsets: list[int | None] = [0]
            for i in range(n_sub):
                fs_sub, dfmp = fs.sub(i).collapse()
                dofmaps.append(dfmp)
                ofst = fs_sub.dofmap.index_map.size_local * fs_sub.dofmap.index_map_bs
                offsets.append(ofst + offsets[-1])
            offsets.append(None)

            slices = []
            for i in range(n_sub):
                slc = slice(offsets[i], offsets[i + 1])
                slices.append(slc)

            self._dofmaps_slices = dofmaps, slices

        return self._dofmaps_slices

    def get_matrix(
        self,
        indices: tuple[Iterable[int], Iterable[int]] | Literal["dense"] | None = None,
        copy: bool = False,
    ) -> PETScMat | np.ndarray | None:
        if self._matrix is None:
            return None
        return view_petsc_matrix(self._matrix, indices, copy)
    
    def get_pc_matrix(
        self,
        indices: tuple[Iterable[int], Iterable[int]] | Literal["dense"] | None = None,
        copy: bool = False,
    ) -> PETScMat | np.ndarray | None:
        if self._pc_matrix is None:
            return None
        return view_petsc_matrix(self._pc_matrix, indices, copy)

    def get_vector(
        self,
        indices: int | Iterable[int] | Literal["dense"] | None = None,
        copy: bool = False,
    ) -> PETScVec | np.ndarray | None:
        if self._vector is None:
            return None
        return view_petsc_vector(self._vector, indices, copy)
    
    def get_matrices(
        self,
        indices,
        copy
    ) -> list[PETScMat | None] | None:
        if self._matrices is None:
            return
        return [view_petsc_matrix(mat, indices, copy) if mat is not None else None for mat in self._matrices]

    def get_vectors(
        self,
        indices,
        copy
    ) -> list[PETScVec | None] | None:
        if self._vectors is None:
            return
        return [view_petsc_vector(vec, indices, copy) if vec is not None else None for vec in self._vectors]

    @property 
    def solver(self) -> PETSc.KSP | None:
        return self._solver
    
    @property
    def blocked(self) -> bool:
        return self._blocked
    
    @property
    def bcs(self) -> list[DirichletBCMetaClass]:
        return self._bcs
    
    @property
    def mpc(self) -> MultiPointConstraint | list[MultiPointConstraint] | None:
        return self._mpc
    
    @property
    def forms(self) -> list[Form] | list[BlockForm]:
        return self._forms
    
    @property
    def bilinear_forms(self) -> list[Form] | list[BlockForm]:
        return self._a_forms
    
    @property
    def linear_forms(self) -> list[Form] | list[BlockForm]:
        return self._l_forms
    
    @property
    def scalings(self) -> list[float | Constant]:
        return self._scalings
    
    @property
    def cache_matrix(self) -> bool | EllipsisType | Iterable[bool | EllipsisType]:
        """
        If `True`, matrix is assembled only in the first `solve` call and is subsequently cached.
        If `False`, matrix is reassembled in each `solve` call.
        If `Ellipsis`, `Function` and `Constant` objects used to create the original `Form` object
        have their present values compared against their previous values, 
        and the matrix is reassembled if any value has changed.
        """
        return self._cache_matrix
    
    @property
    def petsc(self):
        return self._petsc
    
    @property
    def jit(self):
        return self._jit
    
    @property
    def ffcx(self):
        return self._ffcx
    
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
        pc_form: Form | BlockForm | None = None,
        future: bool | int = True,
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
            pc_form,
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
                pc_form=pc_form,
            )
            assert n_init is not None and n_init > 0
            self._n_init = n_init
        else:
            self._initial = None
            assert n_init is None or n_init == 0
            self._n_init = 0

    @classmethod
    def from_forms_factory(
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
        pc_form: Form | BlockForm | None = None,
        future: bool | int = True,
        overwrite: bool = False,
        solution: FunctionSeries | None = None, 
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
                solution = _deduce_from_args(0, args, kwargs)
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
                pc_form,
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
        solution: FunctionSeries,
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
        pc_form: Form | BlockForm | None = None,
        future: bool | int = True,
        overwrite: bool = False,
    ) -> None:
        bcs = None
        super().__init__(
            solution,
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
            pc_form,
            future,
            overwrite,
        )

    @classmethod
    def from_forms_factory(
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
        pc_form: Form | BlockForm | None = None,
        future: bool | int = True,
        overwrite: bool = False,
        solution: FunctionSeries | None = None, 
    ):
        bcs = None
        return InitialBoundaryValueProblem.from_forms_factory(
            forms_func,
            ics,
            bcs,
            petsc,
            jit,
            ffcx,
            corrector,
            cache_matrix,
            assemble_termwise,
            pc_form,
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
        | Iterable[DirichletBCMetaClass | MultiPointConstraint] 
        | None = None,
        slepc: OptionsSLEPc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        cache_matrix: bool | EllipsisType | tuple[bool | EllipsisType, bool | EllipsisType] = False,
        future: bool | int = False,
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
            fs = create_function_space(solutions)
            eigenfunctions = [Function(fs) for _ in range(nev)]
            eigenseries = [FunctionSeries(fs) for _ in range(nev)]

        if bcs is None:
            bcs = []
        if isinstance(bcs, BoundaryConditions):
            forms_weak_bcs = bcs.create_weak_bcs(fs)
            self._bcs = bcs.create_strong_bcs(fs)        
            mpcs = []
            mpc = bcs.create_periodic_mpc(fs)
            if mpc is not None:
                mpcs.append(mpc)
        else:
            self._bcs = [i for i in bcs if isinstance(i, DirichletBCMetaClass)]
            mpcs = [i for i in bcs if isinstance(i, MultiPointConstraint)]

        for mpc in mpcs:
            if not mpc.finalized:
                mpc.finalize()

        self._blocked = False # FIXME
        if self._blocked:
            self._mpc = mpcs
        else:
            if len(mpcs) > 1:
                raise RuntimeError('Multiple multipoint constraints are for blocked solvers.')
            if not mpcs:
                self._mpc = None
            else:
                self._mpc = mpcs[0]

        _create_form = partial(
            create_metaform,
            jit_options=jit,
            ffcx_options=ffcx,
        )
        _create_matrix = partial(create_petsc_matrix, mpc=self._mpc)

        form_lhs, form_rhs = forms
        if forms_weak_bcs:
            form_lhs += sum(forms_weak_bcs)

        for f in (form_lhs, form_rhs):
            if not rhs(f).empty():
                raise EigenvalueFormError(self.__class__, eigenfunctions[0].name)

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

        self._slepcs = slepc
        self._jit = jit
        self._ffcx = ffcx

    @classmethod
    def from_forms_factory(
        cls,
        forms_factory: Callable[P, tuple[Form, Form]],
        bcs: BoundaryConditions | None = None,
        slepc: OptionsSLEPc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        cache_matrix: bool | EllipsisType | tuple[bool | EllipsisType, bool | EllipsisType] = False,
        future: bool | int = False,
        overwrite: bool = False,
        solutions: list[Function] | list[FunctionSeries] | FunctionSpace | tuple[Mesh, str, int] | None = None,
    ):
        def _create(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Self:
            nonlocal solutions
            if solutions is None:
                solutions = _deduce_from_args(0, args, kwargs)
                if isinstance(solutions, (Function, FunctionSeries)):
                    solutions = solutions.function_space
            return cls(
                solutions,
                forms_factory(*args, **kwargs),
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
        future: bool | int | None = None,
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

        assemble_petsc_matrix(self._matrix_lhs, self._metaform_lhs, self._bcs, cache=cache_lhs)
        assemble_petsc_matrix(self._matrix_rhs, self._metaform_rhs, self._bcs, cache=cache_rhs)

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
            view_petsc_matrix(self._matrix_lhs, indices, copy), 
            view_petsc_matrix(self._matrix_rhs, indices, copy), 
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
    def eigenvalues(self) -> list[complex]: 
        return self._eigenvalues
    
    @property
    def eigenfunctions(self) -> list[Function]:
        return self._eigenfunctions

    @property
    def eigenseries(self) -> list[FunctionSeries]:
        return self._eigenseries
    
    @property
    def slepc(self):
        return self._slepc
    
    @property
    def jit(self):
        return self._jit
    
    @property
    def ffcx(self):
        return self._ffcx
        

P = ParamSpec("P")
class Projection(BoundaryValueProblem):

    petsc_default = OptionsPETSc.default()
    jit_default = OptionsJIT.default()
    ffcx_default = OptionsFFCX.default()

    def __init__(
        self, 
        solution: Function | FunctionSeries,
        expression: Function | Expr,
        bcs: BoundaryConditions 
        | Iterable[tuple[Marker, Value] | tuple[Marker, Value, SubspaceIndex]] 
        | None = None, 
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        future: bool | int = False,
        overwrite: bool = False,
    ):
        v = TestFunction(solution.function_space)
        u = TrialFunction(solution.function_space)
        dx = Measure('dx', solution.function_space.mesh)
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
    def from_expr_factory(
        cls,
        solution: Function | FunctionSeries, 
        expr_factory: Callable[P, Function | Expr],
        bcs: BoundaryConditions 
        | Iterable[tuple[Marker, Value] | tuple[Marker, Value, SubspaceIndex]] 
        | None = None, 
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        corrector: Callable[[np.ndarray], None] 
        | tuple[str, Callable[[np.ndarray], None]] 
        | None = None,
        future: bool | int = False,
        overwrite: bool = False,
    ):
        def _create(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> Self:
            return cls(
                solution, 
                expr_factory(*args, **kwargs), 
                bcs, 
                petsc, 
                jit, 
                ffcx, 
                corrector,
                future,
                overwrite,
            )
        return _create


@replicate_callable(BoundaryValueProblem.from_forms_factory)
def bvp():
    pass

@replicate_callable(InitialBoundaryValueProblem.from_forms_factory)
def ibvp():
    pass

@replicate_callable(InitialValueProblem.from_forms_factory)
def ivp():
    pass

@replicate_callable(EigenvalueProblem.from_forms_factory)
def evp():
    pass

@replicate_callable(Projection.from_expr_factory)
def projection():
    pass


def _deduce_from_args(
    index: int | str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
):
    if args and isinstance(index, int):
        return args[index]
    if not args and isinstance(index, int):
        return list(kwargs.values())[index]
    if isinstance(index, str):
        return kwargs[index]
    raise TypeError