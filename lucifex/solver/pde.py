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
from ufl import Form, lhs, rhs
from dolfinx.mesh import Mesh
from dolfinx.fem import Constant, Function, FunctionSpace
from dolfinx.fem.petsc import create_matrix, create_vector
from petsc4py import PETSc
from slepc4py import SLEPc

from ..utils import Perturbation, replicate_callable, MultipleDispatchTypeError, dofs_limits_corrector, fem_function_space
from ..fdm import FiniteDifference, FunctionSeries, finite_difference_order

from .bcs import BoundaryConditions
from .options import (
    OptionsPETSc, OptionsSLEPc,
    OptionsFFCX, OptionsJIT, set_from_options,
)
from .petsc import (
    form,
    assemble_matrix,
    create_mpc_matrix,
    assemble_vector,
    sum_matrix,
    sum_vector,
    array_matrix,
    array_vector,
    PETScMat,
    PETScVec,
)

P = ParamSpec("P")
class BoundaryValueProblem:

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
        bcs: BoundaryConditions | None = None,
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        dofs_corrector: Callable[[Function], None] | tuple[float, float] | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] = False,
        use_partition: tuple[bool, bool] = (False, False),
        future: bool = False,
        overwrite: bool = False,
    ) -> None:
        
        if isinstance(forms, Form):
            forms = (forms, )

        if isinstance(solution, Function):
            series = FunctionSeries(solution.function_space, solution.name)
        elif isinstance(solution, FunctionSeries):
            series = solution
            solution = Function(solution.function_space, name=series.name)
        else:
            raise MultipleDispatchTypeError(solution)
        
        if bcs is None:
            bcs = BoundaryConditions()

        if petsc is None:
            petsc = self.petsc_default
        if jit is None:
            jit = self.jit_default
        if ffcx is None:
            ffcx = self.ffcx_default
        petsc = dict(petsc)
        jit = dict(jit)
        ffcx = dict(ffcx)

        forms_weak_bcs = bcs.create_weak_bcs(solution.function_space)
        forms = (*forms, *forms_weak_bcs)
        scalings = [i[0] if isinstance(i, tuple) else 1.0 for i in forms]
        self._scalings = scalings
        forms = [i[1] if isinstance(i, tuple) else i for i in forms]

        self._bcs = bcs.create_strong_bcs(solution.function_space)        
        # TODO when dof affected by more than one mpc
        self._mpc = bcs.create_periodic_bcs(solution.function_space, self._bcs)
        
        if isinstance(dofs_corrector, tuple):
            self._dofs_corrector = lambda u: dofs_limits_corrector(u, dofs_corrector)
        else:
            self._dofs_corrector = dofs_corrector

        _form = partial(
            form,
            jit_options=jit,
            form_compiler_options=ffcx,
        )
        if self._mpc:
            _create_matrix = partial(create_mpc_matrix, mpc=self._mpc)
        else:
            _create_matrix = create_matrix

        self._a_forms = [lhs(i) if len(i.arguments()) == 2 else None for i in forms]
        self._a_sum_unscaled: Form = sum([i for i in self._a_forms if i is not None])
        if self._a_sum_unscaled == 0:
            raise RuntimeError('Variational problem with test and trial functions `v, u` must have a bilinear form `a(u,v)`.')
        self._a_sum = sum(
            [i * j for i, j in zip(self._scalings, self._a_forms, strict=True) if j is not None]
        )
        self._a_forms_compiled = [_form(ai) if ai is not None else None for ai in self._a_forms]
        self._a_sum_unscaled_compiled = _form(self._a_sum_unscaled)
        self._a_sum_compiled = _form(self._a_sum)
        self._m_forms = [
            _create_matrix(ai) if ai is not None else None
            for ai in self._a_forms_compiled
        ]
        self._m_sum_unscaled = _create_matrix(self._a_sum_unscaled_compiled)
        self._m_sum = _create_matrix(self._a_sum_compiled)
        
        self._l_forms = [rhs(i) if not rhs(i).empty() else None for i in forms]
        self._l_sum_unscaled: Form = sum([i for i in self._l_forms if i is not None])
        if self._l_sum_unscaled == 0:
            raise RuntimeError('Variational problem with test and trial functions `v, u` must have a linear form `l(v)`.')
        self._l_sum = sum(
            [i * j for i, j in zip(self._scalings, self._l_forms, strict=True) if j is not None]
        )
        self._l_forms_compiled = [_form(i) if i is not None else None for i in self._l_forms]
        self._l_sum_unscaled_compiled = _form(self._l_sum_unscaled)
        self._l_sum_compiled = _form(self._l_sum)
        self._v_forms = [
            create_vector(i) if i is not None else None
            for i in self._l_forms_compiled
        ]
        self._v_sum_unscaled = create_vector(self._l_sum_unscaled_compiled)
        self._v_sum = create_vector(self._l_sum_compiled)

        prefix = f"{self.__class__.__name__}_{id(self)}"
        self._solver_sum_unscaled = PETSc.KSP().create(solution.function_space.mesh.comm)
        self._solver_sum_unscaled.setOperators(self._m_sum_unscaled)
        self._solver_sum_unscaled.setOptionsPrefix(prefix)
        self._solver_sum = PETSc.KSP().create(solution.function_space.mesh.comm)
        self._solver_sum.setOperators(self._m_sum)
        self._solver_sum.setOptionsPrefix(prefix)
        set_from_options(self._solver_sum_unscaled, petsc)
        set_from_options(self._solver_sum, petsc)

        mv_all = (
            self._m_sum_unscaled,
            self._v_sum_unscaled,
            self._m_sum,
            self._v_sum,
            *self._m_forms,
            *self._v_forms,
        )
        for mv in mv_all:
            if mv is not None:
                mv.setOptionsPrefix(prefix)
                mv.setFromOptions()

        self._matrix = None
        self._vector = None
        self._solver = None
        self._solution = solution
        self._series = series
        self._use_partition = use_partition
        self._cache_matrix = cache_matrix
        self._future = future
        self._overwrite = overwrite

    @classmethod
    def from_forms_func(
        cls,
        forms_func: Callable[P, Iterable[Form | tuple[Constant | float, Form]] | Form],
        bcs: BoundaryConditions | None = None,
        petsc: OptionsPETSc | dict | None = None,
        jit: OptionsJIT | dict | None = None,
        ffcx: OptionsFFCX | dict | None = None,
        dofs_corrector: Callable[[Function], None] | tuple[float, float] | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] = False,
        use_partition: tuple[bool, bool] = (False, False),
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
                dofs_corrector,
                cache_matrix, 
                use_partition,
                future,
                overwrite,
            )
        return _create

    def solve(
        self,
        future: bool | None = None,
        overwrite: bool | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] | None = None,
        use_partition: tuple[bool, bool] | None = None,
    ) -> None:
        """Mutates the data structures `self.solution` and `self.series` containing the solution"""
        if future is None:
            future = self._future
        if overwrite is None:
            overwrite = self._overwrite
        if cache_matrix is None:
            cache_matrix = self.cache_matrix
        if use_partition is None:
            use_partition = self.use_partition
        use_matrix_partition, use_vector_partition = use_partition

        if not use_matrix_partition:
            assemble_matrix(
                self._m_sum,
                self._a_sum_compiled,
                self._bcs,
                self._mpc,
                cache=cache_matrix,
            )
            self._matrix = self._m_sum
            self._solver = self._solver_sum
        else:
            if not isinstance(cache_matrix, Iterable):
                cache_matrix = [cache_matrix] * len(self._m_forms)
            [
                assemble_matrix(m, a, self._bcs, self._mpc, cache=c)
                for m, a, c in zip(self._m_forms, self._a_forms_compiled, cache_matrix, strict=True)
                if (m, a) != (None, None)
            ]
            sum_matrix(
                self._m_sum_unscaled,
                self._m_forms,
                self._scalings,
                (self._bcs, self._a_sum_unscaled_compiled.function_spaces),
            )
            self._matrix = self._m_sum_unscaled
            self._solver = self._solver_sum_unscaled

        if not use_vector_partition:
            assemble_vector(
                self._v_sum,
                self._l_sum_compiled,
                (self._bcs, self._a_sum_compiled),
            )
            self._vector = self._v_sum
        else:
            [
                assemble_vector(v, l, (self._bcs, a))
                for v, l, a in zip(self._v_forms, self._l_forms_compiled, self._a_forms_compiled, strict=True)
                if (v, l, a) != (None, None, None)
            ]
            sum_vector(
                self._v_sum_unscaled, self._v_forms, self._scalings, (self._bcs, self._a_sum_unscaled_compiled)
            )
            self._vector = self._v_sum_unscaled

        if not use_vector_partition:
            self._solver.solve(self._v_sum, self._solution.vector)
        else:
            self._solver.solve(self._v_sum_unscaled, self._solution.vector)

        self._solution.x.scatter_forward()
        if self._mpc:
            self._mpc.backsubstitution(self._solution.vector)

        if self._dofs_corrector is not None:
            self._dofs_corrector(self._solution)

        self._series.update(self._solution, future, overwrite)

    def forward(self, t: float | Constant | np.ndarray) -> None:
        self._series.forward(t)

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
        
    def get_solver(self) -> PETSc.KSP | None:
        return self._solver
    
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
    def use_partition(self) -> tuple[bool, bool]:
        """Default value `(use_matrix_partition, use_vector_partition)`"""
        return self._use_partition

    @use_partition.setter
    def use_partition(self, value):
        assert isinstance(value, tuple)
        assert len(value) == 2
        self._use_partition = value

    @property
    def series(self) -> FunctionSeries:
        return self._series

    @property
    def solution(self) -> Function:
        return self._solution
    

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
        dofs_corrector: Callable[[Function], None] | tuple[float, float] | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] = False,
        use_partition: tuple[bool, bool] = (False, False),
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
            dofs_corrector, 
            cache_matrix, 
            use_partition, 
            future,
            overwrite,
        )

        if forms_init is not None:
            self._init = InitialBoundaryValueProblem(
                solution, forms_init, bcs=bcs, petsc=petsc, 
                jit=jit, ffcx=ffcx, dofs_corrector=dofs_corrector,
                cache_matrix=cache_matrix, use_partition=use_partition,
            )
            assert n_init is not None and n_init > 0
            self._n_init = n_init
        else:
            self._init = None
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
        dofs_corrector: Callable[[Function], None] | tuple[float, float] | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] = False,
        use_partition: tuple[bool, bool] = (False, False),
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
                if isinstance(arg, FiniteDifference):
                    if arg.init is not None:
                        return arg.init
                    else:
                        return arg
                elif isinstance(arg, (tuple, list)):
                    return type(arg)((_init(i) for i in arg))
                else:
                    return arg

            order = finite_difference_order(*args, *kwargs.values())
            if order is None:
                raise ValueError('No `FiniteDifference` arguments from which to deduce order.')
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
                dofs_corrector, 
                cache_matrix, 
                use_partition, 
                future, 
                overwrite,
            )
        return _create

    @property
    def init(self) -> Self | None:
        return self._init

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
        dofs_corrector: Callable[[Function], None] | tuple[float, float] | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] = False,
        use_partition: tuple[bool, bool] = (False, False),
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
            dofs_corrector, 
            cache_matrix, 
            use_partition, 
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
        dofs_corrector: Callable[[Function], None] | tuple[float, float] | None = None,
        cache_matrix: bool | EllipsisType | Iterable[bool | EllipsisType] = False,
        use_partition: tuple[bool, bool] = (False, False),
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
            dofs_corrector,
            cache_matrix,
            use_partition,
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
        bcs: BoundaryConditions | None = None,
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
            function_space = solutions[0].function_space
            names = [i.name for i in solutions]
            if all(isinstance(i, Function) for i in solutions):
                eigenfunctions = solutions
                eigenseries = [FunctionSeries(function_space, n) for n in names]
            elif all(isinstance(i, FunctionSeries) for i in solutions):
                eigenfunctions = [Function(function_space, name=n) for n in names]
                eigenseries = solutions
            else:
                raise TypeError
        else:
            nev = slepc.get(EPS_NEV_ATTR, self.slepc_default.eps_nev)
            function_space = fem_function_space(solutions)
            eigenfunctions = [Function(function_space) for _ in range(nev)]
            eigenseries = [FunctionSeries(function_space) for _ in range(nev)]
        
        if bcs is None:
            bcs = BoundaryConditions()

        self._bcs = bcs.create_strong_bcs(function_space)  

        _form = partial(
            form,
            jit_options=jit,
            form_compiler_options=ffcx,
        )

        form_lhs, form_rhs = forms
        self._form_lhs_compiled = _form(form_lhs)
        self._form_rhs_compiled = _form(form_rhs)
        self._matrix_lhs = create_matrix(self._form_lhs_compiled)
        self._matrix_rhs = create_matrix(self._form_rhs_compiled)

        prefix = f"{self.__class__.__name__}_{id(self)}"
        self._solver = SLEPc.EPS().create(function_space.mesh.comm)
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

        assemble_matrix(self._matrix_lhs, self._form_lhs_compiled, self._bcs, cache=cache_lhs)
        assemble_matrix(self._matrix_rhs, self._form_rhs_compiled, self._bcs, cache=cache_rhs)

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
        
    def get_solver(self) -> SLEPc.EPS:
        return self._solver
    
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
        
    
@replicate_callable(BoundaryValueProblem.from_forms_func)
def bvp_solver():
    pass

@replicate_callable(InitialBoundaryValueProblem.from_forms_func)
def ibvp_solver():
    pass

@replicate_callable(InitialValueProblem.from_forms_func)
def ivp_solver():
    pass

@replicate_callable(EigenvalueProblem.from_forms_func)
def evp_solver():
    pass