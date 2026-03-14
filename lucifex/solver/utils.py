class FormError(RuntimeError):
    def __init__(
        self, 
        solver_cls, 
        name: str,
        msg: str,
    ):
        msg = f"{solver_cls.__name__} solving for '{name}' {msg}"
        super().__init__(msg)


class BilinearFormError(FormError):
    def __init__(self, solver_cls, name):
        """
        Error to raise if no bilinear form `a(u,v)` can be deduced from the form `F(u, v) = a(u,v) - l(v)`.
        """
        super().__init__(solver_cls, name, 'requires a bilinear form.')


class LinearFormError(FormError):
    def __init__(self, solver_cls, name):
        """
        Error to raise if no linear form `l(v)` can be deduced from the form `F(u, v) = a(u,v) - l(v)`.
        """
        super().__init__(
            solver_cls, 
            name, 
            'requires a linear form. Check that this is not an eigenvalue problem, or try adding a zero form.',
        )


class NonlinearFormError(FormError):
    def __init__(self, solver_cls, name):
        """
        Error to raise if a non-linear form is found`.
        """
        super().__init__(
            solver_cls, 
            name, 
            'contains a non-linear form. Check finite difference discretizations.',
        )


class EigenvalueFormError(FormError):
    def __init__(self, solver_cls, name):
        """
        Error to raise if a linear form is found in the eigenvalue forms `F(u, v) = L(u, v) - λR(u,v).`
        """
        super().__init__(
            solver_cls, 
            name, 
            'cannot have a linear form. Check that this is not a boundary problem.',
        )


class UnsolvedFormError(FormError):
    def __init__(self, solver_cls, name):
        """
        Error to raise if a form contains an unsolved `Function` or `Constant`.
        """
        super().__init__(
            solver_cls, 
            name, 
            "contains an unsolved quantity. Check finite difference discretizations and sequence of solvers.",
        )