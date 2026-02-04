class FormError(RuntimeError):
    def __init__(
        self, 
        cls, 
        name: str,
        msg: str,
    ):
        msg = f"{cls.__name__} solving for '{name}' {msg}"
        super().__init__(msg)


class BilinearFormError(FormError):
    """
    Error to raise if no bilinear form `a(u,v)` can be deduced from the form `F(u, v) = a(u,v) - l(v)`.
    """
    def __init__(self, cls, name):
        super().__init__(cls, name, 'requires a bilinear form.')


class LinearFormError(FormError):
    """
    Error to raise if no linear form `l(v)` can be deduced from the form `F(u, v) = a(u,v) - l(v)`.
    """
    def __init__(self, cls, name):
        super().__init__(cls, name, 'requires a linear form. Check that this is not an eigenvalue problem.')


class NonlinearFormError(FormError):
    """
    Error to raise if a nonlinearform is found`.
    """
    def __init__(self, cls, name):
        super().__init__(cls, name, 'contains a nonlinear form. Check finite difference discretizations')


class EigenvalueFormError(FormError):
    """
    Error to raise if linear form is found in the eigenvalue forms `F(u, v) = L(u, v) - λR(u,v).`
    """
    def __init__(self, cls, name):
        super().__init__(cls, name, 'cannot have a linear form. Check that this is not a boundary problem.')


class UnsolvedFormError(FormError):
    def __init__(self, cls, name):
        super().__init__(cls, name, "contains an unsolved quantity. Check finite difference discretizations and sequence of solvers.")