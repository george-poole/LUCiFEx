from dolfinx.fem import FunctionSpace as DOLFINxFunctionSpace

class FunctionSpace(DOLFINxFunctionSpace):

    def __hash__(self):
        return id(self)
