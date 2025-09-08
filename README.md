# LUCiFEx

Welcome to the ***Linearized Convection in FEniCSx*** package! 

Development has been primarily motivated by the numerical study of 2D convection in porous media, however the tools developed are general-purpose and widely-applicable.

## What does LUCiFEx do?

**Finite differences in time**

Consider the diffusion equation

$$\frac{\partial u}{\partial t} = \nabla^2 u$$

Even this simplest of IBVPs can be discretized in a number of ways, for example

$$~~~\text{Crank-Nicolson}:~~~\frac{u^{n+1} - u^n}{\Delta t} = \nabla^2(\tfrac{1}{2}u^{n+1} + \tfrac{1}{2}u^n)$$
$$~~~\text{forward Euler}:~~~\frac{u^{n+1} - u^n}{\Delta t} = \nabla^2u^{n+1}$$
$$~~~\text{backward Euler}:~~~\frac{u^{n+1} - u^n}{\Delta t} = \nabla^2u^{n}$$

and many more from the Adams-Bashforth, Adams-Moulton and backward differentiation formulae families. How can we avoid hard-coding a particular discretization?

**Time-dependent quantities**

To represent a finite element function in `fenicsx`, we write
```python
from dolfinx.fem import Function
u = Function(mesh, function_space)
```
In `lucifex`, we represent a time-dependent finite element function by

```python
from lucifex.fdm import FunctionSeries
u = FunctionSeries(mesh, function_space)
```

and can access its past, present and future values as
```
u[-1], u[0], u[1]
```

**Time-discretizations in UFL**

The `FiniteDifference` operators enable finite-difference discretizations to be written alongside finite element formulations written in `ufl`. For example, the second-order Adams-Bashforth discretization is written as
```python
from lucifex.fdm import DT, CN
F = v * DT(u, dt) * dx
F += inner(grad(v), grad(CN(u))) * dx
```

Moreover `lucifex` overloads many of `ufl`'s existing operators, such as `grad`, such that `grad(CN(u))` is equivalent to `CN(grad(u))`.


**Simulation management**

A `Simulation` object is created from the sequence of a problems that are to solved repeatedly in a time loop. The decorator function `create_simulation` turns an appropriate user-defined function into a simulation factory.

```python
@create_simulation(
    write_step=2,
)
def diffusion_simulation(...):
    return solvers, t, dt
```

****

interactive, hands-on prototyping in iPython environments
`integrate(simulation)`

long-running, hands-off simulations from the command line
+ many handy utilities

**Postprocessing**

The decorators `postprocess` and `co_postprocess` enable calculations and plots to be made and saved from the paths of a batch or ensemble of data directories, without having to clutter the workspace with repeated `load` statements.

```python
from lucifex.io import postprocess, proxy

@postprocess
def plot_figure(u):
    ...

plot_figure(data_dirs, fig_name, fig_dir)(proxy((u_name, u_type, u_file)))
```

## Installation (macOS)


`git clone https://github.com/george-poole/LUCiFEx.git`

See `conda` directory for files to recreate Conda environment. To create the 
`lucifex` environment, do one of

* `conda create -n lucifex` <br>
`conda install --file conda_explicit.txt` <br>
(requirements file created by `conda list --explicit > conda_explicit.txt`)

* `conda create -n lucifex` <br>
`conda install x --file conda.txt` <br>
(requirements file created by `conda list > conda.txt`)

* `conda env create --name lucifex -f conda_from_history.yml` <br>
(environment file created by `conda env export --from-history > conda_from_history.yml`)

* `conda env create --name lucifex -f conda.yml` <br>
(environment file created by `conda env export > conda.yml`)

Finally `conda activate lucifex` and `pip install .` (or `pip install -e .` for editable mode).

## Demos

See `/demo/ipynb/` and `demo/py/`.

## TODO List

These features remain to be implemented as part of ongoing development:

+ update to latest version of `fenicsx` (currently on 0.6.0)
+ parallelisation with `mpi4py`

## What does LUCiFEx *not* do?

These features are outside the scope of current development, but could be of interest in the future:

+ adaptive mesh refinement
+ time-dependent meshes
+ Runge-Kutta integration