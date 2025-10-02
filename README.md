# LUCiFEx

Welcome to the ***Linearized Convection in FEniCSx*** package! 

Development has been primarily motivated by the numerical study of 2D convection in porous media, however the tools developed are general-purpose and widely-applicable.

## What does LUCiFEx do?

**Time-dependent quantities**

A time-dependent finite element function 

$$u(\textbf{x},t)\approx\sum_ju_j(t)\xi_j(\textbf{x})$$

$$u(\textbf{x}, t\leq t^n)\approx
\begin{bmatrix}
u^{n+1}(\textbf{x}) \\
u^n(\textbf{x}) \\
u^{n-1}(\textbf{x}) \\
\vdots
\end{bmatrix}$$

$$u^n(\textbf{x})=\sum_ju_j^n\xi_j(\textbf{x})$$

is represented by the `FunctionSeries` object

```python
u = FunctionSeries(function_space, 'u', order, store)
```

and its past, present and future values are accessed as
```
u[-1], u[0], u[1]
```

If held in memory in accordance with the `store` parameter passed to `FunctionSeries`, the full sequences $[u^0(\textbf{x}), u^1(\textbf{x}), \dots]$ and $[t^0, t^1, \dots]$ are acessed as
```
u.series
u.time_series
```

**Finite differences in time**

`FiniteDifference` operators act on time-dependent quantitie to produce finite-difference discretizations. For example, the second-order Adams-Bashforth discretization of $u(\textbf{x}, t)$ 

$$u(\textbf{x}, t^n)\approx \tfrac{3}{2}u^n - \tfrac{1}{2}u^{n-1}$$


is produced by the  `AB2` operator
```python
AB2(u)
```

which is equivalent to manually writing out
```python
1.5 * u[0] - 0.5 * u[-1]
```

**Unified problem-solving interface**

Partial differential equations (linear or linearized) to be solved can be of type `BoundaryValueProblem`, `InitialBoundaryValueProblem` or `EigenvalueProblem`. Simple expressions are evaluated by solving an `EvaluationProblem`, which has subclasses `CellIntegrationProblem` and `FacetIntegrationProblem` for evaluating integrals.  Algebraic equations (linear or linearized) can be solved as a `ProjectionProblem`. 

**Time-dependent boundary conditions**

Dirichlet, Neumann, Robin and periodic conditions are specified by `BoundaryConditions`. The boundary condition's value of type `Function`, `Constant` or `Expr` can be updated in the time-stepping loop to implement a time-dependent bundary condition. 

**Running simulations**

A time-dependent simulation is in effect a sequence of (linear or linearized) problems to be solved sequentially, over and over again in a time-stepping loop. Given the sequence of problems `solvers`, time  `t` and timestep `dt`, a simulation object is defined as

```python
simulation = Simulation(solvers, t, dt)
```

The `configure_simulation` decorator functions can be used to customise the configuration of a simulation.

Integration over time is performed by the `integrate` routine

```python
integrate(simulation, n_stop, t_stop)
```

In a script designed to be run from the command line, the `integrate_from_cli` routine can instead be used

```python
integrate_from_cli(simulation)
```

to create a command line interface into which arguments for configuring, creating and integrating the simulation can be passed.

**Postprocessing**

The `grid` function converts structured meshes and finite element functions defined on structured meshes into `numpy` arrays in order to facilitate further postprocessing within the ecosystem of scientific Python packages.

```
x, y = grid(mesh)
uxy = grid(u)
```

The decorator functions `postprocess` and `co_postprocess` enable functions acting on saved simulation data (e.g. to create a plot) to be called using a convenient short-hand syntax, avoiding the need to explicitly load data in advance and write repetitive I/O routines. They furthermore enable the batch-postprocessing of an ensemble of simulation directories in which each individual directory has the same stucture (e.g. the `FunctionSeries` object `u` has been written with the same name and to the same filename).

## Installation (macOS)

Please note that `LUCiFEx` is a research code still under active development.

`git clone https://github.com/george-poole/LUCiFEx.git`

See `conda` directory for files to recreate Conda environment. To create Conda environment named `lucifex`, do one of

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

## Documentation

See `demo/` for notebooks and scripts, which are divided into three categories: `A` (application-focussed with examples of PDEs from fluid mechanics and porous media), `N` (numerical methods for solving time-dependent problems in fluid mechanics) and `T` (technical details and testing of the `lucifex` package).

## Further work

These features remain to be implemented as part of ongoing development:

+ update to latest version of `fenicsx` (currently on 0.6.0) and Python (currently on 3.10.12)
+ parallelisation with `mpi4py`
+ nested solvers
+ preconditioned solvers
+ nonlinear solvers
+ more documentation and testing

## What does LUCiFEx *not* do?

These features are outside the scope of current development, but could be of interest in the future:

+ adaptive mesh refinement
+ time-dependent domains and boundaries
+ time-stepping with Runge-Kutta methods