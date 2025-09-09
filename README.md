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


**Running simulations**

A simulation in `lucifex` is in effect a sequence of (linear) problems to be solved sequentially, over and over again in a time-stepping loop. Given the sequence of problems `solvers`, time  `t` and timestep `dt`

```python
from lucifex.sim import Simulation
simulation = Simulation(solvers, t, dt)
```

Whilst options for writing data to disk during the course of the simulation can be configured separately, the `create_simulation` decorator provides a succinct and convenient way of doing so.

```python
from lucifex.fdm import AB2, FiniteDifference
from lucifex.sim import create_simulation

@create_simulation(
    # default I/O configuration specified here
    write_step=..., 
    write_file=...,
    dir_base=...,
    dir_params=...,
)
def diffusion_1d(Lx: float, Nx: int, dt: float, Dfdm: FiniteDifference):
    ... # code defining solvers, time and timestep
    return solvers, t, dt
```

Now given the simulation's chosen parameters `Lx, Nx, dt, Ddiff`, we can configure and create a simulation by

```python
# simulation with default I/O configuration
simulation = diffusion_1d(Lx, Nx, dt, Ddiff)
# writing data every 3 integration steps
simulation = diffusion_1d(write_step=3)(Lx, Nx, dt, Ddiff) 
# writing data every 2.0 time units
simulation = diffusion_1d(write_step=2.0)(Lx, Nx, dt, Ddiff) 
# writing data to directory `./data/Lx=2.0|Dt=CN`
simulation = diffusion_1d(dir_base='./data', dir_params=('Lx', 'Dfdm'))(Lx, Nx, dt, Ddiff) 
```
 
Integration over time is then performed by the `integrate` routine

```python
from lucifex.sim import integrate
n_stop, t_stop = 10, 2.5
integrate(simulation, n_stop, t_stop)
```

if working in an interactive, hands-on iPython environment (ideal for demonstration, prototyping and testing purposes). In a script designed to be run from the command line, we instead have the `integrate_from_cli` routine 

```python
# `simulate .py` script
if __name__ == "__main__":
    simulation = integrate_from_cli(diffusion_1d)
```

which will create a command line interface from the `diffusion_1d` function into which its arguments `Lx, Nx, dt, Dfdm, ...` and arguments to the usual `integrate` function can be supplied like so

```bash
python simulate.py --Lx 2.0 --Nx 10 --n_stop 10 --t_stop 2.5
```

**Postprocessing**

The decorators `postprocess` and `co_postprocess` enable functions acting on the saved simulation data (e.g. to create a plot) to be called using a convenient short-hand syntax, avoiding the need to explicitly load data in advance and clutter one's script with repetitive statements. 

```python
from matplotlib.figure import Figure
from lucifex.fdm import FunctionSeries
from lucifex.io import postprocess, proxy, load_mesh, load_function_series, write

@postprocess
def plot_figure(u: FunctionSeries) -> Figure:
    ... # code creating figure

@co_postprocess
def co_plot_figure(u: list[FunctionSeries]) -> Figure:
    ... # code creating figure

# long-hand 
mesh = load_mesh(mesh_name, dir_path, u_file)
u = load_function_series(u_name, dir_path, u_file, mesh)
fig = plot_figure(u)
write(fig, fig_name, fig_dir)
# short-hand creating and returning figure
figure = plot_figure(dir_name)(proxy((u_name, u_type, u_file))) 
# short-hand creating and writing figure, then returning `None`
plot_figure(dir_name, fig_name, fig_dir)(proxy((u_name, u_type, u_file, mesh_name)))
```

The `postprocess` decorator furthermore enables the decorated function to act on either a single data directory, or a sequence of data directories. The latter enables the batch-postprocessing of an ensemble of simulation directories in which each individual directory has the same stucture (e.g. the `FunctionSeries` object `u` has been written with the same name and to the same filename).

```python
dir_paths = [dir_path_0, dir_path_1, ...]
# short-hand creating and writing figure in each directory
plot_figure(dir_paths, fig_name, fig_dir)(proxy((u_name, u_type, u_file)))
```

The `co_postprocess` decorator has a slightly different purpose, namely to combine data from an ensemble of directories into a single object (e.g. a plot comparing data as a simulation parameter is changed, or a mean quantity averaged across simulations). For example, we may be interested in comparing the results obtained by different choices of the finite difference operator, in which case the ensemble of simulation directories may look like

```
data/
|___Lx=1.0|Dfdm=AB1/
|___Lx=1.0|Dfdm=AB2/
|___Lx=1.0|Dfdm=AM1/
|___Lx=1.0|Dfdm=AM2/
|___Lx=1.0|Dfdm=CN/
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

See `/demo/ipynb/` and `demo/py/`. The 'A'-stream is more application-focussed with examples from fluid mechanics and porous media, whereas the 'B'-stream illustrates technical details from the `lucifex` package.

## TODO List

These features remain to be implemented as part of ongoing development:

+ further documentation and testing
+ update to latest version of `fenicsx` (currently on 0.6.0)
+ parallelisation with `mpi4py`

## What does LUCiFEx *not* do?

These features are outside the scope of current development, but could be of interest in the future:

+ adaptive mesh refinement
+ time-dependent meshes
+ time-stepping with Runge-Kutta methods