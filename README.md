# LUCiFEx 😈

Welcome to the *&nbsp;**L**inearized **U**nderground **C**onvection **i**n **FE**niCS**x**&nbsp;* package! 

LUCiFEx is a flexible and user-friendly package for the applied mathematician to solve time-dependent PDEs numerically by the finite element method using [FEniCSx](https://github.com/FEniCS/dolfinx). To get started with LUCiFEx, refer to the [user guide](https://george-poole.github.io/LUCiFEx/notebooks/P01_user_guide/T00_introduction.html).

Development has primarily been motivated by the numerical study of convection in 2D porous media, however the tools developed are general-purpose and widely-applicable. For any queries, comments or feedback do not hesitate to email `grp39@cam.ac.uk`.

## Demo

The complete gallery of examples can be viewed [here](https://george-poole.github.io/LUCiFEx/gallery.html). See `demo/notebooks` to browse their source code. A [user guide](https://george-poole.github.io/LUCiFEx/notebooks/P01_user_guide/U00_introduction.html) is also included.  Examples from fluid dynamics include:
* Darcy equations formulated in terms of either velocity and pressure $\textbf{u}$, $p$ or the streamfunction $\psi$
* Navier-Stokes equations formulated in terms of either velocity and pressure $\textbf{u}$, $p$ or the streamfunction and vorticity $\psi$, $\omega$
* Stokes equations formulated in terms of either velocity and pressure $\textbf{u}$, $p$ or the streamfunction $\psi$
* advection-diffusion-reaction equations for the transport of solute and/or heat coupled to fluid flow
* classic instabilities such as Rayleigh-Bénard and Rayleigh-Taylor
<!-- * discontinuous Galerkin and stabilization methods for advection-dominated transport equations -->

## What does LUCiFEx do?

In addition to what can be achieved with FEniCSx, LUCiFEx provides abstractions and utilties for
+ [time-dependent functions, constants and expressions](demo/notebooks/P01_user_guide/U01_time_dependence.ipynb)
+ [finite differences in time for the discretization of time-dependent problems](demo/notebooks/P01_user_guide/U02_finite_differences.ipynb)
+ specifying initial conditions and (possibly time-dependent) boundary conditions
+ solving boundary-value, initial-value and eigenvalue problems via a unified interface
+ creating and configuring time-dependent simulations
+ running simulations both interactively and from the command line
+ postprocessing and visualization with `numpy` and `matplotlib`

### What does LUCiFEx *not* do?

These features are outside the scope of current development, but could be of interest in the future:

+ adaptive mesh refinement
+ time-dependent domains and boundaries
+ time-stepping with Runge-Kutta methods

## Installation (macOS)

Please note that LUCiFEx is a research code still under active development.

`git clone https://github.com/george-poole/LUCiFEx.git`

See `conda` directory for files to recreate Conda environment. To create a Conda environment named `lucifex`, first do `conda create -n lucifex` followed `conda activate lucifex` and then one of

* `conda install --file conda_explicit.txt` <br>
(requirements file created by `conda list --explicit > conda_explicit.txt`)

* `conda install x --file conda.txt` <br>
(requirements file created by `conda list > conda.txt`)

or do

* `conda env create --name lucifex -f conda_from_history.yml` <br>
(environment file created by `conda env export --from-history > conda_from_history.yml`)

* `conda env create --name lucifex -f conda.yml` <br>
(environment file created by `conda env export > conda.yml`)

Finally `conda activate lucifex` and `pip install .` (or `pip install -e .` for editable mode).

## Further work

These features remain to be implemented as part of ongoing development:

+ update to latest version of `fenicsx` (currently on 0.6.0) and Python (currently on 3.10.12)
+ parallelisation with `mpi4py`
+ nested and blocked solvers
+ preconditioned solvers
+ nonlinear solvers
+ more documentation and unit testing