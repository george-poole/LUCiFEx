# LUCiFEx 😈

Welcome to the *&nbsp;**L**inearized **U**nderground **C**onvection **i**n **FE**niCS**x**&nbsp;* package! 

LUCiFEx is a flexible and user-friendly package for the applied mathematician to solve time-dependent PDEs numerically by the finite element method using [FEniCSx](https://github.com/FEniCS/dolfinx). To get started with LUCiFEx, refer to the [user guide](https://george-poole.github.io/LUCiFEx/notebooks/P01_tutorial/U00_introduction.html).

Development has primarily been motivated by the numerical study of two-dimensional convection in porous media, however the tools developed are general-purpose and widely-applicable. For any queries, comments or feedback do not hesitate to email `grp39@cantab.ac.uk`.

## Demo

The complete gallery of examples can be viewed [here](https://george-poole.github.io/LUCiFEx/gallery.html). See `./demo/notebooks/` to browse their source code.  Examples from fluid dynamics include:
* Darcy equations formulated in terms of either velocity and pressure $\textbf{u}$, $p$ or the streamfunction $\psi$
* Stokes equations formulated in terms of either velocity and pressure $\textbf{u}$, $p$ or the streamfunction $\psi$
* Navier-Stokes equations formulated in terms of either velocity and pressure $\textbf{u}$, $p$ or the streamfunction and vorticity $\psi$, $\omega$
* advection-diffusion-reaction equations for the transport of solute and/or heat coupled to fluid flow
* classic instabilities such as Rayleigh-Bénard and Rayleigh-Taylor
<!-- * discontinuous Galerkin and stabilization methods for advection-dominated transport equations -->

These short notebooks are primarily for quick, inexpensive demonstration purposes and as such most do not use a high-resolution mesh, sophisticated stabilization methods or low-level problem-specific optimizations. Users are encouraged to use them as a learning tool and a stepping stone to more detailed numerical experimentation. See `./bench/` for more rigorous benchmarking.

## What does LUCiFEx do?

In addition to what can be achieved with FEniCSx, LUCiFEx provides abstractions and utilities for
+ [time-dependent functions, constants and expressions](demo/notebooks/P01_tutorial/U01_time_dependence.ipynb)
+ [finite difference operators for the discretization of time-dependent functions, constants and expressions](demo/notebooks/P01_tutorial/U02_finite_differences.ipynb)
+ setting initial conditions with optional perturbations
+ imposing (possibly time-dependent) essential, natural and periodic boundary conditions via a unified interface
+ solving boundary-value, initial-value, eigenvalue and evaluation problems via a unified interface
+ creating and configuring time-dependent simulations
+ running time-dependent simulations interactively in a `.ipynb` notebook
+ running time-dependent simulations in the background from a `.py` script
+ postprocessing and visualization with `numpy` and `matplotlib`

### What does LUCiFEx *not* do?

These features are outside the scope of current development, but could be of interest in the future:

+ adaptive mesh refinement
+ time-dependent domains and boundaries
+ time-stepping with Runge-Kutta methods
+ nonlinear solvers

## Installation

See `./install/INSTALL.md` .

## Further work

These features remain to be implemented as part of ongoing development:

+ update to latest version of `fenicsx` (currently on 0.6.0) and Python (currently on 3.10.12)
+ parallelisation with `mpi4py`
+ more documentation 
+ more unit testing