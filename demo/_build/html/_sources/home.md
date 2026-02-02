# LUCiFEx 😈

Welcome to the *&nbsp;**L**inearized **U**nderground **C**onvection **i**n **FE**niCS**x**&nbsp;* package! 

These demo notebooks are divided into the following categories: 

+ `T` tutorial and technical details of the LUCiFEx package
+ `I` introductory partial differential equations
+ `F` fluids 
+ `C` convection
+ `A` advanced 
+ `N` numerical methods


See `demo/notebooks` for notebooks and scripts, which are divided into three categories: `A` (applications to fluid dynamics), `F` (foundations of solving PDEs numerically), `N` (numerical methods for solving time-dependent PDEs) and `T` (technical details and testing of the LUCiFEx package). Fluid dynamics examples include:
* Darcy's equations (formulated in terms of either velocity and pressure $\textbf{u}$, $p$ or the streamfunction $\psi$)
* Navier-Stokes equations (formulated in terms of either velocity and pressure $\textbf{u}$, $p$ or the streamfunction and vorticity $\psi$, $\omega$) 
* Stokes equations
* advection-diffusion-reaction equations for the transport of solute and/or heat coupled to fluid flow
* stabilization methods for advection-dominated transport equations
* classic instability problems such as Rayleigh-Bénard convection and Saffman-Taylor fingering
* customisable perturbations to the initial conditions of instability problems
* simulations on both Cartesian and non-Cartesian domains

These short notebooks are primarily for quick, inexpensive demonstration purposes and as such most do not use a high-resolution mesh, sophisticated stabilization methods or low-level problem-specific optimizations. See `benchmark/` for case studies with more rigorous benchmarking.