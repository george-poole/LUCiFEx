# Notation

Throughout these notebooks a consistent notation shall be used as far as possible.


| Symbol(s) | Description |
| -------- | ------- |
| *Space*| |
| $\Omega$ | domain |
| $\partial\Omega$ | domain boundary |
| $\partial\Omega_{\text{B}}\subset\partial\Omega$ | subset of the domain boundary for a boundary type $\text{B}$ |
| $\text{d}\Omega$ | integration measure over the cells | 
| $\text{d}\Gamma$ | integration measure over the cell facets | 
| $\textbf{e}_x, \textbf{e}_y, \textbf{e}_z$ | unit vectors | 
| $\textbf{x}=\begin{pmatrix} x \\ y \\ z\end{pmatrix} = x\textbf{e}_x + y\textbf{e}_y + z\textbf{e}_z$ | spatial coordinates |
| $~$ | |
| *Time*| |
| $t$ | time |
| $\Delta t$ | time-step |
| $t^n$ | time at the $n^{\text{th}}$ time-level |
| $\mathcal{D}$ | finite difference operator |
| $\mathcal{D}_{u,w,\dots}=\mathcal{D}_u\circ\mathcal{D}_w\dots$ | argument-wise finite difference operator |
| $\mathscr{D}^{\text{IM}}_u$ | the set of all finite difference operators that are explicit with respect to $u$ |
| $\mathscr{D}^{\text{EX}}_u$ | the set of all finite difference operators that are explicit with respect to $u$ |
| $\text{FE}$ | forward Euler finite difference operator |
| $\text{BE}$ | backward Euler finite difference operator |
| $\text{CN}$ | Crank-Nicolson finite difference operator |
| $\text{AB}_n$ | order $n$ Adams-Bashforth finite difference operator |
| $\text{AM}_n$ | order $n$ Adams-Moulton finite difference operator |
| $~$ | |
| *Partial differential equations*| |
| $\mathscr{L}_{\textbf{x}}$ | spatial differential operator |
| $\mathscr{L}_{\textbf{x},t}$ | spatial and temporal differential operator |
| $\mathbb{BVP}_u$ | boundary value problem to solve for $u$ |
| $\mathbb{IBVP}_u$ | initial boundary value problem to solve for $u$ |
| $\mathbb{IVP}_u$ | initial value problem to solve for $u$ |
| $\mathbb{EVP}_u$ | eigenvalue problem to solve for $u$ |
| $\mathbb{S}_u$ | specification of a problem solving for $u$ |
| $u_0$ | initial condition on $u$ |
| $u_\text{D}$ | Dirichlet boundary condition on $u$ |
| $u_\text{E}$ | essential boundary condition on $u$ |
| $u_\text{N}$ | Neumann or natural boundary condition on $u$ |
| $u_\text{S}$ | strong boundary condition on $u$ |
| $u_\text{W}$ | weak boundary condition on $u$ |
| $u_\text{R}$ | Robin boundary condition on $u$ |
| $u_\text{I}$ | inflow boundary condition on $u$ |
| $u_\text{O}$ | outflow boundary condition on $u$ |
| $~$ | |
| *Finite element method*| |
| $\mathscr{T}$ | tesselation of the domain | 
| $\bigcup_{\mathcal{K}\in\mathscr{T}} \mathcal{K}$ | union of cells forming the mesh | 
| $h$ | local cell size |
| $\mathcal{F}$ | set of cell facets |
| $\mathcal{V}$ | set of cell vertices |
| $V_u$ | trial function space to which $u$ belongs |
| $\hat{V}_v$ | test function space to which $v$ belongs  |
| $\xi_j(\mathbf{x})$ | finite element basis functions |  
| $\sum_jU_j\xi_j$ | finite element approximation | 
| $\{U_j\}_j\leftrightarrow\textbf{U}$| degrees of freedom vector |
| $L^2(\Omega)$ | Lebesgue function space on the domain $\Omega$ | 
| $H^1(\Omega)$ | Sobolev function space on the domain $\Omega$ | 
| $C^0(\Omega)$ | set of continuous functions on the domain $\Omega$ |
| $\mathrm{P}_k$ | continuous Lagrange element of degree $k$ | 
| $\mathrm{DP}_k$ | discontinuous Lagrange element of degree $k$ | 
| $\mathrm{BDM}_k$ | Brezzi–Douglas–Marini element of degree $k$ | |
| $\mathcal{R}$ | residual of the equation |  |
| $\left[\!\left[ \cdot \right]\!\right]$ | cell facet jump operator |
| $\{\cdot\}$ | cell facet average operator |
| $\tau_{\text{SUPG}}$ | SUPG stabilization parameter | 
| $\alpha_{\text{DG}}$ | DG penalty parameter | 
| $\alpha_{\text{W}}$ | weak enforcement penalty parameter | 
| $\mathcal{E}$ | the error in the numerical solution | 
| $\mathbb{F}_{u,w\dots}$ | sequence of weak forms solving for $u,w\dots$|
| $~$ | |
| *General mathematics*| |
| $u, v, \dots$ | scalar quantity | 
| $\textbf{u}, \textbf{v}, \dots$ | vector quantity | 
| $\textbf{u} = \begin{pmatrix} u_x \\ u_y \\ u_z\end{pmatrix}=u_x\textbf{e}_x + u_y\textbf{e}_y + u_z\textbf{e}_z$ | vector quantity components | 
| $\mathsf{U}, \mathsf{V}, \dots$ | tensor quantity | 
| $\mathsf{U} = \begin{pmatrix} U_{xx} &  U_{xy} &  U_{xz} \\ U_{yx} &  U_{yy} & U_{yz} \\ U_{zx} &  U_{zy} & U_{zz} \end{pmatrix} $ | tensor quantity components | 
| $\dfrac{\mathrm{d}}{\mathrm{d}x}$ | ordinary derivative operator | 
| $\partial_x = \dfrac{\partial}{\partial x}$ | partial derivative operator | 
| $\nabla = (\partial_x, \partial_y, \partial_z)$ | gradient operator | 
| $\mathbf{n}$ | outward unit normal vector |  
| $\mathrm{H}$ | Heaviside step function | 
| $\mathsf{I}$ | identity tensor | 
| $\det$ | matrix determinant |  
| $\min_{\mathbf{x} \in \Omega}$ | minimum over the domain $\Omega$ |  
| $\max_{\mathbf{x} \in \Omega}$ | maximum over the domain $\Omega$ |  
| $\mathrm{vol}(\Omega)$ | volume of domain $\Omega$ |  
| $\langle \cdot \rangle_{\Omega}$ | space-averaging over the domain $\Omega$ |  
| $\overline{\;\cdot\;}^{[t,t']}$ | time-averaging over interval $[t,t']$ |  
| $\mathbb{R}$ | the set of real numbers |
| $\mathbb{C}$ | the set of complex numbers |
| $~$ | |
| *Fluid mechanics*| |
| $\textbf{u}$ | velocity |
| $p$| pressure |
| $\psi$ | streamfunction |
| $\boldsymbol{\omega}$| vorticity |
| $\rho$ | fluid density |
| $\mu$ | fluid viscosity |
| $\textbf{f}$ | body force |
| $\tau$ | deviatoric stress |
| $\phi$ | porosity |
| $\mathsf{K}$ | permeability |
| $c$ | solute concentration | 
| $\theta$ | temperature | 
| $\mathsf{D}$ | solutal dispersion |
| $\mathsf{G}$ | thermal dispersion |
| $g$ | gravity constant |
| $\,{\textbf{e}}_g$ | gravity unit vector |
| $~$ | |
| *Acronyms*| |
| PDE | partial differential equation |
| ODE | ordinary differential equation |
| DNS | direct numerical simulation |
| CG | continuous Galerkin |
| DG | discontinuous Galerkin |
| SUPG | streamline-upwind Petrov–Galerkin |
| CFL | Courant–Friedrichs–Lewy |