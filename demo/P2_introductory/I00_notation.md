# Notation

Throughout these notebooks a consistent notation shall be used as far as possible.

| Symbol(s) | Description |
| -------- | ------- |
| $\Omega$ | domain |
| $\partial\Omega$ | domain boundary |
| $\partial\Omega_i\subset\partial\Omega$ | subset of the domain boundary |
| $\text{d}\Omega$ | integration measure over the cells | 
| $\text{d}\Gamma$ | integration measure over the cell facets | 
| $\textbf{e}_x, \textbf{e}_y, \textbf{e}_z$ | unit vectors | 
| $\textbf{x}=(x, y, z) = x\textbf{e}_x + y\textbf{e}_y + z\textbf{e}_z$ | spatial coordinates |
| $t$ | time |
| $\Delta t$ | timestep |
| $\mathcal{D}(\cdot)$ | finite difference operator |
| $h(\textbf{x})$ | local cell size |
| $\mathcal{F}$ | set of cell facets |
| $\left[\!\left[ \cdot \right]\!\right]$ | cell facet jump operator |
| $\{\cdot\}$ | cell facet average operator |
| $u, v, \dots$ | scalar quantity | 
| $\textbf{u}, \textbf{v}, \dots$ | vector quantity | 
| $\textbf{u} = u_x\textbf{e}_x + u_y\textbf{e}_y + u_z\textbf{e}_z$ | vector quantity components | 
| $\mathsf{U}, \mathsf{V}, \dots$ | tensor quantity | 
| $\mathsf{U} = ((U_{xx}, U_{xy}), (U_{yx}, U_{yy})) $ | tensor quantity components | 
| $V_u$ | function space to which $u$ belongs
| $\mathbb{BVP}$ | boundary value problem |
| $\mathbb{IBVP}$ | initial boundary value problem |
| $\mathbb{IVP}$ | initial value problem |
| $\mathbb{EVP}$ | eigenvalue problem |
| $\mathbb{S}$ | specification |
| $\mathbb{F}$ | linearized or time-discretized weak forms sequence |
| $u_0$ | initial condition on $u$ |
| $u_\text{D}$ | Dirichlet boundary condition on $u$ |
| $u_\text{N}$ | Neumann or natural boundary condition on $u$ |
| $u_\text{E}$ | essential boundary condition on $u$ |
| $\mathbb{R}$ | the set of real numbers |
| $\mathbb{C}$ | the set of complex numbers |