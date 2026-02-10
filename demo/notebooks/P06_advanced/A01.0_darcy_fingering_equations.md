# Darcy fingering equations

Governing equations for miscible viscous fingering.

## Dimensional equations

$$
\begin{align*}
&\text{Find} \\
&c(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R}, \\
&\textbf{u}(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R}^d, \\
&p(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R} \\
&\text{such that} \\
&\mathbb{IBVP}
\begin{cases}
\phi\frac{\partial c}{\partial t} + \textbf{u}\cdot\nabla c = \nabla\cdot(\mathsf{D}\cdot\nabla c) & \\
\nabla\cdot\textbf{u} = 0 & \\
\textbf{u}=-\frac{\mathsf{K}}{\mu}\cdot\nabla p & \forall(\textbf{x}, t)\in\Omega\times[0,\infty) \\
c=c_0 & \forall(\textbf{x}, t)\in\Omega\times\{0\} \\
c=c_{\text{D}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{D}, c} \times [0,\infty] \\
\textbf{n}\cdot(\mathsf{D}\cdot\nabla c) = c_{\text{N}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{N}, c}
\times [0,\infty]~,~\partial\Omega_{\text{N}, c}=\partial\Omega/\partial\Omega_{\text{D}, c} \\
\textbf{n}\cdot\textbf{u} = u_{\text{E}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{E}} \times [0,\infty] \\
p = p_{\text{N}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{N}}\times [0,\infty]~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{E}}
\end{cases} \\
&\text{given} \\
&\mathbb{S}=
\begin{cases}
\Omega & \text{domain}\\
c_0(\textbf{x}) & \text{concentration initial condition}\\
c_{\text{D}}(\textbf{x}, t)~,~\partial\Omega_{\text{D},c} & \text{concentration Dirichlet boundary condition} \\
c_{\text{N}}(\textbf{x}, t)~,~\partial\Omega_{\text{N},c} & \text{concentration Neumann boundary condition} \\
u_{\text{E}}(\textbf{x}, t)~,~\partial\Omega_{\text{E}} & \text{normal velocity essential boundary condition} \\
p_{\text{N}}(\textbf{x}, t)~,~\partial\Omega_{\text{N}} & \text{pressure natural boundary condition} \\
\phi(\textbf{x}) & \text{porosity}\\
\mathsf{K}(\phi) & \text{permeability}\\
\mathsf{D}(\phi, \textbf{u}) & \text{solutal dispersion}\\
\mu(c) & \text{viscosity}\\
\end{cases}
\end{align*}
$$

## Non-dimensionalization

### Frame of reference

$$
\begin{align*}
\textbf{u}&\to\textbf{u}-\textbf{u}_{\text{in}} \\
\textbf{x}&\to\textbf{x}-t\,\textbf{u}_{\text{in}} \\
\end{align*}
$$

### Scalings

| Quantity | $\vert\textbf{x}\vert$ | $\vert\textbf{u}\vert$ | $t$ | $c$ | $p$ | $\psi$ | $\mu$ | $\phi$ | $K$ | $\vert\mathsf{D}\vert$ |
| -------- | ------- | ------- | ------- | ------- | -------  |  ------- |  ------- | ------- | ------- | ------- | 
| **Scaling** | $\mathcal{L}$ | $\mathcal{U}$ |$\mathcal{T}$ | $\Delta c$ | $\mu_{\text{ref}}\,\mathcal{U}\mathcal{L}/K_{\text{ref}}$ | $\mathcal{U}\mathcal{L}$ | $\mu_{\text{ref}}$ | $\phi_{\text{ref}}$ |$K_{\text{ref}}$ | $D_{\text{ref}}$ |


### Abstract dimensionless numbers

$$
Ad=\frac{\mathcal{U}\mathcal{T}}{\phi_{\text{ref}}\mathcal{L}}~,~
Di=\frac{D_{\text{ref}}\mathcal{T}}{\phi_{\text{ref}}\mathcal{L}^2}~,~
In=\frac{\vert\textbf{u}_{\text{in}}\vert}{\mathcal{U}}~,~
X=\frac{\mathcal{L}_\Omega}{\mathcal{L}}
$$

### Physical dimensionless numbers

| Definition | Name | Physical interpretation | 
| -------- | ------- | ------- |
| $Pe=\frac{\vert\textbf{u}_{\text{in}}\vert\mathcal{L}_\Omega}{D_{\text{ref}}}$ |  Peclet number  | Ratio of injection to diffusive speeds. |

### Scaling choice

| Name | $\mathcal{L}$ | $\mathcal{U}$ |$ \mathcal{T}$ | $\{Ad, Di, In, X\}$ | Examples | 
| -------- | -------- | ------- | ------- | ------- | ------- |
| advective | $\mathcal{L}_\Omega$  | $\vert\textbf{u}_{\text{in}}\vert$ | $\phi_{\text{ref}}\mathcal{L}/\mathcal{U}$ | $\{1, 1/Pe, 1, 1\}$| [Nijjer et al. (2012)](https://www.cambridge.org/core/product/identifier/S0022112017008291/type/journal_article) |
| diffusive | $\mathcal{L}_\Omega$  | $D_{\text{ref}}/\mathcal{L}$ | $\phi_{\text{ref}}\mathcal{L}/\mathcal{U}$ | $\{1, 1, Pe, 1\}$| ... |

## Non-dimensional time-discretized equations
 
### Strong form

$$
\begin{align*}
&\text{Find}~c^{n+1}, \theta^{n+1},~\textbf{u}^n,~p^n~\text{such that}~\forall n\geq0 \\
&\begin{cases}
\phi\frac{c^{n+1}-c^n}{\Delta t^n}+Ad\,\mathcal{D}_{\textbf{u}, c}(\textbf{u}\cdot\nabla c) = Di\nabla\cdot\mathcal{D}_{\mathsf{D},c}(\mathsf{D}(\phi, \textbf{u})\cdot\nabla c) & \\
\nabla\cdot\textbf{u}^n = 0 & \\
\textbf{u}^n + In\,\textbf{e}_{\text{in}} =-\frac{\mathsf{K}}{\mu^n}\cdot\nabla p^n \iff \textbf{u}^n = -\frac{\mathsf{K}}{\mu^n}\cdot\nabla(p^n + In\,\mu^n\mathsf{K}^{-1}\cdot\textbf{e}_{\text{in}} ) \\
c^0=c_0 \\
c^n\vert_{\partial\Omega_{\text{D}, c}}=c^n_{\text{D}} \\
\left(\textbf{n}\cdot(\mathsf{D}^n\cdot\nabla c^n)\right)\vert\partial\Omega_{\text{N}, c} = c_{\text{N}}^n \\
(\textbf{n}\cdot\textbf{u}^n)\vert_{\partial\Omega_{\text{E}}} = u^n_{\text{E}}\\
p^n\vert_{\partial\Omega_{\text{N}}} = p^n_{\text{N}} \\
\end{cases}~.
\end{align*}
$$

### Weak forms

Equivalent to [Darcy convection equations](../P05_convection/C01.0_darcy_convection_equations.md).