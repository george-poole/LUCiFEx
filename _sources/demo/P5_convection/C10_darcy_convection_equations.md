# Darcy convection equations

Governing equations for thermosolutal convection-reaction coupled to Darcy flow.

## Dimensional equations

$$
\begin{align*}
&\text{Find} \\
&c(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R}, \\
&\theta(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R}, \\
&\textbf{u}(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R}^d, \\
&p(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R} \\
&\text{such that} \\
&\mathbb{IBVP}
\begin{cases}
\phi\frac{\partial c}{\partial t} + \textbf{u}\cdot\nabla c = \nabla\cdot(\mathsf{D}\cdot\nabla c) + \Sigma & \\
\phi\frac{\partial\theta}{\partial t} + \textbf{u}\cdot\nabla\theta = \nabla\cdot(\mathsf{G}\cdot\nabla\theta) + H& \\
\nabla\cdot\textbf{u} = 0 & \\
\textbf{u}=-\frac{\mathsf{K}}{\mu}\cdot(\nabla p - \rho g\,\textbf{e}_g) & \forall(\textbf{x}, t)\in\Omega\times[0,\infty) \\
c=c_0 & \forall(\textbf{x}, t)\in\Omega\times\{0\} \\
\theta=\theta_0 & \forall(\textbf{x}, t)\in\Omega\times\{0\} \\
c=c_{\text{D}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{D}, c} \times [0,\infty) \\
\textbf{n}\cdot(\mathsf{D}\cdot\nabla c) = c_{\text{N}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{N}, c}
\times [0,\infty)~,~\partial\Omega_{\text{N}, c}=\partial\Omega/\partial\Omega_{\text{D}, c} \\
\theta=\theta_{\text{D}} & \forall (\textbf{x}, t)\in\partial\Omega_{\text{D}, \theta} \times [0,\infty) \\
\textbf{n}\cdot(\mathsf{G}\cdot\nabla \theta) = \theta_{\text{N}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{N}, \theta}
\times [0,\infty)~,~\partial\Omega_{\text{N}, \theta}=\partial\Omega/\partial\Omega_{\text{D}, \theta} \\
\textbf{n}\cdot\textbf{u} = u_{\text{E}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{E}} \times [0,\infty) \\
p = p_{\text{N}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{N}}\times [0,\infty)~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{E}}
\end{cases} \\
&\text{given} \\
&\mathbb{S}
\begin{cases}
\Omega\subset\mathbb{R}^d & \text{domain}\\
c_0(\textbf{x}) & \text{concentration initial condition}\\
\theta_0(\textbf{x}) & \text{temperature initial condition}\\ 
c_{\text{D}}(\textbf{x}, t)~,~\partial\Omega_{\text{D},c} & \text{concentration Dirichlet boundary condition} \\
\theta_{\text{D}}(\textbf{x}, t)~,~\partial\Omega_{\text{D},\theta} & \text{temperature Dirichlet boundary condition} \\
c_{\text{N}}(\textbf{x}, t)~,~\partial\Omega_{\text{N},c} & \text{concentration Neumann boundary condition} \\
\theta_{\text{N}}(\textbf{x}, t)~,~\partial\Omega_{\text{N}, \theta} & \text{concentration Neumann boundary condition} \\
u_{\text{E}}(\textbf{x}, t)~,~\partial\Omega_{\text{E}} & \text{normal velocity essential boundary condition} \\
p_{\text{N}}(\textbf{x}, t)~,~\partial\Omega_{\text{N}} & \text{pressure natural boundary condition} \\
\phi(\textbf{x}) & \text{porosity}\\
\mathsf{K}(\phi) & \text{permeability}\\
\mathsf{D}(\phi, \textbf{u}) & \text{solutal dispersion}\\
\mathsf{G}(\phi, \textbf{u}) & \text{thermal dispersion}\\
\rho(c, \theta) & \text{density}\\
\mu(c, \theta) & \text{viscosity}\\
\Sigma(c,\theta, \phi) & \text{solutal reaction}\\
H(c,\theta, \phi) & \text{thermal reaction}\\
\end{cases}
\end{align*}
$$

## Non-dimensionalization

### Scalings

| Quantity | $\vert\textbf{x}\vert$ | $\vert\textbf{u}\vert$ | $t$ | $c$ | $\theta$ | $\rho g$ | $p$ | $\psi$ |
| -------- | ------- | ------- | ------- | ------- | ------- |  ------- |  ------- |  ------- | 
| **Scaling** | $\mathcal{L}$ | $\mathcal{U}$ |$\mathcal{T}$ | $\Delta c$ | $\Delta\theta$ | $g \Delta\rho$ | $\mu_{\text{ref}}\,\mathcal{U}\mathcal{L}/K_{\text{ref}}$ | $\mathcal{U}\mathcal{L}$ |

| $\mu$ | $\phi$ | $K$ | $\vert\mathsf{D}\vert$ | $\vert\mathsf{G}\vert$ | $\Sigma$ | $H$ |
| ------- | ------- | ------- | ------- | ------- |  ------- | ------- |  
| $\mu_{\text{ref}}$ | $\phi_{\text{ref}}$ |$K_{\text{ref}}$ | $D_{\text{ref}}$ | $G_{\text{ref}}$ | $\Delta \Sigma$ | $\Delta H$ |


### Abstract dimensionless numbers

$$
Ad=\frac{\mathcal{U}\mathcal{T}}{\phi_{\text{ref}}\mathcal{L}}~,~
Di=\frac{D_{\text{ref}}\mathcal{T}}{\phi_{\text{ref}}\mathcal{L}^2}~,~
Ki=\frac{\mathcal{T}\Delta \Sigma}{\phi_{\text{ref}}\Delta c}~,~
Bu=\frac{K_{\text{ref}}\,g\Delta\rho}{\mu_{\text{ref}}\,\mathcal{U}}~,~
X=\frac{\mathcal{L}_\Omega}{\mathcal{L}}
$$

### Physical dimensionless numbers

| Definition | Name | Physical interpretation | 
| -------- | ------- | ------- |
| $Ra=\frac{\mathcal{L}_\Omega K_{\text{ref}}g\Delta\rho}{\mu_{\text{ref}}D_{\text{ref}}}=\underbrace{\frac{K_{\text{ref}}\,g\Delta\rho}{\mu_{\text{ref}}}}_{\text{convective speed}} \big/ \underbrace{\frac{D_{\text{ref}}}{\mathcal{L}_\Omega}}_{\text{diffusive speed}}$  |  Rayleigh  | Ratio of convective to diffusive speeds, defined with respect to solutal transport and domain length scale. |
| $Da=\frac{\mathcal{L}_\Omega \mu_{\text{ref}}\,\Delta \Sigma}{K_{\text{ref}}\,g\Delta\rho\Delta c} = \underbrace{\frac{\Delta \Sigma}{\Delta c}}_{\text{reaction rate}} \big/ \underbrace{\frac{K_{\text{ref}}\,g\Delta\rho}{\mathcal{L}_\Omega \mu_{\text{ref}}}}_{\text{convection rate}}$  |  Damköhler  | Ratio of reaction to convection rates, defined with respect to solutal transport and domain length scale. |
| $Le=\frac{G_{\text{ref}}}{D_{\text{ref}}}$  |  Lewis  | Ratio of thermal to solutal diffusivities. |
| $Lr=\frac{\Delta H\Delta c}{\Delta\theta \Delta \Sigma} = \underbrace{\frac{\Delta H}{\Delta\theta}}_{\text{thermal reaction rate}} \big/ \underbrace{\frac{\Delta \Sigma}{\Delta c}}_{\text{solutal reaction rate}}$  |  reaction Lewis  | Ratio of thermal to solutal reaction rates. |

### Scaling choice

| Name | $\mathcal{L}$ | $\mathcal{U}$ | $\mathcal{T}$ | $Ad$ | $Di$ | $Ki$ | $Bu$ | $X$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| advective | $\mathcal{L}_\Omega$ | $K_{\text{ref}}g\Delta\rho/\mu_{\text{ref}}$ | $\phi_{\text{ref}}\mathcal{L}/\mathcal{U}$ | $1$ | $1/Ra$ | $Da$ | $1$ | 1 |
| diffusive | $\mathcal{L}_\Omega$ | $D_{\text{ref}}/\mathcal{L}$ | $\phi_{\text{ref}}\mathcal{L}/\mathcal{U}$ | $1$ | $1$ | $RaDa$ | $Ra$ | 1 |
| advective-diffusive | $D_{\text{ref}}/\mathcal{U}$ | $K_{\text{ref}}g\Delta\rho/\mu_{\text{ref}}$ | $\phi_{\text{ref}}\mathcal{L}/\mathcal{U}$ | $1$ | $1$ | $Da/Ra$ | $1$ | $Ra$ |
| reactive | $\sqrt{D_{\text{ref}}\mathcal{T}/\phi_{\text{ref}}}$ | $\phi_{\text{ref}}\mathcal{L}/\mathcal{T}$ | $\phi_{\text{ref}}\Delta c/\Delta\Sigma$ | $1$ | $1$ | $1$ | $\sqrt{Ra/Da}$ | $\sqrt{RaDa}$ |

## Non-dimensional time-discretized equations

### Strong form

$$
\begin{align*}
&\text{Find}~c^{n+1}, \theta^{n+1},~\textbf{u}^n,~p^n~\text{such that}~\forall n\geq0 \\
&\begin{cases}
\phi\frac{c^{n+1}-c^n}{\Delta t^n}+Ad\,\mathcal{D}_{\textbf{u},c}(\textbf{u}\cdot\nabla c)=Di\nabla\cdot\mathcal{D}_{\mathsf{D},c}(\mathsf{D}\cdot\nabla c) + Ki\,\mathcal{D}_\Sigma(\Sigma) \\
\phi\frac{\theta^{n+1}-\theta^n}{\Delta t^n}+Ad\,\mathcal{D}_{\textbf{u},\theta}(\textbf{u}\cdot\nabla\theta)=LeDi\nabla\cdot\mathcal{D}_{\mathsf{G},\theta}(\mathsf{G}\cdot\nabla\theta) + LrKi\,\mathcal{D}_H(H) \\
\nabla\cdot\textbf{u}^n=0 \\
\textbf{u}^n=-\frac{\mathsf{K}}{\mu^n}\cdot\left(\nabla p^n - Bu\,\rho^n\,\textbf{e}_g\right) \\
c^0=c_0 \\
\theta^0=\theta_0  \\
c^n\vert_{\partial\Omega_{\text{D}, c}}=c^n_{\text{D}} \\
\left(\textbf{n}\cdot(\mathsf{D}^n\cdot\nabla c^n)\right)\vert_{\partial\Omega_{\text{N}, c}} = c_{\text{N}}^n \\
\theta^n\vert_{\partial\Omega_{\text{D}, \theta}}=\theta^n_{\text{D}} \\
\left(\textbf{n}\cdot(\mathsf{G}^n\cdot\nabla\theta^n)\right)\vert_{\partial\Omega_{\text{N}, \theta}} = \theta_{\text{N}}^n \\
(\textbf{n}\cdot\textbf{u}^n)\vert_{\partial\Omega_{\text{E}}} = u^n_{\text{E}}\\
p^n\vert_{\partial\Omega_{\text{N}}} = p^n_{\text{N}} \\
\end{cases}
\end{align*}
$$

### Weak forms

#### Velocity-pressure formulation

$$
\begin{align*}
&\text{Find} \\
&(\textbf{u}^n, p^n)\in V_\textbf{u}\times V_p \\
&c^{n+1}\in V_c \\
&\theta^{n+1}\in V_\theta \\
&\text{such that} \\
&\mathbb{F}_{\textbf{u},p,c,\theta} 
\begin{cases}
F_{\textbf{u},p}(\textbf{u}^n, p^n, \textbf{v}, q)=0 \quad\forall(\textbf{v}, q)\in V_{\textbf{u}} \times V_p\\
F_c(c^{n+1}, v) = 0\quad\forall v\in V_c \\ 
F_\theta(\theta^{n+1}, v) = 0\quad\forall v\in V_\theta \\ 
\end{cases}
\end{align*}
$$

#### Streamfunction formulation

$$
\begin{align*}
&\text{Find} \\
&\psi^n\in V_\psi \\
&\textbf{u}^n\in V_\textbf{u} \\
&c^{n+1}\in V_c \\
&\theta^{n+1}\in V_\theta \\
&\text{such that} \\
&\mathbb{F}_{\psi,c,\theta} 
\begin{cases}
F_\psi(\psi^n, v) = 0\quad\forall v\in V_\psi \\
F_{\textbf{u}}(\textbf{u}^n, \textbf{v}) = 0\quad\forall \textbf{v}\in V_{\textbf{u}} \\ 
F_c(c^{n+1}, v) = 0\quad\forall v\in V_c \\ 
F_\theta(\theta^{n+1}, v) = 0\quad\forall v\in V_\theta \\ 
\end{cases}
\end{align*}
$$