# Darcy-Brinkman convection equations

Governing equations for convection coupled to Darcy-Brinkman flow, working in the Boussinesq approximation with constant viscosity and Newtonian deviatoric stress.

## Dimensional equations

$$
\begin{align*}
&\text{Find} \\
&c(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R}, \\
&\textbf{u}(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R}^d, \\
&p(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R} \\
&\text{such that} \\
&\mathbb{IBVP}_{\textbf{u},p,c}\begin{cases}
\phi\frac{\partial c}{\partial t}+\textbf{u}\cdot\nabla c=\nabla\cdot(\mathsf{D}(\phi, \textbf{u})\cdot\nabla c) & \\
\nabla\cdot\textbf{u} = 0 & \\
\rho_{\text{ref}}\left(\frac{\partial\textbf{u}}{\partial t}+\textbf{u}\cdot\nabla(\phi^{-1}\textbf{u})\right)=-\phi\nabla p + \nabla\cdot\tau + \phi\rho g\,\textbf{e}_g - \mu\phi\mathsf{K}^{-1}\cdot\textbf{u} \\
c=c_0 & \forall(\textbf{x}, t)\in\Omega\times\{0\} \\
\textbf{u}=\textbf{u}_0 & \forall(\textbf{x}, t)\in\Omega\times\{0\} \\
p=p_0 & \forall(\textbf{x}, t)\in\Omega\times\{0\} \\
c=c_{\text{D}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{D}} \times [0,\infty) \\
\textbf{n}\cdot(\mathsf{D}\cdot\nabla c) = c_{\text{N}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{N}}
\times [0,\infty)~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}} \\
\textbf{u} = \textbf{u}_{\text{E}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{E}} \times [0,\infty) \\
(-p\mathsf{I}+\tau)\cdot\textbf{n} = \boldsymbol{\tau}_{\text{N}} & \forall(\textbf{x},t)\in\partial\Omega_{\text{N}}\times[0, \infty)~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{E}}
\end{cases} \\
&\text{given} \\
&\mathbb{S}_{\textbf{u},p,c}=
\begin{cases}
\Omega\subset\mathbb{R}^d & \text{domain}\\
\mu_{\text{ref}} & \text{constant viscosity}
\end{cases} 
\end{align*}
$$


## Non-dimensionalization

### Scalings

| Quantity | $\vert\textbf{x}\vert$ | $\vert\textbf{u}\vert$ | $t$ | $c$ | $\rho g$ | $p$ | $\psi$ |
| -------- | ------- | ------- | ------- | ------- | ------- |  ------- |  ------- |  
| **Scaling** | $\mathcal{L}$ | $\mathcal{U}$ |$\mathcal{T}$ | $\Delta c$ | $g \Delta\rho$ | $\mu_{\text{ref}}\,\mathcal{U}\mathcal{L}/K_{\text{ref}}$ | $\mathcal{U}\mathcal{L}$ |

| $\mu$ | $\phi$ | $K$ | $\vert\mathsf{D}\vert$ | $\vert\mathsf{G}\vert$ | $R$ | $Q$ |
| ------- | ------- | ------- | ------- | ------- |  ------- | ------- |  
| $\mu_{\text{ref}}$ | $\phi_{\text{ref}}$ |$K_{\text{ref}}$ | $D_{\text{ref}}$ | $G_{\text{ref}}$ | $\Delta R$ | $\Delta Q$ |


### Generic dimensionless numbers

$$
Ad=\frac{\mathcal{U}\mathcal{T}}{\mathcal{L}} ~,~
Di=\frac{D_{\text{ref}}\mathcal{T}}{\mathcal{L}^2} ~,~
Vi=\frac{\mu_{\text{ref}}\mathcal{T}}{\rho_{\text{ref}}\mathcal{L}^2} ~,~
Bu=\frac{\,g\Delta\rho\,\mathcal{T}}{\rho_{\text{ref}}\,\mathcal{U}} ~,~
Pm=\frac{\mu_{\text{ref}}\,\,\mathcal{T}}{K_{\text{ref}}\,\rho_{\text{ref}}\,\mathcal{U}} ~,~
X=\frac{\mathcal{L}_\Omega}{\mathcal{L}}
$$

### Physical dimensionless numbers

| Definition | Name | Physical interpretation | 
| -------- | ------- | ------- |
| $Pr=\frac{\mu_{\text{ref}}}{\rho_{\text{ref}}D_{\text{ref}}}$ | Prandtl | Ratio of kinematic viscosity to diffusivity, defined with respect to solutal transport |
| $Ra=\frac{\mathcal{L}_\Omega^3g\Delta\rho}{\mu_{\text{ref}}D_{\text{ref}}}$  |  Rayleigh  | Ratio of convective to diffusive speeds, defined with respect to solutal transport and domain length scale. |
| $Dr = \frac{K_{\text{ref}}}{\mathcal{L}_\Omega^2}$  |  Darcy  | Ratio of permeability to squared length scale.  |
| $Le=\frac{G_{\text{ref}}}{D_{\text{ref}}}$  |  Lewis  | Ratio of thermal to solutal diffusivities. |

### Scaling choice

| Name | $\mathcal{L}$ | $\mathcal{U}$ | $\mathcal{T}$ | $Ad$ | $Di$ | $Vi$ | $Bu$ | $Pm$ | $X$ |
|---|---|---|---|---|---|---|---|---|---|
| advective | $\mathcal{L}_\Omega$ | $\sqrt{\mathcal{L}g\Delta\rho/\rho_{\text{ref}}}$ | $\mathcal{L}/\mathcal{U}$ | $1$ | $1/\sqrt{Ra\,Pr}$ | $\sqrt{Pr/Ra}$ | $1$ | $\sqrt{Pr/(Ra\,Dr^2)}$ | $1$ |

## Non-dimensional time-discretized equations

### Strong form

$$
\begin{align*}
&\text{Find}~c^{n+1},~\textbf{u}^{n+1},~p^{n+1}~\text{such that}~\forall n\geq0 \\
&\begin{cases}
\frac{c^{n+1}-c^n}{\Delta t^n}+Ad\,\mathcal{D}_{\textbf{u},c}(\textbf{u}\cdot\nabla c)=Di\nabla\cdot\mathcal{D}_{\mathsf{D},c}(\mathsf{D}\cdot\nabla c) \\
\nabla\cdot\textbf{u}^{n+1}=0 \\
\frac{\textbf{u}^{n+1}-\textbf{u}^n}{\Delta t^n}+Ad\,\mathcal{D}_{\textbf{u}}(\textbf{u}\cdot\nabla(\phi^{-1}\textbf{u})) = -\phi\mathcal{D}_{p}(\nabla p) + Vi\,\nabla^2\mathcal{D}_{\textbf{u}}(\textbf{u}) + Bu\,\mu\phi\mathcal{D}_{\rho}(\rho\,\textbf{e}_g) - Pm\,\phi\mathsf{K}^{-1}\cdot\mathcal{D}_{u}(\textbf{u})\\
\vdots \\
\end{cases}
\end{align*}
$$