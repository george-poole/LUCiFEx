# Navier-Stokes equations

## Velocity-pressure formulation

### Nonlinear strong form

$$
\begin{align*}
&\text{Find}~\textbf{u}(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R}^d~\text{and}~p(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R}~\text{such that} \\
&\mathbb{IBVP}_{\textbf{u},p}\begin{cases}
\nabla\cdot\textbf{u} = 0 & \\
\rho \left(\frac{\partial\textbf{u}}{\partial t}+\textbf{u}\cdot\nabla\textbf{u}\right)=-\nabla p + \nabla\cdot\tau + \textbf{f} & \forall(\textbf{x}, t)\in\Omega\times[0,\infty) \\
\textbf{u}=\textbf{u}_0 & \forall(\textbf{x},t)\in\Omega\times\{0\} \\
p=p_0 & \forall(\textbf{x},t)\in\Omega\times\{0\} \\
\textbf{u} = \textbf{u}_{\text{E}} & \forall \textbf{x}\in\partial\Omega_{\text{E}} \times[0, \infty) \\
(-p\mathsf{I}+\tau)\cdot\textbf{n} = \boldsymbol{\tau}_{\text{N}} & \forall(\textbf{x},t)\in\partial\Omega_{\text{N}}\times[0, \infty)~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{E}}
\end{cases} \\
&\text{given} \\
&\mathbb{S}_{\textbf{u},p}\begin{cases}
\Omega & \text{domain}\\
\textbf{u}_0(\textbf{x}) & \text{velocity initial condition}\\
p_0(\textbf{x}) & \text{pressure initial condition}\\ 
\textbf{u}_{\text{E}}(\textbf{x}, t)~,~\partial\Omega_{\text{E}} & \text{velocity essential boundary condition} \\
\boldsymbol{\tau}_{\text{N}}(\textbf{x}, t)~,~\partial\Omega_{\text{N}} & \text{traction natural boundary condition} \\
\tau(\textbf{u}) & \text{deviatoric stress constitutive relation} \\
\textbf{f}(\textbf{x}, t) & \text{body force} \\
\rho(\textbf{x}, t) & \text{density}
\end{cases}
\end{align*}
$$

### Linearized weak forms

#### Incremental pressure correction scheme

$$
\begin{aligned}
&\text{Find} \\
&\widetilde{\textbf{u}}^{n+1}\in V_{\textbf{u}}, \\
&p^{n+1}\in V_p, \\
&\textbf{u}^{n+1}\in V_{\textbf{u}} \\
&\text{such that} \\
&\mathbb{F} 
\begin{cases}
\begin{align*}
F_1(\widetilde{\textbf{u}}^{n+1}, \textbf{v}) &= 
\int_\Omega\text{d}\Omega~\mathcal{D}_\rho(\rho)\textbf{v}\cdot\frac{\widetilde{\textbf{u}}^{n+1}-\textbf{u}^n}{\Delta t^n} + \mathcal{D}_\rho(\rho)\textbf{v}\cdot\mathcal{D}_{\textbf{u}}(\textbf{u}\cdot\nabla\textbf{u}) \\
&\qquad\quad +\varepsilon(\textbf{v}):(-p^n\mathsf{I} + \mathcal{D}_\tau(\boldsymbol{\tau})) -\textbf{v}\cdot\mathcal{D}_{\textbf{f}}(\textbf{f}) \\ 
&\quad -\int_{\partial\Omega}\text{d}\Gamma~\textbf{v}\cdot(-p^n\mathsf{I} + \mathcal{D}_\tau(\boldsymbol{\tau}))\cdot\textbf{n} \\
&=0 \quad\forall \textbf{v} \in V_{\textbf{u}} 
\end{align*} \\
F_2(p^{n+1}, q) = \int_\Omega\text{d}\Omega~\nabla q\cdot\nabla p^{n+1} - \nabla q\cdot\nabla p^n + q\rho\frac{\nabla\cdot\widetilde{\textbf{u}}^{n+1}}{\Delta t^n} =0 \quad\forall q\in V_p \\
F_3(\textbf{u}^{n+1}, \textbf{v}) = \int_\Omega\text{d}\Omega~\textbf{v}\cdot\rho\frac{\textbf{u}^{n+1}-\widetilde{\textbf{u}}^{n+1}}{\Delta t^n} + \textbf{v}\cdot\nabla p^{n+1} - \textbf{v}\cdot\nabla p^n = 0 \quad\forall \textbf{v} \in V_{\textbf{u}}
\end{cases}
\end{aligned}
$$


#### Chorin's scheme

$$
\begin{aligned}
&\text{Find} \\
&\widetilde{\textbf{u}}^{n+1}\in V_{\textbf{u}}, \\
&p^{n+1}\in V_p, \\
&\textbf{u}^{n+1}\in V_{\textbf{u}} \\
&\text{such that} \\
&\mathbb{F} 
\begin{cases}
\begin{align*}
F_1(\widetilde{\textbf{u}}^{n+1}, \textbf{v}) &= 
\int_\Omega\text{d}\Omega~\mathcal{D}_\rho(\rho)\textbf{v}\cdot\frac{\widetilde{\textbf{u}}^{n+1}-\textbf{u}^n}{\Delta t^n} + \mathcal{D}_\rho(\rho)\textbf{v}\cdot\mathcal{D}_{\textbf{u}}(\textbf{u}\cdot\nabla\textbf{u}) \\
&\qquad\quad + \nabla\textbf{v}\cdot\nabla\mathcal{D}_{\tau}(\boldsymbol{\tau}) -\textbf{v}\cdot\mathcal{D}_{\textbf{f}}(\textbf{f}) \\
&=0 \quad\forall \textbf{v} \in V_{\textbf{u}} 
\end{align*} \\
F_2(p^{n+1}, q) = \int_\Omega\text{d}\Omega~\nabla q\cdot\nabla p^{n+1} + q\rho\frac{\nabla\cdot\widetilde{\textbf{u}}^{n+1}}{\Delta t^n}=0 \quad\forall q\in V_p \\
F_3(\textbf{u}^{n+1}, \textbf{v}) = \int_\Omega\text{d}\Omega~\textbf{v}\cdot\rho\frac{\textbf{u}^{n+1}-\widetilde{\textbf{u}}^{n+1}}{\Delta t^n} + \textbf{v}\cdot\nabla p^{n+1} =0 \quad\forall \textbf{v} \in V_{\textbf{u}}
\end{cases}
\end{aligned}
$$

## Streamfunction-vorticity formulation

### Strong form
$$
\begin{align*}
&\text{Find}~\psi(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R}~\text{and}~\omega(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R}~\text{such that} \\
&\mathbb{IBVP}_{\psi,\omega}\begin{cases}
\nabla^2\psi =\omega & \\
\rho\left(\frac{\partial\omega}{\partial t}+\left(-\frac{\partial\psi}{\partial y}, \frac{\partial\psi}{\partial x}\right)\cdot\nabla\omega\right) =\mu\nabla^2\omega + \frac{\partial f_y}{\partial x} - \frac{\partial f_x}{\partial y} & \forall(\textbf{x}, t)\in\Omega\times[0,\infty) \\
\omega=\omega_0 & \forall(\textbf{x},t)\in\Omega\times\{0\} \\
\psi=\psi_{\text{D}} & \forall \textbf{x}\in\partial\Omega_{\text{D}, \psi} \times [0,\infty) \\
\textbf{n}\cdot\nabla\psi = \psi_{\text{N}} & \forall\textbf{x}\in\partial\Omega_{\text{N}, \psi}
\times [0,\infty)~,~\partial\Omega_{\text{N}, \psi}=\partial\Omega/\partial\Omega_{\text{D}, \psi} \\
\omega=\omega_{\text{D}} & \forall \textbf{x}\in\partial\Omega_{\text{D},\omega} \times [0,\infty) \\
\textbf{n}\cdot\nabla\omega = \omega_{\text{N}} & \forall\textbf{x}\in\partial\Omega_{\text{N},\omega}\times[0,\infty)~,~\partial\Omega_{\text{N},\omega}=\partial\Omega/\partial\Omega_{\text{D},\omega} 
\end{cases} \\
&\text{given}\\
&\mathbb{S}_{\psi,\omega}\begin{cases}
\Omega\subset\mathbb{R}^2  & \text{domain}\\
\psi_{\text{D}}(\textbf{x})~,~\partial\Omega_{\text{D}} & \text{Dirichlet boundary condition} \\
\psi_{\text{N}}(\textbf{x})~,~\partial\Omega_{\text{N}} & \text{Neumann boundary condition} \\
f_x(\textbf{x}), f_y(\textbf{x}) & \text{body force} \\
\mu & \text{viscosity} \\
\rho & \text{density} \\
\end{cases}\\
&\text{where}\\
&\textbf{u}=\nabla\times\boldsymbol{\psi}=\textbf{u}=\nabla\times\psi\textbf{e}_z=\frac{\partial\psi}{\partial y}\textbf{e}_x - \frac{\partial\psi}{\partial x}\textbf{e}_y \iff \nabla\cdot\textbf{u}=0\\
&\boldsymbol{\omega}=\nabla\times\textbf{u}=\nabla\times(\nabla\times\boldsymbol{\psi})=\omega\textbf{e}_z=\left(\frac{\partial u_y}{\partial x} - \frac{\partial u_x}{\partial y}\right)\textbf{e}_z\\
&\textbf{f}=f_x\textbf{e}_x + f_y\textbf{e}_y \\
&\tau(\textbf{u}) = \tfrac{\mu}{2}\left(\nabla\textbf{u} + \nabla\textbf{u}^{\mathsf{T}}\right) \\
&\nabla\mu=\nabla\rho=\textbf{0} \\
\end{align*}
$$