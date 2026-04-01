# Darcy equations

## Velocity-pressure formulation

### Strong form

$$
\begin{align*}
&\text{Find}~\textbf{u}(\textbf{x}): \Omega \to \mathbb{R}^d~\text{and}~p(\textbf{x}): \Omega \to \mathbb{R}~\text{such that} \\
&\mathbb{BVP}_{\textbf{u},p}\begin{cases}
\nabla\cdot\textbf{u} = 0 & \\
\textbf{u} = -\frac{\mathsf{K}}{\mu}\cdot(\nabla p - \textbf{f}\,) & \forall\textbf{x}\in\Omega \\
\textbf{n}\cdot\textbf{u} = u_{\text{E}} & \forall \textbf{x}\in\partial\Omega_{\text{E}} \\
p = p_{\text{N}} & \forall\textbf{x}\in\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{E}} \\
\int_{\partial\Omega}\text{d}\Gamma~u_{\text{E}}=0 & \text{if}~\partial\Omega_{\text{N}}=\varnothing\iff\partial\Omega_{\text{E}}=\partial\Omega \\
\end{cases} \\
&\text{given} \\
&\mathbb{S}_{\textbf{u},p}\begin{cases}
\Omega\subset\mathbb{R}^d  & \text{domain}\\
u_{\text{E}}(\textbf{x})~,~\partial\Omega_{\text{E}} & \text{normal velocity essential boundary condition} \\
p_{\text{N}}(\textbf{x})~,~\partial\Omega_{\text{N}} & \text{pressure natural boundary condition} \\
\mathsf{K}(\textbf{x}) & \text{permeability}\\
\mu(\textbf{x}) & \text{viscosity} \\
\end{cases}
\end{align*}
$$

### Weak form 

$$
\begin{aligned}
&\text{Find}~(\textbf{u}, p)\in V_{\textbf{u}} \times V_p~\text{such that} \\
&\begin{align*}
F(\textbf{u}, p, \textbf{v}, q) &= \int_\Omega\text{d}\Omega~q(\nabla\cdot\textbf{u}) + \textbf{v}\cdot(\mu\mathsf{K}^{-1}\cdot\textbf{u}) - p(\nabla\cdot\textbf{v}) - \textbf{v}\cdot\textbf{f} \\
&\quad +\int_{\partial\Omega_{\text{N}}}\text{d}\Gamma~p_{\text{N}}\,\textbf{v}\cdot\textbf{n} \\
&=0 \quad\forall(\textbf{v}, q)\in V_{\textbf{u}} \times V_p~.
\end{align*}
\end{aligned}
$$

## Streamfunction formulation

### Strong form

$$
\begin{align*}
&\text{Find}~\psi(\textbf{x}): \Omega \to \mathbb{R}~\text{such that} \\
&\mathbb{BVP}_\psi\begin{cases}
\nabla\cdot\left(\frac{\mu\mathsf{K}^{\mathsf{T}}\cdot\nabla\psi}{\text{det}(\mathsf{K})}\right)=-\frac{\partial(f_y)}{\partial x} + \frac{\partial(f_x)}{\partial y} & \forall\textbf{x}\in\Omega \\
\psi=\psi_{\text{D}} & \forall \textbf{x}\in\partial\Omega_{\text{D}} \\
\textbf{n}\cdot\left(\frac{\mu\mathsf{K}^{\mathsf{T}}\cdot\nabla\psi}{\text{det}(\mathsf{K})}\right) = \psi_{\text{N}} & \forall\textbf{x}\in\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}}
\end{cases} \\
&\text{given} \\
&\mathbb{S}_{\psi}\begin{cases}
\Omega\subset\mathbb{R}^2  & \text{domain}\\
\psi_{\text{D}}(\textbf{x})~,~\partial\Omega_{\text{D}} & \text{Dirichlet boundary condition} \\
\psi_{\text{N}}(\textbf{x})~,~\partial\Omega_{\text{N}} & \text{Neumann boundary condition} \\
\mathsf{K}(\textbf{x}) & \text{permeability}\\
\mu(\textbf{x}) & \text{viscosity} \\
f_x(\textbf{x}), f_y(\textbf{x}) & \text{body force} \\
\end{cases}\\
&\text{where}\\
&\textbf{u}=\nabla\times\boldsymbol{\psi}=\textbf{u}=\nabla\times\psi\textbf{e}_z=\frac{\partial\psi}{\partial y}\textbf{e}_x - \frac{\partial\psi}{\partial x}\textbf{e}_y \iff \nabla\cdot\textbf{u}=0\\
&\textbf{f}=f_x\textbf{e}_x + f_y\textbf{e}_y \\
\end{align*}
$$

### Weak form

Equivalent to [Poisson equation](../P2_introductory/I10_poisson_equation.md).

## Pressure formulation

### Strong form

$$
\begin{align*}
&\text{Find}~p(\textbf{x}): \Omega \to \mathbb{R}~\text{such that} \\
&\mathbb{BVP}_p\begin{cases}
\nabla\cdot\left(\frac{\mathsf{K}}{\mu}\cdot\nabla p\right)=\nabla\cdot\left(\frac{\mathsf{K}}{\mu}\cdot\textbf{f}\right) & \forall\textbf{x}\in\Omega \\
p=p_{\text{D}} & \forall \textbf{x}\in\partial\Omega_{\text{D}} \\
\textbf{n}\cdot\left(\frac{\mathsf{K}}{\mu}\cdot\nabla p\right) = p_{\text{N}} & \forall\textbf{x}\in\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}}
\end{cases} \\
&\text{given} \\
&\mathbb{S}_{p}\begin{cases}
\Omega\subset\mathbb{R}^d  & \text{domain}\\
p_{\text{D}}(\textbf{x})~,~\partial\Omega_{\text{D}} & \text{Dirichlet boundary condition} \\
p_{\text{N}}(\textbf{x})~,~\partial\Omega_{\text{N}} & \text{Neumann boundary condition} \\
\mathsf{K}(\textbf{x}) & \text{permeability}\\
\mu(\textbf{x}) & \text{viscosity} \\
\textbf{f}(\textbf{x}) & \text{body force} \\
\end{cases}
\end{align*}
$$

### Weak form

Equivalent to [Poisson equation](../P2_introductory/I10_poisson_equation.md).