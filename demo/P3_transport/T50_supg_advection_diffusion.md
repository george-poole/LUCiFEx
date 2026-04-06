# SUPG stabilization of the advection-diffusion-reaction equation

## Strong form

$$
\begin{align*}
&\text{Find}~u(\textbf{x}, t): \Omega\times[0,\infty) \to \mathbb{R}~\text{such that } \\
&\mathbb{IBVP}_u\begin{cases}
\frac{\partial u}{\partial t} + \textbf{a}\cdot\nabla u= \nabla\cdot(\mathsf{D}\cdot\nabla u) + Ru + J & \forall(\textbf{x}, t)\in\Omega\times[0,\infty) \\
u=u_0 & \forall(\textbf{x},t)\in\Omega\times\{0\}\\
u=u_{\text{D}} & \forall(\textbf{x},t)\in\partial\Omega_{\text{D}}\times[0,\infty) \\
\textbf{n}\cdot(\mathsf{D}\cdot\nabla{u})  = u_{\text{N}} & \forall(\textbf{x},t)\in\partial\Omega_{\text{N}}\times[0,\infty)~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}}
\end{cases} \\
&\text{given} \\
&\mathbb{S}_u\begin{cases}
\Omega\subset\mathbb{R}^d & \text{domain} \\
u_0(\textbf{x}) & \text{initial condition} \\
u_{\text{D}}(\textbf{x}, t)~,~\partial\Omega_{\text{D}} & \text{Dirichlet boundary condition}\\
u_{\text{N}}(\textbf{x}, t)~,~\partial\Omega_{\text{N}} & \text{Neumann boundary condition}\\
\textbf{a}(\textbf{x}, t) & \text{velocity} \\
\mathsf{D}(\textbf{x}, t) & \text{dispersion} \\
R(\textbf{x}, t) & \text{reaction rate} \\
J(\textbf{x}, t) & \text{reaction source} \\
\end{cases}
\end{align*}
$$