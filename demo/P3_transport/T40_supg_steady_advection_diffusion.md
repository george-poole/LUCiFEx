# SUPG stabilization of the steady advection-diffusion-reaction equation

## Strong form

$$
\begin{align*}
&\text{Find}~u(\textbf{x}): \Omega \to \mathbb{R}~\text{such that} \\
&\mathbb{IBVP}_u\begin{cases}
\textbf{a}\cdot\nabla u= \nabla\cdot(\mathsf{D}\cdot\nabla u) + Ru + J & \forall\textbf{x}\in\Omega \\
u=u_{\text{D}} & \forall \textbf{x}\in\partial\Omega_{\text{D}} \\
\textbf{n}\cdot(\mathsf{D}\cdot\nabla{u}) = u_{\text{N}} & \forall\textbf{x}\in\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}}
\end{cases} \\
&\text{given} \\
&\mathbb{S}_u\begin{cases}
\Omega\subset\mathbb{R}^d & \text{domain} \\
u_{\text{D}}(\textbf{x})~,~\partial\Omega_{\text{D}} & \text{Dirichlet boundary condition}\\
u_{\text{N}}(\textbf{x})~,~\partial\Omega_{\text{N}} & \text{Neumann boundary condition}\\
\textbf{a}(\textbf{x}) & \text{velocity} \\
\mathsf{D}(\textbf{x}) & \text{dispersion} \\
R(\textbf{x}) & \text{reaction rate} \\
J(\textbf{x}) & \text{reaction source} \\
\end{cases}
\end{align*}
$$

## Weak form

$$
\begin{align*}
&\text{Find}~u\in V ~\text{such that} \\
&F(u,v)+F_{\text{SUPG}}(u,v)=0 \quad\forall v\in V \\
&\text{where}\\
&F_{\text{SUPG}}(u,v)=\tau P(v)\mathcal{R}(u)\\
&\mathcal{R}(u)= ...
\end{align*}
$$
