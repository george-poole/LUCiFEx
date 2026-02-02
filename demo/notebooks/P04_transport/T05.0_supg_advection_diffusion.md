# SUPG stabilization of the advection-diffusion-reaction equation

## Strong form

$$
\begin{align*}
&\text{Find $u(\textbf{x}): \Omega \to \mathbb{R}$ such that } \\
&\begin{cases}
\frac{\partial u}{\partial t} + \textbf{a}\cdot\nabla u= \nabla\cdot(\mathsf{D}\cdot\nabla u) + Ru + J & \forall\textbf{x}\in\Omega \\
u=u_{\text{D}} & \forall \textbf{x}\in\partial\Omega_{\text{D}} \\
\textbf{n}\cdot(\mathsf{D}\cdot\nabla{u}) = u_{\text{N}} & \forall\textbf{x}\in\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}}
\end{cases}~.
\end{align*}
$$