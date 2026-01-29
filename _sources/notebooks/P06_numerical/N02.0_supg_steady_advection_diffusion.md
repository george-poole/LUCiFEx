# SUPG stabilization of the steady advection-diffusion-reaction equation

## Strong form
$$
\begin{align*}
&\text{Find}~u(\textbf{x}): \Omega \to \mathbb{R}~\text{such that} \\
&\mathbb{IBVP}\begin{cases}
\textbf{a}\cdot\nabla u= \nabla\cdot(\mathsf{D}\cdot\nabla u) + Ru + J & \forall\textbf{x}\in\Omega \\
u=u_{\text{D}} & \forall \textbf{x}\in\partial\Omega_{\text{D}} \\
\textbf{n}\cdot(\mathsf{D}\cdot\nabla{u}) = u_{\text{N}} & \forall\textbf{x}\in\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}}
\end{cases}~.
\end{align*}
$$

## Weak form
$$
\begin{align*}
&\text{Find}~u\in V ~\text{such that} \\
&F(u,v)+F_{\text{SUPG}}(u,v)=0 \quad\forall v\in V~.
\end{align*}
$$

stabilization term
$$
F_{\text{SUPG}}(u,v)=...
$$
