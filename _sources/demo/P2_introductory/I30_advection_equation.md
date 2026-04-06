# Advection equation

## Strong form

$$
\begin{align*}
&\text{Find}~u(\textbf{x}, t): \Omega\times[0,\infty) \to \mathbb{R}~\text{such that} \\
&\mathbb{IBVP}_u\begin{cases}
\frac{\partial u}{\partial t}+\textbf{a}\cdot\nabla u = 0 & \forall(\textbf{x}, t)\in\Omega\times[0,\infty) \\
u=u_0 & \forall(\textbf{x},t)\in\Omega\times\{0\}\\
u=u_{\text{I}} & \forall(\textbf{x}, t)\in\Omega_{\text{I}}\times[0,\infty)~,~\partial\Omega_{I} = \{\textbf{x}\in\partial\Omega~:~\textbf{n}\cdot\textbf{a}<0\}
\end{cases} \\
&\text{given} \\
&\mathbb{S}_u\begin{cases}
\Omega\subset\mathbb{R}^d & \text{domain} \\
u_0(\textbf{x}) & \text{initial condition} \\
u_{\text{I}}(\textbf{x}, t) & \text{inflow boundary condition} \\
\textbf{a}(\textbf{x}, t) & \text{velocity} \\
\end{cases}
\end{align*}
$$

## Time-discretized weak form

$$
\begin{align*}
&\text{Find}~u^{n+1}\in V~\text{such that} \\
&F(u^{n+1}, v)=\int_\Omega~\text{d}x~v\frac{u^{n+1} - u^n}{\Delta t^n} + v\,\mathcal{D}_{{\textbf{a}}, u}(\textbf{a}\cdot\nabla u)=0 \quad\forall v\in V \\
&\text{with finite difference operator}~\mathcal{D}_{\textbf{a},u}. \\
\end{align*}
$$