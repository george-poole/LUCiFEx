# Advection-diffusion equation

## Strong form

$$
\begin{align*}
&\text{Find}~u(\textbf{x}, t): \Omega\times[0,\infty) \to \mathbb{R}~\text{such that } \\
&\mathbb{IBVP}\begin{cases}
\frac{\partial u}{\partial t} + \textbf{a}\cdot\nabla u= \nabla\cdot(\mathsf{D}\cdot\nabla u) & \forall(\textbf{x}, t)\in\Omega\times[0,\infty) \\
u=u_0 & \forall(\textbf{x},t)\in\Omega\times\{0\}\\
u=u_{\text{D}} & \forall(\textbf{x},t)\in\partial\Omega_{\text{D}}\times[0,\infty) \\
\textbf{n}\cdot(\mathsf{D}\cdot\nabla{u}) = u_{\text{N}} & \forall(\textbf{x},t)\in\partial\Omega_{\text{N}}\times[0,\infty)~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}}
\end{cases}~.
\end{align*}
$$

## Time discretization

$$\frac{u^{n+1} - u^n}{\Delta t^n} + \mathcal{D}_{{\textbf{a}},u}(\textbf{a}\cdot\nabla u) = \nabla\cdot(\mathcal{D}_{{\mathsf{D}},u}(\mathsf{D}\cdot\nabla u))$$

## Weak form

$$
\begin{aligned}
&\text{Find}~u^{n+1}\in V~\text{such that } \\
&\begin{align*}
F(u^{n+1},v)&=\int_\Omega\text{d}\Omega~v\frac{u^{n+1} - u^n}{\Delta t^n} + v\mathcal{D}_{{\textbf{a}},u}(\textbf{a}\cdot\nabla u) \\
&\qquad\quad + \nabla v\cdot\mathcal{D}_{{\mathsf{D}},u}(\mathsf{D}\cdot\nabla u) \\
&\quad - \int_{\partial\Omega_{\text{N}}}\text{d}\Gamma~vu_{\text{N}}=0 \quad\forall v\in V~.
\end{align*}
\end{aligned}
$$

## Specification

$$\mathbb{S}\begin{cases}
\Omega \\
u_0(\textbf{x}) \\
u_{\text{D}}(\textbf{x}, t)~,~\partial\Omega_{\text{D}} \\
u_{\text{N}}(\textbf{x}, t)~,~\partial\Omega_{\text{N}} \\
\textbf{a}(\textbf{x}, t) \\
\mathsf{D}(\textbf{x}, t) \\
\end{cases}
$$