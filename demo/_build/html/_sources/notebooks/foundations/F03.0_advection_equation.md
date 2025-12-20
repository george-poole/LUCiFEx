# Advection equation

## Strong form

$$
\begin{align*}
&\text{Find}~u(\textbf{x}, t): \Omega\times[0,\infty) \to \mathbb{R}~\text{such that} \\
&\mathbb{IBVP}\begin{cases}
\frac{\partial u}{\partial t}+\textbf{a}\cdot\nabla u = 0 & \forall(\textbf{x}, t)\in\Omega\times[0,\infty) \\
u=u_0 & \forall(\textbf{x},t)\in\Omega\times\{0\}\\
u=u_{\text{I}} & \forall(\textbf{x}, t)\in\Omega_{\text{I}}\times[0,\infty)~,~\partial\Omega_{I} = \{\textbf{x}\in\partial\Omega~:~\textbf{n}\cdot\textbf{a}<0\}
\end{cases}~.
\end{align*}
$$

## Weak form

$$
\begin{align*}
&\text{Find}~u^{n+1}\in V~\text{such that} \\
&F(u^{n+1}, v)=\int_\Omega~\text{d}x~v\frac{u^{n+1} - u^n}{\Delta t^n} + v\,\mathcal{D}_{{\textbf{a}}, u}(\textbf{a}\cdot\nabla u)=0 \quad\forall v\in V~.
\end{align*}
$$

## Specification

$$\mathbb{S}\begin{cases}
\Omega \\
u_0(\textbf{x}) \\
u_{\text{I}}(\textbf{x}, t) \\
\textbf{a}(\textbf{x}, t) \\ 
\end{cases}$$