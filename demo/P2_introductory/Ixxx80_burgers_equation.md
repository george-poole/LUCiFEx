# Burgers' equation

## Strong form

$$
\begin{align*}
&\text{Find}~u(\textbf{x}, t): \Omega\times[0,\infty) \to \mathbb{R}~\text{such that } \\
&\mathbb{IBVP}\begin{cases}
\frac{\partial u}{\partial t} + au\frac{\partial u}{\partial x}= \frac{\partial}{\partial x}\left(D\frac{\partial u}{\partial x}\right) & \forall(\textbf{x}, t)\in\Omega\times[0,\infty) \\
u=u_0 & \forall(\textbf{x},t)\in\Omega\times\{0\}\\
u=u_{\text{D}} & \forall(\textbf{x},t)\in\partial\Omega_{\text{D}}\times[0,\infty) \\
\textbf{n}\cdot(\mathsf{D}\cdot\nabla{u}) = u_{\text{N}} & \forall(\textbf{x},t)\in\partial\Omega_{\text{N}}\times[0,\infty)~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}}
\end{cases}~.
\end{align*}
$$

$$\Omega\subset\mathbb{R}$$