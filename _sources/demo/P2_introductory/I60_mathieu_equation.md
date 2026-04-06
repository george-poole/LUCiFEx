# Mathieu equation

## Strong form

$$
\begin{align*}
&\text{Find}~u(\textbf{x}): \Omega \to \mathbb{R}~\text{such that} \\
&(\mathbb{EVP} | \mathbb{BVP})_u\begin{cases}
-\nabla^2u + 2q\cos(\textbf{k}\cdot\textbf{x}) u = \lambda u & \forall\textbf{x}\in\Omega \\
u=u_{\text{D}} & \forall \textbf{x}\in\partial\Omega_{\text{D}} \\
\textbf{n}\cdot\nabla{u} = u_{\text{N}} & \forall\textbf{x}\in\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}}
\end{cases} \\
&\text{given} \\
&\mathbb{S}_u
\begin{cases}
\Omega\subset\mathbb{R}^d & \text{domain} \\
u_{\text{D}}(\textbf{x})~,~\partial\Omega_{\text{D}} & \text{Dirichlet boundary condition}\\
u_{\text{N}}(\textbf{x})~,~\partial\Omega_{\text{N}} & \text{Neumann boundary condition}\\
q(\textbf{x}) & \text{perturbation parameter}\\
\lambda\in\mathbb{C} & \text{eigenvalue parameter}\\
\end{cases}
\end{align*}
$$

## Weak form

$$
\begin{aligned}
&\partial\Omega_{\text{N}}=\varnothing \implies 
\begin{array}{ll}
\text{Find}~u\in V~\text{and}~\lambda\in\mathbb{C}~\text{such that}\\
a(u, v)=\lambda b(u,v) \quad\forall v\in V \\
\text{where }\\
a(u, v) = \int_\Omega\text{d}\Omega~\nabla v\cdot\nabla u + 2q\cos(\textbf{k}\cdot\textbf{x}) vu\\
b(u,v) = \int_\Omega\text{d}\Omega~vu \\
\end{array}
\\
\\
&\partial\Omega_{\text{N}}\neq\varnothing \implies 
\begin{array}{ll}
\text{Find}~u\in V~\text{such that}\\
F(u, v) = \dots \\
\forall v\in V \\
\end{array}
\end{aligned}
$$




<!-- 
$$
\begin{aligned}
\partial\Omega_{\text{N}}=\varnothing & 
&\begin{align*}
\implies &\text{Find}~u\in V~\text{and}~\lambda\in\mathbb{C}~\text{such that}\\
&a(u, v)=\lambda b(u,v) \quad\forall v\in V \\
&\text{where }\\
&a(u, v) = \int_\Omega\text{d}\Omega~\nabla v\cdot\nabla u + 2q\cos(\textbf{k}\cdot\textbf{x}) vu\\
&b(u,v) = \int_\Omega\text{d}\Omega~vu \\
\end{align*} \\
\end{aligned}
$$


$$
\begin{aligned}
& \partial\Omega_{\text{N}}\neq\varnothing \\
&\begin{align*}
\implies &\text{Find}~u\in V~\text{such that}\\
&F(u, v) = \dots
\end{align*}
\end{aligned}
$$ -->