# Poisson equation

## Strong form

$$
\begin{align*}
&\text{Find}~u(\textbf{x}): \Omega \to \mathbb{R}~ \text{such that} \\
&\mathbb{BVP}_u\begin{cases}
\nabla^2 u = f & \forall\textbf{x}\in\Omega \\
u=u_{\text{D}} & \forall \textbf{x}\in\partial\Omega_{\text{D}} \\
\textbf{n}\cdot\nabla{u} = u_{\text{N}} & \forall\textbf{x}\in\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}}
\end{cases} \\
&\text{given} \\
&\mathbb{S}_u
\begin{cases}
\Omega\subset\mathbb{R}^d & \text{domain} \\
u_{\text{D}}(\textbf{x})~,~\partial\Omega_{\text{D}} & \text{Dirichlet boundary condition}\\
u_{\text{N}}(\textbf{x})~,~\partial\Omega_{\text{N}} & \text{Neumann boundary condition}\\
f(\textbf{x}) & \text{forcing}\\
\end{cases}
\end{align*}
$$

## Weak form

$$
\begin{align*}
&\text{Find}~u\in V~\text{such that} \\
&F(u, v)=-\int_\Omega\text{d}\Omega~\nabla v\cdot\nabla u + vf + \int_{\partial\Omega_{\text{N}}}\text{d}\Gamma~vu_{\text{N}}=0\quad\forall v\in V~.
\end{align*}
$$

## Linear algebra

$$
\begin{align*}
&v=\xi_i~,~u=\sum_jU_j\xi_j\implies A_{ij}U_j=b_i \iff \mathsf{A}\cdot\textbf{U}=\textbf{b} \\
&\text{where} \\
&A_{ij} = \dots \\
&b_i = \dots
\end{align*}
$$

<!-- strong form

$$
\begin{align*}
&\text{Find}~\textbf{u}(\textbf{x}): \Omega \to \mathbb{R}^d~\text{such that} \\
&\mathbb{BVP}\begin{cases}
\nabla^2\textbf{u} = \textbf{f} & \forall\textbf{x}\in\Omega \\
\textbf{u}=\textbf{u}_{\text{D}} & \forall \textbf{x}\in\partial\Omega_{\text{D}} \\
\textbf{n}\cdot\nabla{\textbf{u}} = \textbf{u}_{\text{N}} & \forall\textbf{x}\in\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}}
\end{cases}~.
\end{align*}
$$

weak form

$$
\begin{align*}
&\text{Find}~\textbf{u}\in V~\text{such that} \\
&F(\textbf{u}, \textbf{v})=-\int_\Omega\text{d}\Omega~\nabla\textbf{v}\cdot\nabla\textbf{u} + \textbf{v}\cdot\textbf{f} + \int_{\partial\Omega}\text{d}\Gamma~\textbf{v}\cdot(\textbf{n}\cdot\nabla\textbf{u})=0\quad\forall \textbf{v}\in V~.
\end{align*}
$$ -->
