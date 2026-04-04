# Predictor-corrector methods for the transport equation

## Strong form

$$
\begin{align*}
&\text{Find}~u(\textbf{x}, t): \Omega\times[0,\infty) \to \mathbb{R}\text{~and~}\textbf{a}(\textbf{x}, t): \Omega\times[0,\infty) \to \mathbb{R}^d~\text{such that } \\
&\mathbb{IBVP}_{u,\textbf{a}}\begin{cases}
\frac{\partial u}{\partial t} + \textbf{a}\cdot\nabla u= \nabla\cdot(\mathsf{D}\cdot\nabla u) \\
\mathscr{L}_{\textbf{x}}(\textbf{a}, u)=\textbf{0} & \forall(\textbf{x}, t)\in\Omega\times[0,\infty) \\
u=u_0 & \forall(\textbf{x},t)\in\Omega\times\{0\}\\
u=u_{\text{D}} & \forall(\textbf{x},t)\in\partial\Omega_{\text{D}}\times[0,\infty) \\
\textbf{n}\cdot(\mathsf{D}\cdot\nabla{u}) = u_{\text{N}} & \forall(\textbf{x},t)\in\partial\Omega_{\text{N}}\times[0,\infty)~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}} \\
\mathcal{B}_{\textbf{x}}(\textbf{a})=\textbf{a}_{\text{B}} & \forall(\textbf{x},t)\in\partial\Omega_{\text{B}}\times[0,\infty) \\
\end{cases} \\
&\text{given} \\
&\mathbb{S}_{u,\textbf{a}}\begin{cases}
\Omega\subset\mathbb{R}^d & \text{domain} \\
u_0(\textbf{x}) & \text{initial condition} \\
u_{\text{D}}(\textbf{x}, t)~,~\partial\Omega_{\text{D}} & \text{Dirichlet boundary condition}\\
u_{\text{N}}(\textbf{x}, t)~,~\partial\Omega_{\text{N}} & \text{Neumann boundary condition}\\
\mathscr{L}_{\textbf{x}} & \text{velocity governing equation spatial operator} \\
\mathcal{B}_{\textbf{x}} & \text{velocity boundary condition spatial operator} \\
\textbf{a}_{\text{B}}(\textbf{x}, t)~,~\partial\Omega_{\text{B}} & \text{velocity boundary condition}\\
\mathsf{D}(\textbf{x}, t) & \text{dispersion} \\
\end{cases}
\end{align*}
$$

## Linearized weak forms

$$
\begin{aligned}
&\text{Find} \\
&\textbf{a}^{n}\in V_{\textbf{a}}, \\
&\widetilde{u}^{n+1}\in V_{u}, \\
&\widetilde{\textbf{a}}^{n+1}\in V_{\textbf{a}}, \\
&u^{n+1}\in V_{u} \\
&\text{such that} \\
&\mathbb{F}_{\textbf{a}, \widetilde{u},\widetilde{\textbf{a}},u} 
\begin{cases}
F_1(\textbf{a}^{n}, \textbf{v}) = 0 \quad\forall \textbf{v}\in V_{\textbf{a}} \\
F_2(\widetilde{u}^{n+1}, v) = 0 \quad\forall v\in V_u \\
F_3(\widetilde{\textbf{a}}^{n+1}, \textbf{v}) = 0 \quad\forall \textbf{v}\in V_{\textbf{a}} \\
F_4(u^{n+1}, v) = 0 \quad\forall v\in V_u \\
\end{cases}
\end{aligned}
$$