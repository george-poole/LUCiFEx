# Darcy-Brinkman equations

## Nonlinear strong form

$$
\begin{align*}
&\text{Find} \\
&\textbf{u}(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R}^d, \\
&p(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R} \\
&\text{such that} \\
&\mathbb{IBVP}_{\textbf{u},p}\begin{cases}
\nabla\cdot\textbf{u} = 0 & \\
\rho\left(\frac{\partial\textbf{u}}{\partial t}+\textbf{u}\cdot\nabla(\phi^{-1}\textbf{u})\right)=-\phi\nabla p + \nabla\cdot\tau + \phi\,\textbf{f} - \mu\phi\mathsf{K}^{-1}\cdot\textbf{u} \\
\textbf{u}=\textbf{u}_0 & \forall(\textbf{x}, t)\in\Omega\times\{0\} \\
p=p_0 & \forall(\textbf{x}, t)\in\Omega\times\{0\} \\
\textbf{u} = \textbf{u}_{\text{E}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{E}} \times [0,\infty) \\
(-p\mathsf{I}+\tau)\cdot\textbf{n} = \boldsymbol{\tau}_{\text{N}} & \forall(\textbf{x},t)\in\partial\Omega_{\text{N}}\times[0, \infty)~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{E}}
\end{cases} \\
&\text{given} \\
&\mathbb{S}_{\textbf{u},p}=
\begin{cases}
\Omega\subset\mathbb{R}^d & \text{domain} \\
\mu_{\text{ref}} & \text{constant viscosity} \\
\tau(\mu,\textbf{u}) & \text{deviatoric stress} \\
\end{cases} 
\end{align*}
$$