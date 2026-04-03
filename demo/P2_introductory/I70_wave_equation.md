# Wave equation

## Strong form

$$
\begin{align*}
&\text{Find}~u(\textbf{x}, t): \Omega\times[0,\infty) \to \mathbb{R}~\text{such that } \\
&\mathbb{IBVP}\begin{cases}
\frac{\partial^2u}{\partial t^2} = \nabla\cdot(\mathsf{C}\cdot\nabla u) & \forall(\textbf{x}, t)\in\Omega\times[0,\infty) \\
u=u_0 & \forall(\textbf{x},t)\in\Omega\times\{0\}\\
\frac{\partial u}{\partial t} = \dot{u}_0 & \forall(\textbf{x},t)\in\Omega\times\{0\}\\
u=u_{\text{D}} & \forall(\textbf{x},t)\in\partial\Omega_{\text{D}}\times[0,\infty) \\
\textbf{n}\cdot(\mathsf{C}\cdot\nabla{u}) = u_{\text{N}} & \forall(\textbf{x},t)\in\partial\Omega_{\text{N}}\times[0,\infty)~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}}
\end{cases}~.
\end{align*}
$$

## Time-discretization

$$
\frac{u^{n+1}-2u^n-u^{n-1}}{(\Delta t^n)^2} = \nabla\cdot\mathcal{D}_{\mathsf{C},u}(\mathsf{C}\cdot\nabla u) 
$$

$$u^0=u_0$$

$$u^{-1}=u^0 - \Delta t^0\dot{u}_0$$

## Second-order time derivative formulation

...

## First-order time derivatives formulation

...