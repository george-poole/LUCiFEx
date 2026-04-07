# Nonlinear equations

## Burgers' equation

### Strong form

$$
\begin{align*}
&\text{Find}~u(x, t): \Omega\times[0,\infty) \to \mathbb{R}~\text{such that } \\
&\mathbb{IBVP}_u\begin{cases}
\frac{\partial u}{\partial t} + au\frac{\partial u}{\partial x}= \frac{\partial}{\partial x}\left(D\frac{\partial u}{\partial x}\right) & \forall(x, t)\in\Omega\times[0,\infty) \\
u=u_0 & \forall(x,t)\in\Omega\times\{0\}\\
u=u_{\text{D}} & \forall(x,t)\in\partial\Omega_{\text{D}}\times[0,\infty) \\
\pm D\frac{\partial u}{\partial x} = u_{\text{N}} & \forall(x,t)\in\partial\Omega_{\text{N}}\times[0,\infty)~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}}
\end{cases}\\
&\text{given} \\
&\mathbb{S}_u\begin{cases}
\Omega\subset\mathbb{R} & \text{one-dimensional domain} \\
u_0(x) & \text{initial condition} \\
u_{\text{D}}(x, t)~,~\partial\Omega_{\text{D}} & \text{Dirichlet boundary condition}\\
u_{\text{N}}(x, t)~,~\partial\Omega_{\text{N}} & \text{Neumann boundary condition}\\
a(x, t) & \text{velocity} \\
D(x, t) & \text{dispersion} \\
\end{cases}
\end{align*}
$$

## Korteweg–De Vries equation

### Strong form

$$
\begin{align*}
&\text{Find}~u(x, t): \Omega\times[0,\infty) \to \mathbb{R}~\text{such that } \\
&\mathbb{IBVP}_u\begin{cases}
\frac{\partial u}{\partial t} + Du\frac{\partial^3u}{\partial x^3} = au\frac{\partial u}{\partial x} & \forall(x, t)\in\Omega\times[0,\infty) \\
u=u_0 & \forall(x,t)\in\Omega\times\{0\}\\
u=u_{\text{D}} & \forall(x,t)\in\partial\Omega_{\text{D}}\times[0,\infty) \\
\pm D\frac{\partial u}{\partial x} = u_{\text{N}}  & \forall(x,t)\in\partial\Omega_{\text{N}}\times[0,\infty)~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{D}}
\end{cases}\\
&\text{given} \\
&\mathbb{S}_u\begin{cases}
\Omega\subset\mathbb{R} & \text{one-dimensional domain} \\
u_0(x) & \text{initial condition} \\
u_{\text{D}}(x, t)~,~\partial\Omega_{\text{D}} & \text{Dirichlet boundary condition}\\
u_{\text{N}}(x, t)~,~\partial\Omega_{\text{N}} & \text{Neumann boundary condition}\\
a(x, t) & \text{velocity} \\
D(x, t) & \text{dispersion} \\
\end{cases}
\end{align*}
$$