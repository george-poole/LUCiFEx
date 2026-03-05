# Stokes equations

## Velocity-pressureity-pressure formulation

### Strong form

$$
\begin{align*}
&\text{Find}~\textbf{u}(\textbf{x}): \Omega \to \mathbb{R}^d~\text{and}~p(\textbf{x}): \Omega \to \mathbb{R}~\text{such that} \\
&\mathbb{BVP}_{\textbf{u}, p}\begin{cases}
\nabla\cdot\textbf{u} = 0 & \\
\textbf{0}=-\nabla p + \nabla\cdot\tau + \textbf{f} & \forall\textbf{x}\in\Omega \\
\textbf{u} = \textbf{u}_{\text{E}} & \forall \textbf{x}\in\partial\Omega_{\text{E}} \\
(-p\mathsf{I}+\tau)\cdot\textbf{n} = \boldsymbol{\tau}_{\text{N}} & \forall\textbf{x}\in\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{E}}
\end{cases} \\
&\text{given} \\
&\mathbb{S}_{\textbf{u}, p}
\begin{cases}
\Omega & \text{domain}\\
\textbf{u}_{\text{E}}(\textbf{x})~,~\partial\Omega_{\text{E}} & \text{velocity essential boundary condition} \\
\boldsymbol{\tau}_{\text{N}}(\textbf{x})~,~\partial\Omega_{\text{N}} & \text{traction natural boundary condition} \\
\tau(\textbf{u}) & \text{deviatoric stress constitutive relation} \\
\textbf{f}(\textbf{x}) & \text{body force} \\
\end{cases}
\end{align*}
$$

### Weak form 

$$
\begin{aligned}
&\text{Find}~(\textbf{u}, p)\in V_{\textbf{u}} \times V_p~\text{such that} \\
&\begin{align*}
F(\textbf{u}, p, \textbf{v}, q)&=\int_\Omega\text{d}\Omega~q(\nabla\cdot\textbf{u}) - p(\nabla\cdot\textbf{v}) + \nabla\textbf{v}:\tau - \textbf{v}\cdot\textbf{f} \\
&\quad -\int_{\partial\Omega_{\text{N}}}\text{d}\Gamma~\,\textbf{v}\cdot\boldsymbol{\tau}_{\text{N}} \\
&=0 \quad\forall(\textbf{v}, q)\in V_{\textbf{u}} \times V_p~.
\end{align*}
\end{aligned}
$$

#### Block structure

$$
\begin{align*}
F_{\textbf{u}\textbf{u}}(\textbf{u}, \textbf{v}) + F_{\textbf{u}p}(p, \textbf{v})  &= 0 \quad\forall \textbf{v}\in V_{\textbf{u}} \\
F_{p\textbf{u}}(\textbf{u}, q) &= 0 \quad\forall q\in V_{p} \\
\end{align*}
\implies
\begin{pmatrix}
\mathsf{A}_{\textbf{u}\textbf{u}} & \mathsf{A}_{\textbf{u}p} \\
\mathsf{A}_{p\textbf{u}} & \mathsf{0}
\end{pmatrix}
\begin{pmatrix}
\textbf{U} \\
\textbf{P}
\end{pmatrix}
\begin{pmatrix}
\textbf{b}_{\textbf{u}} \\
\textbf{b}_{p} \\
\end{pmatrix}
$$

## Streamfunction formulation

### General definition

$$
\begin{align*}
&\textbf{u}=\nabla\times\boldsymbol{\psi} \iff \nabla\cdot\textbf{u}=0 \\
&\text{and}~\tau(\textbf{u}) = \tfrac{1}{2}\left(\nabla\textbf{u} + \nabla\textbf{u}^{\mathsf{T}}\right) \\
&\implies\textbf{0}=\nabla^2(\nabla\times\boldsymbol{\psi}) + \nabla\times\textbf{f}
\end{align*}
$$

### Two-dimensional Cartesian definition

$$
\begin{align*}
&\boldsymbol{\psi}=\psi\textbf{e}_z\implies\textbf{u}=\frac{\partial\psi}{\partial y}\textbf{e}_x - \frac{\partial\psi}{\partial x}\textbf{e}_y \\
&\textbf{f}=f_x\textbf{e}_x + f_y\textbf{e}_y
\end{align*}
$$

### Strong form

$$
\begin{align*}
&\text{Find}~\psi(\textbf{x}): \Omega \to \mathbb{R}~\text{such that} \\
&\mathbb{BVP}_\psi\begin{cases}
\nabla^2(\nabla^2\psi) = \frac{\partial f_y}{\partial x}- \frac{\partial f_x}{\partial y} & \forall\textbf{x}\in\Omega \\
\psi=\psi_{\text{D}} & \forall \textbf{x}\in\partial\Omega \\
\nabla^2\psi=\psi_{\text{L}} & \forall \textbf{x}\in\partial\Omega
\end{cases}~.
\end{align*}
$$

### Weak form 

$$
\begin{aligned}
&\text{Find}~\psi\in V~\text{such that} \\
&\begin{align*}
F(\psi, v) &= 
\int_\Omega\text{d}\Omega~\nabla^2v \nabla^2u - v\frac{\partial f_y}{\partial x} + v\frac{\partial f_x}{\partial y} \\ 
&\quad + \int_{\mathcal{F}}\text{d}\Gamma~\frac{\alpha}{h(\textbf{x})}\left[\!\left[\nabla v\right]\!\right]\left[\!\left[\nabla u\right]\!\right] - \left[\!\left[\nabla v\right]\!\right]\langle\nabla^2u\rangle - \langle\nabla^2v\rangle\left[\!\left[\nabla u\right]\!\right] \\ 
&=0\quad\forall v\in V~.
\end{align*} \\
&\text{where}~\alpha\in\mathbb{R}~\text{is a penalty parameter and}~h(\textbf{x})~\text{is the local mesh cell size.}
\end{aligned}
$$