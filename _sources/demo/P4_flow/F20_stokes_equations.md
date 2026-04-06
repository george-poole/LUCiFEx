# Stokes equations

## Velocity-pressure formulation

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
\tau(\mu,\textbf{u}) & \text{deviatoric stress constitutive relation} \\
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

### Linear algebra

#### Monolithic structure

$$
\begin{align*}
&\begin{pmatrix}\textbf{v}\\q\end{pmatrix}=\boldsymbol{\xi}_i~,~\begin{pmatrix}\textbf{u}\\p\end{pmatrix}=\sum_jX_j\boldsymbol{\xi}_j \\
&\implies A_{ij}X_j=b_i \iff \mathsf{A}\cdot\textbf{X}=\textbf{b} 
\end{align*}
$$

#### Block structure

$$
\begin{align*}
F_{\textbf{u}\textbf{u}}(\textbf{u}, \textbf{v}) + F_{\textbf{u}p}(p, \textbf{v})  &= 0 \quad\forall \textbf{v}\in V_{\textbf{u}} \\
F_{p\textbf{u}}(\textbf{u}, q) + F_{pp}(p, q) &= 0 \quad\forall q\in V_{p} \\
\end{align*}
$$

$$
\begin{align*}
&\textbf{v}=\boldsymbol{\xi}^u_i~,~q=\xi^p_i~,~\textbf{u}=\sum_jU_j\boldsymbol{\xi}^u_j~,~u=\sum_jP_j\xi^p_j \\
&\implies
\begin{pmatrix}
\mathsf{A}_{\textbf{u}\textbf{u}} & \mathsf{A}_{\textbf{u}p} \\
\mathsf{A}_{p\textbf{u}} & \mathsf{A}_{pp}
\end{pmatrix}
\begin{pmatrix}
\textbf{U} \\
\textbf{P}
\end{pmatrix}
\begin{pmatrix}
\textbf{b}_{\textbf{u}} \\
\textbf{b}_{p} \\
\end{pmatrix}
\end{align*}
$$

## Streamfunction formulation

### Strong form

$$
\begin{align*}
&\text{Find}~\psi(\textbf{x}): \Omega \to \mathbb{R}~\text{such that} \\
&\mathbb{BVP}_\psi\begin{cases}
\mu\nabla^2(\nabla^2\psi) = \frac{\partial f_y}{\partial x}- \frac{\partial f_x}{\partial y} & \forall\textbf{x}\in\Omega \\
\psi=\psi_{\text{E}} & \forall \textbf{x}\in\partial\Omega \\
\mu\nabla^2\psi=\psi_{\text{W}} & \forall \textbf{x}\in\partial\Omega
\end{cases}\\
&\text{given} \\
&\mathbb{S}_{\psi}\begin{cases}
\Omega\subset\mathbb{R}^2  & \text{domain}\\
\psi_{\text{E}}(\textbf{x})~,~\partial\Omega_{\text{E}} & \text{essential boundary condition} \\
\psi_{\text{W}}(\textbf{x})~,~\partial\Omega_{\text{W}} & \text{weak boundary condition} \\
f_x(\textbf{x}), f_y(\textbf{x}) & \text{body force} \\
\end{cases}\\
&\text{where}\\
&\textbf{u}=\nabla\times\boldsymbol{\psi}=\textbf{u}=\nabla\times\psi\textbf{e}_z=\frac{\partial\psi}{\partial y}\textbf{e}_x - \frac{\partial\psi}{\partial x}\textbf{e}_y \iff \nabla\cdot\textbf{u}=0\\
&\textbf{f}=f_x\textbf{e}_x + f_y\textbf{e}_y \\
&\tau(\mu,\textbf{u}) = \tfrac{\mu}{2}\left(\nabla\textbf{u} + \nabla\textbf{u}^{\mathsf{T}}\right) \\
&\nabla\mu=\textbf{0} \\
\end{align*}
$$

### Weak form 

$$
\begin{aligned}
&\text{Find}~\psi\in V~\text{such that} \\
&\begin{align*}
F(\psi, v) &= 
\int_\Omega\text{d}\Omega~\mu\nabla^2v \nabla^2u - v\frac{\partial f_y}{\partial x} + v\frac{\partial f_x}{\partial y} \\ 
&\quad + \int_{\mathcal{F}}\text{d}\Gamma~\frac{\alpha\mu}{h(\textbf{x})}\left[\!\left[\nabla v\right]\!\right]\left[\!\left[\nabla u\right]\!\right] - \mu\left[\!\left[\nabla v\right]\!\right]\langle\nabla^2u\rangle - \mu\langle\nabla^2v\rangle\left[\!\left[\nabla u\right]\!\right] \\ 
&=0\quad\forall v\in V
\end{align*} \\
&\text{where}~\alpha\in\mathbb{R}~\text{is a penalty parameter}\\
&\text{and}~h(\textbf{x})~\text{is the local mesh cell size.}
\end{aligned}
$$