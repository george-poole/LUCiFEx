# Darcy ABC convection-reaction equations

Governing equations for triple solutal convection with an $\text{A}+\text{B}\to\text{C}$ reaction coupled to Darcy flow.

## Dimensional equations

$$
\begin{align*}
&\text{Find} \\
&a(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R},  \\
&b(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R},  \\
&c(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R},  \\
&\textbf{u}(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R}^d, \\
&p(\textbf{x}, t): \Omega\times[0, \infty) \to \mathbb{R} \\
&\text{such that} \\
&\begin{cases}
\phi\frac{\partial a}{\partial t} + \textbf{u}\cdot\nabla a = \nabla\cdot(\mathsf{D}_a(\phi, \textbf{u})\cdot \nabla a) - R \\
\phi\frac{\partial b}{\partial t} + \textbf{u}\cdot\nabla b = \nabla\cdot(\mathsf{D}_b(\phi, \textbf{u})\cdot \nabla b) - R \\
\phi\frac{\partial c}{\partial t} + \textbf{u}\cdot\nabla c = \nabla\cdot(\mathsf{D}_c(\phi, \textbf{u})\cdot \nabla c) + R \\
\nabla\cdot\textbf{u}=0 \\
\textbf{u}=-\frac{\mathsf{K}}{\mu}\cdot\left(\nabla p - \rho g\,\textbf{e}_g\right) & \forall(\textbf{x}, t)\in\Omega\times[0,\infty)\\
w=w_0  & \forall(\textbf{x}, t)\in\partial\Omega_{\text{D},w} \times [0,\infty)\quad\forall w\in\{a, b, c\}\\
w = w_{\text{D},w} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{D},w} \times [0,\infty)\quad\forall w\in\{a, b, c\}\\
\textbf{n}\cdot(\mathsf{D}_w\nabla w) = w_{\text{N}, w} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{N},w} \times [0,\infty)\quad\forall w\in\{a, b, c\}\\
\textbf{n}\cdot\textbf{u} = u_{\text{E}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{E}} \times [0,\infty) \\
p = p_{\text{N}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{N}}\times [0,\infty)~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{E}}
\end{cases} \\
&\text{given} \\
&\mathbb{S}=
\begin{cases}
\Omega\subset\mathbb{R}^d & \text{domain}\\
w_0(\textbf{x})~\forall w\in\{a,b,c\} & \text{solutal initial conditions}\\ 
\phi(\textbf{x}) \\
\mathsf{K}(\phi) \\
\mathsf{D}_w(\phi, \textbf{u}) ~\forall w\in\{a,b,c\} & \text{solutal dispersions}\\ 
\rho(a,b,c) \\
\mu(a,b,c) \\
R(a,b)
\end{cases}
\end{align*}
$$

## Non-dimensionalization

### Scalings


| Quantity | $\vert\textbf{x}\vert$ | $\vert\textbf{u}\vert$ | $t$ | $a$ | $b$ | $c$ | $\rho g$ | $p$ | $\psi$ |
| -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |  
| **Scaling** | $\mathcal{L}$ | $\mathcal{U}$ |$\mathcal{T}$ | $\Delta a$ |$\Delta b$ | $\Delta c$ | $g \Delta\rho$ | $\mu_{\text{ref}}\,\mathcal{U}\mathcal{L}/K_{\text{ref}}$ | $\mathcal{U}\mathcal{L}$ |

| $\mu$ | $\phi$ | $K$ | $\vert\mathsf{D}_a\vert$ | $\vert\mathsf{D}_b\vert$ | $\vert\mathsf{D}_c\vert$ | $R$ |
| ------- | ------- | ------- | ------- | ------- |  ------- | ------- | 
| $\mu_{\text{ref}}$ | $\phi_{\text{ref}}$ |$K_{\text{ref}}$ | $D_{a, \text{ref}}$ | $D_{b, \text{ref}}$ | $D_{c, \text{ref}}$ | $\Delta R$ |

### Abstract dimensionless numbers

$$
\begin{align*}
Ad&=\frac{\mathcal{U}\mathcal{T}}{\phi_{\text{ref}}\mathcal{L}} \\
Di&=\frac{D_{a, \text{ref}}\mathcal{T}}{\phi_{\text{ref}}\mathcal{L}^2} \\
Ki&=\frac{\mathcal{T}\Delta R}{\phi_{\text{ref}}\Delta c} \\
Bu&=\frac{K_{\text{ref}}\,g\Delta\rho}{\mu_{\text{ref}}\,\mathcal{U}} \\
\end{align*} 
$$

### Physical dimensionless numbers

...

## Non-dimensional time-discretized equations

### Strong form

$$
\begin{align*}
&\text{Find}~a^{n+1}, b^{n+1}, c^{n+1},~\textbf{u}^n,~p^n~\text{such that}~\forall n\geq0 \\
&\begin{cases}
\phi\frac{a^{n+1}-a^n}{\Delta t^n} + Ad\,\mathcal{D}_{\textbf{u}, a}(\textbf{u}\cdot\nabla a) = Di\nabla\cdot\mathcal{D}_{\mathsf{D},a}(\mathsf{D}_{a}(\phi, \textbf{u})\cdot \nabla a) - Ki R(a,b) \\
\phi\frac{b^{n+1}-b^n}{\Delta t^n} + Ad\,\mathcal{D}_{\textbf{u}, b}(\textbf{u}\cdot\nabla b) = \frac{Di}{Le_b}\nabla\cdot\mathcal{D}_{\mathsf{D},b}(\mathsf{D}_b(\phi, \textbf{u})\cdot \nabla b) - Ki R(a,b) \\
\phi\frac{c^{n+1}-c^n}{\Delta t^n} + Ad\,\mathcal{D}_{\textbf{u}, c}(\textbf{u}\cdot\nabla c) = \frac{Di}{Le_c}\nabla\cdot\mathcal{D}_{\mathsf{D},c}(\mathsf{D}_c(\phi, \textbf{u})\cdot \nabla c) + Ki R(a,b) \\
\nabla\cdot\textbf{u}^n=0 \\
\textbf{u}^n=-\frac{\mathsf{K}}{\mu^n}\cdot\left(\nabla p^n + Bu\,\rho^ng\,\textbf{e}_g\right) \\
\end{cases}
\end{align*}
$$

### Weak forms

...