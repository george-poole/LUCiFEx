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
&\mathbb{IBVP}_{\textbf{u},p,a,b,c}\begin{cases}
\phi\frac{\partial a}{\partial t} + \textbf{u}\cdot\nabla a = \nabla\cdot(\mathsf{D}_a(\phi, \textbf{u})\cdot \nabla a) - \Sigma \\
\phi\frac{\partial b}{\partial t} + \textbf{u}\cdot\nabla b = \nabla\cdot(\mathsf{D}_b(\phi, \textbf{u})\cdot \nabla b) - \Sigma \\
\phi\frac{\partial c}{\partial t} + \textbf{u}\cdot\nabla c = \nabla\cdot(\mathsf{D}_c(\phi, \textbf{u})\cdot \nabla c) + \Sigma \\
\nabla\cdot\textbf{u}=0 \\
\textbf{u}=-\frac{\mathsf{K}}{\mu}\cdot\left(\nabla p - \rho g\,\textbf{e}_g\right) & \forall(\textbf{x}, t)\in\Omega\times[0,\infty)\\
w=w_0  & \forall(\textbf{x}, t)\in\partial\Omega_{\text{D},w} \times [0,\infty)\quad\forall w\in\{a, b, c\}\\
w = w_{\text{D},w} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{D},w} \times [0,\infty)\quad\forall w\in\{a, b, c\}\\
\textbf{n}\cdot(\mathsf{D}_w\nabla w) = w_{\text{N}, w} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{N},w} \times [0,\infty)\quad\forall w\in\{a, b, c\}\\
\textbf{n}\cdot\textbf{u} = u_{\text{E}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{E}} \times [0,\infty) \\
p = p_{\text{N}} & \forall(\textbf{x}, t)\in\partial\Omega_{\text{N}}\times [0,\infty)~,~\partial\Omega_{\text{N}}=\partial\Omega/\partial\Omega_{\text{E}}
\end{cases} \\
&\text{given} \\
&\mathbb{S}_{\textbf{u},p,a,b,c}=
\begin{cases}
\Omega\subset\mathbb{R}^d & \text{domain}\\
w_0(\textbf{x})~\forall w\in\{a,b,c\} & \text{solutal initial conditions}\\ 
\phi(\textbf{x}) \\
\mathsf{K}(\phi) \\
\mathsf{D}_w(\phi, \textbf{u}) ~\forall w\in\{a,b,c\} & \text{solutal dispersions}\\ 
\rho(a,b,c) \\
\mu(a,b,c) \\
\Sigma(a,b,c)
\end{cases}
\end{align*}
$$

## Non-dimensionalization

### Scalings


| Quantity | $\vert\textbf{x}\vert$ | $\vert\textbf{u}\vert$ | $t$ | $a$ | $b$ | $c$ | $\rho g$ | $p$ | $\psi$ |
| -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |  
| **Scaling** | $\mathcal{L}$ | $\mathcal{U}$ |$\mathcal{T}$ | $\Delta a$ |$\Delta b$ | $\Delta c$ | $g \Delta\rho$ | $\mu_{\text{ref}}\,\mathcal{U}\mathcal{L}/K_{\text{ref}}$ | $\mathcal{U}\mathcal{L}$ |

| $\mu$ | $\phi$ | $K$ | $\vert\mathsf{D}_a\vert$ | $\vert\mathsf{D}_b\vert$ | $\vert\mathsf{D}_c\vert$ | $\Sigma$ |
| ------- | ------- | ------- | ------- | ------- |  ------- | ------- | 
| $\mu_{\text{ref}}$ | $\phi_{\text{ref}}$ |$K_{\text{ref}}$ | $D_{a, \text{ref}}$ | $D_{b, \text{ref}}$ | $D_{c, \text{ref}}$ | $\Delta\Sigma$ |

### Generic dimensionless numbers

$$
Ad=\frac{\mathcal{U}\mathcal{T}}{\phi_{\text{ref}}\mathcal{L}}~,~
Di=\frac{D_{a,\text{ref}}\mathcal{T}}{\phi_{\text{ref}}\mathcal{L}^2}~,~
Ki=\frac{\mathcal{T}\Delta\Sigma}{\phi_{\text{ref}}\Delta a}~,~
Bu=\frac{K_{\text{ref}}\,g\Delta\rho}{\mu_{\text{ref}}\,\mathcal{U}}~,~
X=\frac{\mathcal{L}}{\mathcal{L}_\Omega}
$$

### Physical dimensionless numbers

| Definition | Name | Physical interpretation | 
| -------- | ------- | ------- |
| $Ra=\frac{\mathcal{L}_\Omega K_{\text{ref}}g\Delta\rho}{\mu_{\text{ref}}D_{\text{ref}}}=\underbrace{\frac{K_{\text{ref}}\,g\Delta\rho}{\mu_{\text{ref}}}}_{\text{convective speed}} \big/ \underbrace{\frac{D_{a,\text{ref}}}{\mathcal{L}_\Omega}}_{\text{diffusive speed}}$  |  Rayleigh  | Ratio of convective to diffusive speeds, defined with respect to the transport of $a$ and domain length scale. |
| $Da=\frac{\mathcal{L}_\Omega \mu_{\text{ref}}\,\Delta\Sigma}{K_{\text{ref}}\,g\Delta\rho\Delta a} = \underbrace{\frac{\Delta\Sigma}{\Delta a}}_{\text{reaction rate}} \big/ \underbrace{\frac{K_{\text{ref}}\,g\Delta\rho}{\mathcal{L}_\Omega \mu_{\text{ref}}}}_{\text{convection rate}}$  |  Damköhler  | Ratio of reaction to convection rates, defined with respect to the transport of $a$ and domain length scale. |
| $Le_w=\frac{D_{w,\text{ref}}}{D_{a,\text{ref}}}\quad\forall w\in\{b,c\}$  |  Lewis  | Ratio of diffusivities. |
| $Lr_w=\frac{\Delta w}{\Delta a}\quad\forall w\in\{b,c\}$    |  | Effective stoichiometric coefficient. |

### Scaling choice

| Name | $\mathcal{L}$ | $\mathcal{U}$ | $\mathcal{T}$ | $Ad$ | $Di$ | $Ki$ | $Bu$ | $X$ |
|------|--------------|--------------|--------------|-----|-----|-----|-----|-----|
| advective | $\mathcal{L}_\Omega$ | $K_{\text{ref}}\,g\Delta\rho/\mu_{\text{ref}}$ | $\phi_{\text{ref}}\mathcal{L}/\mathcal{U}$ | $1$ | $1/Ra$ | $Da$ | $1$ | $1$ |
| diffusive | $\mathcal{L}_\Omega$ | $D_{\text{ref}}/\mathcal{L}$ | $\phi_{\text{ref}}\mathcal{L}/\mathcal{U}$ | $1$ | $1$ | $RaDa$ | $Ra$ | $1$ |
| advective-diffusive | $D_{\text{ref}}/\mathcal{U}$ | $K_{\text{ref}}\,g\Delta\rho/\mu_{\text{ref}}$ | $\phi_{\text{ref}}\mathcal{L}/\mathcal{U}$ | $1$ | $1$ | $Da/Ra$ | $1$ | $Ra$ |
| reactive | $\sqrt{D_{\text{ref}}\mathcal{T}/\phi_{\text{ref}}}$ | $\phi_{\text{ref}}\mathcal{L}/\mathcal{T}$ | $\phi_{\text{ref}}\Delta a/\Delta\Sigma$ | $1$ | $1$ | $1$ | $\sqrt{Ra/Da}$ | $\sqrt{RaDa}$ |

## Non-dimensional time-discretized equations

### Strong form

$$
\begin{align*}
&\text{Find}~a^{n+1}, b^{n+1}, c^{n+1},~\textbf{u}^n,~p^n~\text{such that}~\forall n\geq0 \\
&\begin{cases}
\phi\frac{a^{n+1}-a^n}{\Delta t^n} + Ad\,\mathcal{D}_{\textbf{u}, a}(\textbf{u}\cdot\nabla a) = Di\,\nabla\cdot\mathcal{D}_{\mathsf{D}_a,a}(\mathsf{D}_{a}(\phi, \textbf{u})\cdot \nabla a) - Ki\,\Sigma \\
\phi\frac{b^{n+1}-b^n}{\Delta t^n} + Ad\,\mathcal{D}_{\textbf{u}, b}(\textbf{u}\cdot\nabla b) = Le_bDi\,\nabla\cdot\mathcal{D}_{\mathsf{D}_b,b}(\mathsf{D}_b(\phi, \textbf{u})\cdot \nabla b) - Lr_b Ki\,\Sigma \\
\phi\frac{c^{n+1}-c^n}{\Delta t^n} + Ad\,\mathcal{D}_{\textbf{u}, c}(\textbf{u}\cdot\nabla c) = Le_cDi\,\nabla\cdot\mathcal{D}_{\mathsf{D}_c,c}(\mathsf{D}_c(\phi, \textbf{u})\cdot \nabla c) + Lr_c Ki\,\Sigma \\
\nabla\cdot\textbf{u}^n=0 \\
\textbf{u}^n=-\frac{\mathsf{K}}{\mu^n}\cdot\left(\nabla p^n + Bu\,\rho^ng\,\textbf{e}_g\right) \\
\end{cases}
\end{align*}
$$