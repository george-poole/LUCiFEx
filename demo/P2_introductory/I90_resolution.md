# Spatial and temporal resolution

## Spatial resolution

$h$-refinement refines the mesh (locally or globally) to achieve a smaller cell size $h$ where desired.

$p$-refinement increases the polynomial degree $p$ (locally or globally) of the finite element basis functions. However `dolfinx` has no support for local $p$-refinement.

$hp$-refinement combines both strategies.

### Method of manufactured solutions

$$
\begin{align*}
&\text{Find}~u(\textbf{x}): \Omega \to \mathbb{R}~\text{such that} \\
&\mathbb{BVP}_u\begin{cases}
\mathscr{L}_{\textbf{x}}(u) = \mathscr{L}_{\textbf{x}}(u_{\text{e}}) & \forall\textbf{x}\in\Omega \\
u=u_{\text{e}} & \forall \textbf{x}\in\partial\Omega \\
\end{cases}~.
\end{align*}
$$

The goal is to demonstrate that the error norm tends to zero as mesh cell cell size tends to zero. Asymptotically as $h\to0$ for the error $\mathcal{E}=u_{\text{e}} - u$, we expect

$$||\mathcal{E}||\sim C h^r$$

for some numerical factor $C>0$ and convergence rate $r>0$. The $L_p$ and $\ell_p$ norms are defined by

$$||\mathcal{E}||_{L_p}=\left(\int_\Omega |\mathcal{E}|^p~\text{d}\Omega\right)^{1/p}$$

$$||\mathcal{E}||_{\ell_p}=\left(\sum_j|E_j|^p\right)^{1/p}$$

if $\sum_jE_j\xi_j$ is the approximation of $\mathcal{E}$ in the finite element function space spanned by basis functions $\{\xi_j\}_j$.


## Temporal resolution

$\Delta t$-refinement uses a smaller constant time-step.

$\Delta t$-adaptivity computes an adaptive time-step based on constraints such as the CFL condition.

### Adaptive time-steps for the advection-diffusion-reaction equation

The precise value of Courant number necessary for stability can depend on the choice of discretization in space and time, as well as the number of spatial dimensions.

$$
\begin{align*}
\Delta t &= \min\{C_{\textbf{u}}\Delta t_{\textbf{u}}, C_{\mathsf{D}}\Delta t_{\mathsf{D}}, C_{R}\Delta t_{R}\} \\
\Delta t_{\textbf{u}} &= \min_{\textbf{x}}\left(\frac{h}{|\textbf{u}|}\right)  \\
\Delta t_{\mathsf{D}} & = \min_{\textbf{x}}\left(\frac{h^2}{2|\mathsf{D}|}\right) \\
\Delta t_{R} &= \min_{\textbf{x}}\left(\frac{1}{|R|}\right) \\
\end{align*}
$$

### Method of manufactured solutions

$$
\begin{align*}
&\text{Find}~u(\textbf{x}, t): \Omega\times[0,\infty) \to \mathbb{R}~\text{such that} \\
&\begin{cases}
\frac{\partial u}{\partial t} = \mathscr{L}_{\textbf{x},t}(u) & \forall(\textbf{x}, t)\in\Omega\times[0,\infty) \\
u=u_{\text{e}} & \forall(\textbf{x},t)\in\Omega\times\{0\}\\
u=u_{\text{e}} & \forall(\textbf{x},t)\in\partial\Omega\times[0,\infty)
\end{cases}~.
\end{align*}
$$

Given a fixed time-step $\Delta t$ and mesh cell size $h$, the error norm at time $t$ is expected to scale as

$$||\mathcal{E}(t)||\sim C_{\Delta t}\Delta t^{\,r_{\Delta t}} + C_hh^{r_h}$$

with numerical factors $C_{\Delta t}, C_h$ convergence rates $r_\Delta, r_h$ themselves being time-dependent. The maximum-in-time error norm is given by

$$\max_{t\geq0}||\mathcal{E}(t)|| = \max_{n\geq0}||\mathcal{E}(n\Delta t)||$$