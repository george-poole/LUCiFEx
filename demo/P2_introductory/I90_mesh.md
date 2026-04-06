# Mesh resolution

## $h$ and $p$ refinement

$h$-refinement refines the mesh (locally or globally) to achieve a smaller cell size $h$ where desired.

$p$-refinement increases the polynomial degree $p$ (locally or globally) of the finite element basis functions. However `dolfinx` has no support for local $p$-refinement.

$hp$-refinement combines both strategies.

## Method of manufactured solutions

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

for some constant $C>0$ and convergence rate $r>0$. The $L_p$ and $\ell_p$ norms are defined by

$$||\mathcal{E}||_{L_p}=\left(\int_\Omega |\mathcal{E}|^p~\text{d}\Omega\right)^{1/p}$$

$$||\mathcal{E}||_{\ell_p}=\left(\sum_j|E_j|^p\right)^{1/p}$$

if $\sum_jE_j\xi_j$ is approximation of $\mathcal{E}$ in the finite element function space spanned by basis functions $\{\xi_j\}_j$.