# DG formulation of the advection equation

Solutions obtained by the continous Galerkin (CG) formulation of PDEs dominated by advection or containing discontinuities typically suffer from numerical instabilities. A discontinuous Galerkin (DG) formulation provides a better alternative.

## Strong form

See advection demo.

## Weak form

$$
\begin{aligned}
&\text{Find}~u^{n+1}\in V~\text{such that} \\
&\begin{align*}
F(u^{n+1}, v)&=\int_\Omega\text{d}\Omega~v\frac{u^{n+1} - u^n}{\Delta t^n} - \nabla\cdot(v\mathcal{D}_u(u)\mathcal{D}_{\textbf{a}}(\textbf{a})) \\
&\quad + \int_{\mathcal{F}}\text{d}\Gamma~\left[\!\left[ v\right]\!\right] f(\mathcal{D}_u(u)^+, \mathcal{D}_u(u)^-, \textbf{n}\cdot\textbf{a}) \\
&\quad + \int_{\partial\Omega_{\text{I}}}\text{d}\Gamma~vu_{\text{I}}\,\textbf{n}\cdot\mathcal{D}_{\textbf{a}}(\textbf{a}) \\
&\quad + \int_{\partial\Omega/\partial\Omega_{\text{I}}}\text{d}\Gamma~v\mathcal{D}_u(u)\,\textbf{n}\cdot\mathcal{D}_{\textbf{a}}(\textbf{a}) \\
&= 0 \quad\forall v\in V~.
\end{align*} \\
&\text{where $f$ is a numerical flux between facets} \\
& f = \begin{cases}
(\textbf{n}\cdot\textbf{a})u^+  & \text{if }\textbf{n}\cdot\textbf{a} > 0 \\ 
(\textbf{n}\cdot\textbf{a})u^-  & \text{if } \textbf{n}\cdot\textbf{a} \leq 0 \\ 
\end{cases}
\end{aligned}
$$