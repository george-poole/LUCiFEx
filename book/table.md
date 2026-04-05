# Summary

Name | Equation | Classification(s) |
| -------- | -------- | ------- |
| Poisson | $\nabla\cdot(\mathsf{D}\cdot\nabla u) = f$ | elliptic, second-order in space |
| diffusion | $\frac{\partial u}{\partial t} = \nabla\cdot(\mathsf{D}\cdot\nabla u)$ | parabolic, second-order in space, first-order in time  |
| advection| $\frac{\partial u}{\partial t}+\textbf{a}\cdot\nabla u = 0$ | hyperbolic, first-order in space, first-order in time| 
| advection-diffusion| $\frac{\partial u}{\partial t}+\textbf{a}\cdot\nabla u = \nabla\cdot(\mathsf{D}\cdot\nabla u)$ | parabolic, second-order in space, first-order in time |
| advection-diffusion-reaction (nonlinear) | $\frac{\partial u}{\partial t}+\textbf{a}\cdot\nabla u = \nabla\cdot(\mathsf{D}\cdot\nabla u) + \Sigma(u)$ | parabolic, second-order in space, first-order in time |
| advection-diffusion-reaction (linear) | $\frac{\partial u}{\partial t}+\textbf{a}\cdot\nabla u = \nabla\cdot(\mathsf{D}\cdot\nabla u) + Ru + J$ | parabolic, second-order in space, first-order in time |
| steady advection-diffusion-reaction | $\textbf{a}\cdot\nabla u = \nabla\cdot(\mathsf{D}\cdot\nabla u) + Ru + J$ | parabolic, second-order in space |
| steady diffusion-reaction | $0 = \nabla\cdot(\mathsf{D}\cdot\nabla u) + Ru + J$ | parabolic, second-order in space |
| steady advection-reaction | $\textbf{a}\cdot\nabla u = Ru + J$ | hyperbolic, first-order in space |
| Helmholtz | $\nabla\cdot(\mathsf{D}\cdot\nabla u) + k^2 u = f$ | elliptic, second-order in space | 
| wave | $\frac{\partial^2u}{\partial t^2} = \nabla\cdot(\mathsf{D}\cdot\nabla u)$ | hyperbolic, second-order in space, second-order in time |
| Darcy | $\begin{matrix}\nabla\cdot\textbf{u} = 0\\ \textbf{u} = -\frac{\mathsf{K}}{\mu}\cdot(\nabla p - \textbf{f}\,)\end{matrix}$ | mixed, first-order in space | 
| Darcy streamfunction | $\nabla\cdot\left(\frac{\mu\mathsf{K}^{\mathsf{T}}\cdot\nabla\psi}{\text{det}(\mathsf{K})}\right)=-\frac{\partial(f_y)}{\partial x} + \frac{\partial(f_x)}{\partial y}$ | elliptic, second-order in space | 
| Darcy pressure | $\nabla\cdot\left(\frac{\mathsf{K}}{\mu}\cdot\nabla p\right)=\nabla\cdot\left(\frac{\mathsf{K}}{\mu}\cdot\textbf{f}\right)$ | elliptic, second-order in space |
| Stokes | $\begin{matrix}\nabla\cdot\textbf{u} = 0\\ \textbf{0}=-\nabla p + \nabla\cdot\tau + \textbf{f}\end{matrix}$ | mixed, second-order in space |  
| Navier-Stokes | $\begin{matrix}\nabla\cdot\textbf{u} = 0\\ \rho \left(\frac{\partial\textbf{u}}{\partial t}+\textbf{u}\cdot\nabla\textbf{u}\right)=-\nabla p + \nabla\cdot\tau + \textbf{f}\end{matrix}$ | mixed, second-order in space, first-order in time |  
  