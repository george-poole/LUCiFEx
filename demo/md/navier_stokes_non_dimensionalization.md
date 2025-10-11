# Non-dimensionalizations of the Navier-Stokes double-diffusive convection equations

Working in the Boussinesq approximation.

## Scalings

length scale $|\textbf{x}|\sim\mathcal{L}$

velocity scale $|\textbf{u}|\sim\mathcal{U}$

time scale $t\sim\mathcal{T}$

pressure scale $p\sim\mu\mathcal{U}/\mathcal{L}$

buoyancy scale $\rho g\sim g \Delta\rho$

viscosity scale $\mu\sim\mu_{\text{ref}}$

domain length scale $\mathcal{L}_\Omega$

## Dimensionless numbers

Lewis number 
$$Le=G_{\text{ref}}/D_{\text{ref}}$$

Prandtl number (defined with respect to the transport of $c$)
$$Pr=\frac{\mu_{\text{ref}}}{\rho_{\text{ref}}D_{\text{ref}}}$$

Rayleigh number (defined with respect to the transport of $c$ and domain length scale)
$$Ra=\frac{\mathcal{L}^3_\Omega g\Delta\rho}{\mu_{\text{ref}}D_{\text{ref}}}$$

dimensionless domain extent 
$$Xl=\frac{\mathcal{L}_\Omega}{\mathcal{L}}$$

## Dimensional equations

$$\frac{\partial c}{\partial t}+\textbf{u}\cdot\nabla c=\nabla\cdot(\mathsf{D}\cdot\nabla c)$$

$$\frac{\partial\theta}{\partial t}+\textbf{u}\cdot\nabla\theta=\nabla\cdot(\mathsf{G}\cdot\nabla\theta)$$

$$\rho_{\text{ref}}\left(\frac{\partial\textbf{u}}{\partial t}+\textbf{u}\cdot\nabla\textbf{u}\right)=-\nabla p + \mu\nabla^2\textbf{u} + \rho g\,\textbf{e}_g$$

## Non-dimensional equations

$$\frac{\partial c}{\partial t}+\frac{\mathcal{U}\mathcal{T}}{\mathcal{L}}\textbf{u}\cdot\nabla c=\frac{D_{\text{ref}}\mathcal{T}}{\mathcal{L}^2}\nabla\cdot(\mathsf{D}\cdot\nabla c)$$

$$\frac{\partial\theta}{\partial t}+\frac{\mathcal{U}\mathcal{T}}{\mathcal{L}}\textbf{u}\cdot\nabla\theta=\frac{G_{\text{ref}}\mathcal{T}}{\mathcal{L}^2}\nabla\cdot(\mathsf{G}\cdot\nabla\theta)$$

$$\frac{\partial\textbf{u}}{\partial t}+\frac{\mathcal{U}\mathcal{T}}{\mathcal{L}}\textbf{u}\cdot\nabla\textbf{u}=-\frac{\mu_{\text{ref}}\mathcal{T}}{\rho_{\text{ref}}\mathcal{L}^2}\left(\nabla p + \mu\nabla^2\textbf{u}\right) + \frac{\mathcal{T}g\Delta\rho}{\rho_{\text{ref}}\,\mathcal{U}}\textbf{e}_g$$

$$Ad=$$

$$Pe=$$

$$Vi=$$

$$Bu=$$


$$\frac{\partial c}{\partial t}+Ad\,\textbf{u}\cdot\nabla c=\frac{1}{Pe}\nabla\cdot(\mathsf{D}\cdot\nabla c)$$

$$\frac{\partial\theta}{\partial t}+Ad\,\textbf{u}\cdot\nabla\theta=\frac{1}{LePe}\nabla\cdot(\mathsf{G}\cdot\nabla\theta)$$

$$\frac{\partial\textbf{u}}{\partial t}+Ad\,\textbf{u}\cdot\nabla\textbf{u}=-Vi\left(\nabla p + \mu\nabla^2\textbf{u}\right) + Bu\,\textbf{e}_g$$


## Some common choices

| $\mathcal{L}$ | $\mathcal{U}$ |$ \mathcal{T}$ | $\{Ad, Pe, Vi, Bu, Xl\}$ |
| -------- | ------- | ------- | ------- |
| $\mathcal{L}_\Omega$  |  $D_{\text{ref}}/\mathcal{L}$  | $\mathcal{L}/\mathcal{U}$ | $\{1, 1, Pr, PrRa, 1\}$|
|  |      | | |
|     |  | | |


