# LUCiFEx Demo Book

## Building the Juptyer Book

To execute only notebooks with name matching the glob pattern `XYZ*` do

`bash build.sh "XYZ*"`


For a dry-run do

`bash build.sh --dry "XYZ*"`


For a complete rebuild do

`bash build.sh "*" "" "--allow-errors" "--all"` 

## Style guide

+ no gaps between imports in notebooks
+ problem specification listed in the order: domain, domain boundaries, initial conditions, boundary conditions and lastly coefficients or constitutive relations

## TODO

+ user guide notebooks
+ vector Poisson equation
+ Darcy-Brinkman notebooks
+ ABC convection notebook
+ fix Nitsche
+ ~~DG advection multifigures~~