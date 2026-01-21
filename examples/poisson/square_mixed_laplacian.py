from findiff import FinDiff, PDE, BoundaryConditions
import numpy as np
import matplotlib.pyplot as plt


shape = (100, 100)
x, y = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
dx, dy = x[1]-x[0], y[1]-y[0]
X, Y = np.meshgrid(x, y, indexing='ij')

L = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)
f = np.zeros(shape)

bc = BoundaryConditions(shape)
bc[1,:] = FinDiff(0, dx, 1), 0  # Neumann BC
bc[-1,:] = 300. - 200*Y   # Dirichlet BC
bc[:, 0] = 300.   # Dirichlet BC
bc[1:-1, -1] = FinDiff(1, dy, 1), 0  # Neumann BC

pde = PDE(L, f, bc)
u = pde.solve()

# visualization
plt.imshow(u, extent=(0, 1, 0, 1), origin='lower')
plt.colorbar(label='Temperature')
plt.title('Steady-State Heat Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()