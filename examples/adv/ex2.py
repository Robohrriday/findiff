import numpy as np
import matplotlib.pyplot as plt

from findiff import BoundaryConditions, FinDiff, Identity, Coefficient, PDE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

eps = 1e-4
b_x, b_y = np.cos(-np.pi/3), np.sin(-np.pi/3)
c = 0
nx, ny = 400, 400
x = np.linspace(0.0, 1.0, nx)
y = np.linspace(0.0, 1.0, ny)
dx, dy = x[1] - x[0], y[1] - y[0]

X, Y = np.meshgrid(x, y, indexing="ij")

f = np.zeros_like(X)

# Dirichlet BCs on all sides using full-grid arrays
bc = BoundaryConditions((nx, ny))

bc_bottom = np.zeros_like(X)
bc[:, 0] = bc_bottom

bc_top = np.ones_like(X)
bc[:, -1] = bc_top

bc_left = np.zeros_like(X)
bc_left[0, int(0.7*ny):] = np.ones_like(bc_left[0, int(0.7*ny):])
bc[0, :] = bc_left

bc_right = np.zeros_like(X)
bc[-1, :] = bc_right

# Operator
diffusion = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)
convection = Coefficient(b_x) * FinDiff(0, dx, 1) + Coefficient(b_y) * FinDiff(1, dy, 1)
L = Coefficient(-eps) * diffusion + convection + Coefficient(c) * Identity()

pde = PDE(L, f, bc)
u = pde.solve()

fig, axes = plt.subplots(1, 1, figsize=(4, 4))

im0 = axes.contourf(X, Y, u, levels=50, cmap="viridis")
axes.set_title("Numerical solution")
axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_aspect("equal")
plt.colorbar(im0, ax=axes)

plt.tight_layout()
plt.savefig("ex2.png", dpi=150, bbox_inches="tight")
plt.show()

# Create a new figure for the 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, u, cmap='viridis') # linewidth=0, antialiased=False

# Add labels and title
ax.set_title("Numerical Solution (3D)")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

# Show the plot
plt.tight_layout()
plt.show()

print("=== Convection-diffusion Ex2 ===")
print(f"Domain: x in [0,1], y in [0,1]")
print(f"Grid size: {nx} x {ny}")
print(f"Coefficients: b = ({b_x}, {b_y}), c = {c}")
print(f"U_min: {np.min(np.min(u, axis = 1), axis = 0)}, U_max: {np.max(np.max(u, axis = 1), axis = 0)}")
print("Figure saved as ex2.png")
