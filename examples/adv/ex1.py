import numpy as np
import matplotlib.pyplot as plt

from findiff import BoundaryConditions, FinDiff, Identity, Coefficient, PDE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


eps = 1e-3
b_x, b_y = 2, 1
c = 1
nx, ny = 400, 400
x = np.linspace(0.0, 1.0, nx)
y = np.linspace(0.0, 1.0, ny)
dx, dy = x[1] - x[0], y[1] - y[0]

X, Y = np.meshgrid(x, y, indexing="ij")

u_exact = Y*(1-Y)*(X - (np.exp((X-1)/eps) - np.exp(-1/eps))/(1 - np.exp(-1/eps)))
du_dx = Y*(1-Y)*(1 - (np.exp((X-1)/eps))/(eps - eps*np.exp(-1/eps)))
du_dy = (1 - 2 * Y) * (X - (np.exp((X-1)/eps) - np.exp(-1/eps))/(1 - np.exp(-1/eps)))
du2_dx2 = Y*(1-Y)*(0 - (np.exp((X-1)/eps))/(eps**2 - eps**2*np.exp(-1/eps)))
du2_dy2 = -2 * (X - (np.exp((X-1)/eps) - np.exp(-1/eps))/(1 - np.exp(-1/eps)))
laplace_u = du2_dx2 + du2_dy2
f = -eps*laplace_u + b_x * du_dx + b_y * du_dy + c * u_exact

# Dirichlet BCs on all sides using full-grid arrays
bc = BoundaryConditions((nx, ny))

bc_bottom = np.zeros_like(u_exact)
bc_bottom[:, 0] = u_exact[:, 0]
bc[:, 0] = bc_bottom

bc_top = np.zeros_like(u_exact)
bc_top[:, -1] = u_exact[:, -1]
bc[:, -1] = bc_top

bc_left = np.zeros_like(u_exact)
bc_left[0, :] = u_exact[0, :]
bc[0, :] = bc_left

bc_right = np.zeros_like(u_exact)
bc_right[-1, :] = u_exact[-1, :]
bc[-1, :] = bc_right

# Operator
diffusion = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)
convection = Coefficient(b_x) * FinDiff(0, dx, 1) + Coefficient(b_y) * FinDiff(1, dy, 1)
L = Coefficient(-eps) * diffusion + convection + Coefficient(c) * Identity()

pde = PDE(L, f, bc)
u = pde.solve()

error = u - u_exact
l2_rel = np.linalg.norm(error) / np.linalg.norm(u_exact)
linf = np.abs(error).max()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im0 = axes[0].contourf(X, Y, u, levels=50, cmap="viridis")
axes[0].set_title("Numerical solution")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_aspect("equal")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].contourf(X, Y, u_exact, levels=50, cmap="viridis")
axes[1].set_title("Exact solution")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_aspect("equal")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].contourf(X, Y, error, levels=50, cmap="RdBu_r")
axes[2].set_title("Error (u - u_exact)")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].set_aspect("equal")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("ex1.png", dpi=150, bbox_inches="tight")
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

print("=== Convection-diffusion Ex1 ===")
print(f"Domain: x in [0,1], y in [0,1]")
print(f"Grid size: {nx} x {ny}")
print(f"Coefficients: b = ({b_x}, {b_y}), c = {c}")
print(f"Relative L2 error: {l2_rel:.3e}")
print(f"Max error (L-inf): {linf:.3e}")
print(f"U_min: {np.min(np.min(u, axis = 1), axis = 0)}, U_max: {np.max(np.max(u, axis = 1), axis = 0)}")
print("Figure saved as ex1.png")
