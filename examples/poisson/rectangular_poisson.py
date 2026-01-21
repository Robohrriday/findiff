from findiff import FinDiff, PDE, BoundaryConditions
import numpy as np
import matplotlib.pyplot as plt


# Define rectangular domain (0,2)x(0,1)
nx, ny = 100, 50
x = np.linspace(0, 2, nx)
y = np.linspace(0, 1, ny)
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing='ij')

# Laplacian operator
L = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)

# Source term for Poisson equation (can be modified as needed)
# For example, f = 0 gives us Laplace's equation
f = np.zeros((nx, ny))

# Boundary conditions: u = sin(pi*x)*sin(pi*y) on all boundaries
bc = BoundaryConditions((nx, ny))

# Left boundary (x=0): u = sin(pi*0)*sin(pi*y) = 0
bc[0, :] = 0

# Right boundary (x=2): u = sin(pi*2)*sin(pi*y) = 0
bc[-1, :] = 0

# Bottom boundary (y=0): u = sin(pi*x)*sin(pi*0) = 0
bc[:, 0] = 0

# Top boundary (y=1): u = sin(pi*x)*sin(pi*1) = 0 (since sin(pi) = 0)
bc[:, -1] = 0

# Note: With homogeneous boundary conditions and f=0, the solution will be trivial (u=0)
# To get a non-trivial solution, we need a non-zero source term f
# Let's add a source term
f = - 2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

# Create and solve the PDE
pde = PDE(L, f, bc)
u = pde.solve()

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot solution
im1 = ax1.contourf(X, Y, u, levels=20, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Solution u(x,y)')
ax1.set_aspect('equal')
plt.colorbar(im1, ax=ax1)

# Plot source term
im2 = ax2.contourf(X, Y, f, levels=20, cmap='viridis')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Source term f(x,y)')
ax2.set_aspect('equal')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig('rectangular_poisson_solution.png', dpi=150, bbox_inches='tight')
plt.show()

# Print some statistics
print(f"Domain: x ∈ [0, 2], y ∈ [0, 1]")
print(f"Grid size: {nx} × {ny}")
print(f"Solution range: [{u.min():.6f}, {u.max():.6f}]")
