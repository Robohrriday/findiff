from findiff import FinDiff, PDE, BoundaryConditions, Coefficient
import numpy as np
import matplotlib.pyplot as plt


# Define circular disc domain in polar coordinates
# r ∈ [0, 1], θ ∈ [0, 2π)
nr, ntheta = 40, 80
r = np.linspace(0.01, 1, nr)
theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
dr, dtheta = r[1] - r[0], theta[1] - theta[0]
R, THETA = np.meshgrid(r, theta, indexing='ij')


# u_boundary(r=1, θ) = sin(3θ) + cos(2θ) + 1
u_boundary = np.sin(3 * THETA[-1, :]) + np.cos(2 * THETA[-1, :]) + 1
f = R * np.sin(THETA)

# Set up boundary conditions
bc = BoundaryConditions((nr, ntheta))
# Outer boundary (r = 1): Dirichlet BC with non-trivial function
bc_dirichlet = np.zeros((nr, ntheta))
bc_dirichlet[-1, :] = u_boundary
bc[-1, :] = bc_dirichlet

# Inner boundary (r ≈ 0): Use homogeneous condition or symmetry
bc[0, :] = 0

# Construct the Laplacian operator with polar coordinate terms
# L = ∂²u/∂r² + (1/r)∂u/∂r + (1/r²)∂²u/∂θ²
# Use periodic differences in θ to respect the 0↔2π continuity
L = (
	FinDiff(0, dr, 2)
	+ Coefficient(1.0 / R) * FinDiff(0, dr, 1)
	+ Coefficient(1.0 / (R ** 2)) * FinDiff(1, dtheta, 2, periodic=True)
)

# Create and solve the PDE
pde = PDE(L, f, bc)
u = pde.solve()

# Convert back to Cartesian for visualization
x_cart = R * np.cos(THETA)
y_cart = R * np.sin(THETA)

# Create visualizations
fig = plt.figure(figsize=(15, 5))

# Plot 1: Solution in polar coordinates
ax1 = fig.add_subplot(131)
im1 = ax1.contourf(THETA, R, u, levels=20, cmap='viridis')
ax1.set_xlabel('θ (radians)')
ax1.set_ylabel('r')
ax1.set_title('Solution u(r,θ) in Polar Coordinates')
plt.colorbar(im1, ax=ax1)

# Plot 2: Solution in Cartesian coordinates (on disc)
ax2 = fig.add_subplot(132)
im2 = ax2.contourf(x_cart, y_cart, u, levels=20, cmap='viridis')
circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
ax2.add_patch(circle)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Solution u(x,y) on Circular Disc')
ax2.set_aspect('equal')
plt.colorbar(im2, ax=ax2)

# Plot 3: Boundary condition (non-trivial)
ax3 = fig.add_subplot(133)
ax3.plot(theta, u_boundary, 'b-', linewidth=2)
ax3.set_xlabel('θ (radians)')
ax3.set_ylabel('u(r=1, θ)')
ax3.set_title('Non-trivial Dirichlet BC at r=1')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('circular_disc_poisson_solution.png', dpi=150, bbox_inches='tight')
print("Plot saved as circular_disc_poisson_solution.png")
plt.show()

# Print statistics
print(f"\n=== Circular Disc Poisson Equation ===")
print(f"Domain: r ∈ [0, 1], θ ∈ [0, 2π)")
print(f"Grid size: {nr} × {ntheta}")
print(f"Solution range: [{u.min():.6f}, {u.max():.6f}]")
print(f"Boundary condition at r=1: u(1,θ) = sin(3θ) + cos(2θ) + 1")
print(f"Source term: f(r,θ) = r·sin(θ)")
print(f"\nBoundary values statistics:")
print(f"  Min: {u_boundary.min():.6f}")
print(f"  Max: {u_boundary.max():.6f}")
print(f"  Mean: {u_boundary.mean():.6f}")
