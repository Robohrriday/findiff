import numpy as np
import matplotlib.pyplot as plt

from findiff import FinDiff, PDE, BoundaryConditions


def main():
    # Plate domain [0, 1] x [0, 1]
    nx, ny = 120, 120
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Operator and source term
    L = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)
    f = np.zeros_like(X)

    # Boundary conditions from README heat-conduction example
    bc = BoundaryConditions((nx, ny))

    # Dirichlet: u(x,0) = 300
    bc_dirichlet_y0 = np.zeros_like(X)
    bc_dirichlet_y0[:, 0] = 300.0
    bc[:, 0] = bc_dirichlet_y0

    # Dirichlet: u(1,y) = 300 - 200*y
    bc_dirichlet_x1 = np.zeros_like(X)
    bc_dirichlet_x1[-1, :] = 300.0 - 200.0 * Y[-1, :]
    bc[-1, :] = bc_dirichlet_x1

    # Neumann: du/dx = 0 at x = 0 (outward normal n = (-1, 0))
    d_dx = FinDiff(0, dx, 1)
    bc_neumann_x0 = np.zeros_like(X)
    bc_neumann_x0[0, :] = 0.0
    bc[0, :] = (d_dx, bc_neumann_x0)

    # Neumann: du/dy = 0 at y = 1 (outward normal n = (0, 1))
    d_dy = FinDiff(1, dy, 1)
    bc_neumann_y1 = np.zeros_like(X)
    bc_neumann_y1[:, -1] = 0.0
    bc[:, -1] = (d_dy, bc_neumann_y1)

    # Solve PDE
    pde = PDE(L, f, bc)
    u = pde.solve()

    # Visualization
    fig, ax = plt.subplots(figsize=(6, 5))
    levels = 30
    cf = ax.contourf(X, Y, u, levels=levels, cmap="inferno")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Steady heat conduction")
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, label="Temperature")

    plt.tight_layout()
    plt.savefig("heat_conduction_plate.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("=== Heat conduction example ===")
    print(f"Grid size: {nx} x {ny}")
    print("Dirichlet: u(x,0)=300, u(1,y)=300-200y")
    print("Neumann: du/dx=0 at x=0, du/dy=0 at y=1")
    print(f"Temperature range: [{u.min():.3f}, {u.max():.3f}]")
    print("Figure saved as heat_conduction_plate.png")


if __name__ == "__main__":
    main()
