import numpy as np
import matplotlib.pyplot as plt

from findiff import FinDiff, PDE, BoundaryConditions, Identity, Coefficient


def main():
    # Rectangular domain [0, 1] x [0, 1]
    nx, ny = 100, 80
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Coefficients for -Delta u + b · grad u + c u = f
    b_x, b_y = 0.5, -0.2
    c = 1.5

    # Manufactured solution and derived terms
    u_exact = np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
    du_dx = np.pi * np.cos(np.pi * X) * np.sin(2 * np.pi * Y)
    du_dy = 2 * np.pi * np.sin(np.pi * X) * np.cos(2 * np.pi * Y)
    laplace_u = - (np.pi**2 + (2 * np.pi) ** 2) * u_exact

    f = -laplace_u + b_x * du_dx + b_y * du_dy + c * u_exact

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
    L = Coefficient(-1.0) * diffusion + convection + Coefficient(c) * Identity()

    pde = PDE(L, f, bc)
    u = pde.solve()

    error = u - u_exact
    l2_rel = np.linalg.norm(error) / np.linalg.norm(u_exact)
    linf = np.abs(error).max()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].contourf(X, Y, u, levels=24, cmap="viridis")
    axes[0].set_title("Numerical solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X, Y, u_exact, levels=24, cmap="viridis")
    axes[1].set_title("Exact solution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(X, Y, error, levels=24, cmap="RdBu_r")
    axes[2].set_title("Error (u - u_exact)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig("rectangular_convection_diffusion_dirichlet.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("=== Convection-diffusion (Dirichlet) ===")
    print(f"Domain: x in [0,1], y in [0,1]")
    print(f"Grid size: {nx} x {ny}")
    print(f"Coefficients: b = ({b_x}, {b_y}), c = {c}")
    print(f"Relative L2 error: {l2_rel:.3e}")
    print(f"Max error (L-inf): {linf:.3e}")
    print("Figure saved as rectangular_convection_diffusion_dirichlet.png")


if __name__ == "__main__":
    main()
