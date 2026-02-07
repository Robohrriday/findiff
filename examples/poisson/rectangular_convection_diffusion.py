import numpy as np
import matplotlib.pyplot as plt

from findiff import FinDiff, PDE, BoundaryConditions, Identity, Coefficient


def main():
    # Rectangular domain [0, 1] x [0, 1]
    nx, ny = 80, 80
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Coefficients in -Delta u + b dot grad u + c u = f
    b_x, b_y = -1.0, 0.5
    c = 2.0

    # Manufactured smooth solution to enforce mixed BCs
    u_exact = np.exp(X) * np.sin(2 * np.pi * Y)
    du_dx = np.exp(X) * np.sin(2 * np.pi * Y)
    du_dy = 2 * np.pi * np.exp(X) * np.cos(2 * np.pi * Y)
    laplace_u = (1 - (2 * np.pi) ** 2) * u_exact
    f = -laplace_u + b_x * du_dx + b_y * du_dy + c * u_exact

    # Build mixed boundary conditions:
    # - Neumann on x = 0 and y = 1
    # - Dirichlet on x = 1 and y = 0
    bc = BoundaryConditions((nx, ny))

    # Dirichlet on x = 1
    bc_dirichlet_right = np.zeros_like(u_exact)
    bc_dirichlet_right[-1, :] = u_exact[-1, :]
    bc[-1, :] = bc_dirichlet_right

    # Dirichlet on y = 0
    bc_dirichlet_bottom = np.zeros_like(u_exact)
    bc_dirichlet_bottom[:, 0] = u_exact[:, 0]
    bc[:, 0] = bc_dirichlet_bottom

    # Neumann on x = 0 (outward normal n = (-1, 0)): du/dn = -du/dx
    d_dx = FinDiff(0, dx, 1)
    bc_neumann_x0 = np.zeros_like(u_exact)
    bc_neumann_x0[0, 1:-1] = du_dx[0, 1:-1]
    bc[0, 1:-1] = (d_dx, bc_neumann_x0)

    # Neumann on y = 1 (outward normal n = (0, 1)): du/dn = du/dy
    d_dy = FinDiff(1, dy, 1)
    bc_neumann_y1 = np.zeros_like(u_exact)
    bc_neumann_y1[1:-1, -1] = du_dy[1:-1, -1]
    bc[1:-1, -1] = (d_dy, bc_neumann_y1)

    # Anchor the top-left corner to the exact value to avoid a mixed corner
    bc[0, -1] = u_exact[0, -1]

    # Discrete operator for -Delta u + b dot grad u + c u
    diffusion = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)
    convection = Coefficient(b_x) * FinDiff(0, dx, 1) + Coefficient(b_y) * FinDiff(1, dy, 1)
    L = Coefficient(-1.0) * diffusion + convection + Coefficient(c) * Identity()

    pde = PDE(L, f, bc)
    u = pde.solve()

    error = u - u_exact
    l2_rel = np.linalg.norm(error) / np.linalg.norm(u_exact)
    linf = np.abs(error).max()

    # Diagnostics: compare imposed Neumann data with numerical normal derivatives
    du_dx_num = d_dx(u)
    du_dy_num = d_dy(u)
    neumann_x0_exact = -du_dx[0, 1:-1]
    neumann_x0_num = -du_dx_num[0, 1:-1]
    neumann_y1_exact = du_dy[1:-1, -1]
    neumann_y1_num = du_dy_num[1:-1, -1]

    max_rel_x0 = np.linalg.norm(neumann_x0_num - neumann_x0_exact) / np.linalg.norm(neumann_x0_exact)
    max_rel_y1 = np.linalg.norm(neumann_y1_num - neumann_y1_exact) / np.linalg.norm(neumann_y1_exact)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].contourf(X, Y, u, levels=20, cmap="viridis")
    axes[0].set_title("Numerical solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].contourf(X, Y, u_exact, levels=20, cmap="viridis")
    axes[1].set_title("Exact solution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].contourf(X, Y, error, levels=20, cmap="RdBu_r")
    axes[2].set_title("Error (u - u_exact)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig("rectangular_convection_diffusion.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Plot Neumann boundary comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    axes2[0].plot(y[1:-1], neumann_x0_exact, label="exact -du/dx|x=0")
    axes2[0].plot(y[1:-1], neumann_x0_num, "--", label="numerical -du/dx|x=0")
    axes2[0].set_xlabel("y")
    axes2[0].set_ylabel("normal derivative")
    axes2[0].set_title("Neumann boundary x=0")
    axes2[0].legend()

    axes2[1].plot(x[1:-1], neumann_y1_exact, label="exact du/dy|y=1")
    axes2[1].plot(x[1:-1], neumann_y1_num, "--", label="numerical du/dy|y=1")
    axes2[1].set_xlabel("x")
    axes2[1].set_ylabel("normal derivative")
    axes2[1].set_title("Neumann boundary y=1")
    axes2[1].legend()

    plt.tight_layout()
    plt.savefig("rectangular_convection_diffusion_neumann_check.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("=== Convection-diffusion on rectangle ===")
    print(f"Domain: x in [0,1], y in [0,1]")
    print(f"Grid size: {nx} x {ny}")
    print(f"Coefficients: b = ({b_x}, {b_y}), c = {c}")
    print(f"Relative L2 error: {l2_rel:.3e}")
    print(f"Max error (L-inf): {linf:.3e}")
    print(f"Neumann rel error x=0: {max_rel_x0:.3e}")
    print(f"Neumann rel error y=1: {max_rel_y1:.3e}")
    print("Figure saved as rectangular_convection_diffusion.png")


if __name__ == "__main__":
    main()
