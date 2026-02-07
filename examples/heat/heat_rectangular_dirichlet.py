import numpy as np
import matplotlib.pyplot as plt

from findiff import FinDiff


def main():
	# Physical and numerical parameters
	kappa = 1.0  # thermal diffusivity
	nx, ny = 81, 81
	x = np.linspace(0.0, 1.0, nx)
	y = np.linspace(0.0, 1.0, ny)
	dx, dy = x[1] - x[0], y[1] - y[0]

	# Explicit FTCS stability for diffusion:
	# dt <= 1 / (2*kappa*(1/dx^2 + 1/dy^2))
	# For dx = dy = h this reduces to dt <= h^2 / (4*kappa).
	dt_stable = 1.0 / (2.0 * kappa * (1.0 / dx**2 + 1.0 / dy**2))
	dt = 0.8 * dt_stable  # a bit below the limit for safety
	t_final = 0.2
	n_steps = int(np.ceil(t_final / dt))
	dt = t_final / n_steps  # snap dt to land exactly on t_final

	X, Y = np.meshgrid(x, y, indexing="ij")

	# Initial condition: smooth bump satisfying Dirichlet zeros on boundary
	u = np.sin(np.pi * X) * np.sin(np.pi * Y)

	# Dirichlet boundary values (here zero)
	u_bc = np.zeros_like(u)

	# Spatial operators
	lap = FinDiff(0, dx, 2) + FinDiff(1, dy, 2)

	times = [0.0]
	energies = [np.linalg.norm(u)]
	snapshots = []
	snapshot_times = []
	snapshot_every = max(1, n_steps // 100)

	for step in range(n_steps):
		# Impose Dirichlet boundaries
		u[0, :] = u_bc[0, :]
		u[-1, :] = u_bc[-1, :]
		u[:, 0] = u_bc[:, 0]
		u[:, -1] = u_bc[:, -1]

		# Explicit FTCS update: u^{n+1} = u^n + dt * kappa * Lap u^n
		u_new = u + dt * kappa * lap(u)

		u = u_new
		times.append((step + 1) * dt)
		energies.append(np.linalg.norm(u))

		# store snapshots for animation
		if (step + 1) % snapshot_every == 0 or step == n_steps - 1:
			snapshots.append(u.copy())
			snapshot_times.append((step + 1) * dt)

	# Exact solution for this IC: u(x,y,t) = exp(-2 pi^2 kappa t) sin(pi x) sin(pi y)
	u_exact = np.exp(-2 * np.pi**2 * kappa * t_final) * np.sin(np.pi * X) * np.sin(np.pi * Y)

	error = u - u_exact
	l2_rel = np.linalg.norm(error) / np.linalg.norm(u_exact)
	linf = np.max(np.abs(error))

	fig, axes = plt.subplots(1, 3, figsize=(15, 4))
	im0 = axes[0].contourf(X, Y, u, levels=30, cmap="inferno")
	axes[0].set_title("Numerical u(x,y,t_final)")
	axes[0].set_xlabel("x")
	axes[0].set_ylabel("y")
	axes[0].set_aspect("equal")
	plt.colorbar(im0, ax=axes[0])

	im1 = axes[1].contourf(X, Y, u_exact, levels=30, cmap="inferno")
	axes[1].set_title("Exact u(x,y,t_final)")
	axes[1].set_xlabel("x")
	axes[1].set_ylabel("y")
	axes[1].set_aspect("equal")
	plt.colorbar(im1, ax=axes[1])

	im2 = axes[2].contourf(X, Y, error, levels=30, cmap="RdBu_r")
	axes[2].set_title("Error (num - exact)")
	axes[2].set_xlabel("x")
	axes[2].set_ylabel("y")
	axes[2].set_aspect("equal")
	plt.colorbar(im2, ax=axes[2])

	plt.tight_layout()
	plt.savefig("heat_rectangular_dirichlet.png", dpi=150, bbox_inches="tight")
	plt.show()

	# Energy decay plot
	plt.figure(figsize=(5, 3))
	plt.plot(times, energies, label="||u||_2")
	plt.xlabel("time")
	plt.ylabel("L2 norm")
	plt.title("Energy decay")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig("heat_rectangular_dirichlet_energy.png", dpi=150, bbox_inches="tight")
	plt.show()

	# Animation of solution evolution (use imshow for stable blitting)
	from matplotlib import animation

	fig_anim, ax_anim = plt.subplots(figsize=(5, 4))
	im = ax_anim.imshow(
		snapshots[0],
		extent=(0, 1, 0, 1),
		origin="lower",
		cmap="inferno",
		aspect="auto",
		vmin=min(np.min(s) for s in snapshots),
		vmax=max(np.max(s) for s in snapshots),
	)
	cb = plt.colorbar(im, ax=ax_anim)
	text_t = ax_anim.text(0.02, 1.02, f"t={snapshot_times[0]:.3f}", transform=ax_anim.transAxes)
	ax_anim.set_xlabel("x")
	ax_anim.set_ylabel("y")
	ax_anim.set_title("Heat equation evolution")

	def animate(i):
		im.set_data(snapshots[i])
		text_t.set_text(f"t={snapshot_times[i]:.3f}")
		return [im, text_t]

	anim = animation.FuncAnimation(
		fig_anim,
		animate,
		frames=len(snapshots),
		interval=80,
		blit=True,
		repeat=False,
	)

	anim.save("heat_rectangular_dirichlet_anim.mp4", writer="ffmpeg", dpi=120)
	plt.close(fig_anim)

	print("=== Heat equation (Dirichlet) ===")
	print(f"Grid: {nx} x {ny}, dx={dx:.4f}, dy={dy:.4f}")
	print(f"kappa={kappa}, dt={dt:.4e}, steps={n_steps}, CFL limit={dt_stable:.4e}")
	print(f"t_final={t_final}")
	print(f"Relative L2 error: {l2_rel:.3e}")
	print(f"Max error (L-inf): {linf:.3e}")
	print("Saved: heat_rectangular_dirichlet.png, heat_rectangular_dirichlet_energy.png")


if __name__ == "__main__":
	main()
