import orbax.checkpoint as ocp
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import gridspec

"""
Plot checkpointed data.

Philip Mocz (2025), @pmocz
Flatiron Institute
"""

path = os.path.join(os.path.dirname(__file__), "checkpoints")
async_checkpoint_manager = ocp.CheckpointManager(path)


def main():
    all_vSq = np.array([])
    vSq_t = np.array([])
    vx0 = np.array([])
    vy0 = np.array([])
    grid_i = 10
    grid_j = 10

    for i in range(grid_i):
        row_vSq = np.array([])
        for j in range(grid_j):
            # Load the checkpoint
            restored = async_checkpoint_manager.restore(i * grid_j + j)
            vx, vy, vz = restored
            N = vz.shape[0]
            vSq = vx**2 + vy**2 + vz**2
            vSq_mean = np.mean(vSq)
            vSq_t = np.hstack((vSq_t, vSq_mean))
            print(vSq_mean / vSq_t[0])
            vSq = vSq[:, :, N // 2]  # tavSq a slice
            if row_vSq.size == 0:
                row_vSq = vSq
            else:
                row_vSq = np.hstack((row_vSq, vSq))
            if i == 0 and j == 0:
                vx0 = vx
                vy0 = vy
        if all_vSq.size == 0:
            all_vSq = row_vSq
        else:
            all_vSq = np.vstack((all_vSq, row_vSq))

    # Create a gridspec layout: first subplot (all_vSq) on the left (half), other two stacked on the right (half)

    fig = plt.figure(figsize=(21, 11), dpi=80)
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

    # imshow all_vSq (left, spans both rows)
    ax1 = fig.add_subplot(gs[:, 0])
    im = ax1.imshow(
        all_vSq,
        cmap="bwr",
        origin="lower",
        extent=(0, grid_i, 0, grid_j),
    )
    fig.colorbar(im, ax=ax1, label=r"$v^2$")
    ax1.set_aspect("equal")
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title(r"$v^2$ (slice)")

    # plot vSq as function of time (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(vSq_t / vSq_t[0])
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("checkpoint")
    ax2.set_ylabel(r"$v^2$")

    # quiver plot of vx, vy at the center slice (bottom right)
    ax3 = fig.add_subplot(gs[1, 1])
    N = vx0.shape[0]
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    ax3.quiver(X, Y, vx0[:, :, N // 2], vy0[:, :, N // 2])
    ax3.set_aspect("equal")
    ax3.set_title(r"initial $v_x, v_y$ (slice)")
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
