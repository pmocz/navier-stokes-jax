import jax
import orbax.checkpoint as ocp
import numpy as np
import matplotlib.pyplot as plt
import os

"""
Plot checkpointed data.

Philip Mocz (2025), @pmocz
Flatiron Institute
"""

##############
# Checkpointer
path = os.path.join(os.path.dirname(__file__), "checkpoints")
async_checkpoint_manager = ocp.CheckpointManager(path)


def main():
    all_ke = np.array([])
    ke_t = np.array([])
    grid_i = 10
    grid_j = 10

    for i in range(grid_i):
        row_ke = np.array([])
        for j in range(grid_j):
            # Load the checkpoint
            restored = async_checkpoint_manager.restore(i * grid_j + j)
            vx, vy, vz = restored
            N = vz.shape[0]
            ke = 0.5 * (vx**2 + vy**2 + vz**2)
            ke_mean = np.mean(ke)
            ke_t = np.hstack((ke_t, ke_mean))
            print(ke_mean)
            ke = ke[:, :, N // 2]  # take a slice
            if row_ke.size == 0:
                row_ke = ke
            else:
                row_ke = np.hstack((row_ke, ke))
        if all_ke.size == 0:
            all_ke = row_ke
        else:
            all_ke = np.vstack((all_ke, row_ke))

    # imshow all_ke
    fig = plt.figure(figsize=(16, 8), dpi=80)
    ax = fig.add_subplot(111)
    plt.imshow(
        all_ke,
        cmap="bwr",
        origin="lower",
        extent=(0, grid_i, 0, grid_j),
    )
    plt.colorbar(label="ke")
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()

    # plot ke as function of time
    fig = plt.figure()
    plt.plot(ke_t)
    plt.show()


if __name__ == "__main__":
    main()
