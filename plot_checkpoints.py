import orbax.checkpoint as ocp
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib import gridspec
import argparse

"""
Plot checkpointed data.

Philip Mocz (2025), @pmocz
Flatiron Institute

Example Usage:

python plot_checkpoints.py --path=checkpoints32

python plot_checkpoints.py --path=checkpoints64_init

"""

parser = argparse.ArgumentParser(description="Plot checkpointed data.")
parser.add_argument(
    "--path",
    type=str,
    default="checkpoints32_init",
    help="Path to the checkpoint directory (default: checkpoints32_init)",
)
args = parser.parse_args()

path = os.path.join(os.path.dirname(__file__), args.path)
async_checkpoint_manager = ocp.CheckpointManager(path)


def curl(vx, vy, vz, kx, ky, kz):
    """return curl of (vx,vy,vz) as (wx, wy, wz)"""
    # wx = dvy/dz - dvz/dy
    # wy = dvz/dx - dvx/dz
    # wz = dvx/dy - dvy/dx
    dvy_z = np.real(fft.ifftn(1j * kz * fft.fftn(vy)))
    dvz_y = np.real(fft.ifftn(1j * ky * fft.fftn(vz)))
    dvz_x = np.real(fft.ifftn(1j * kx * fft.fftn(vz)))
    dvx_z = np.real(fft.ifftn(1j * kz * fft.fftn(vx)))
    dvx_y = np.real(fft.ifftn(1j * ky * fft.fftn(vx)))
    dvy_x = np.real(fft.ifftn(1j * kx * fft.fftn(vy)))
    wx = dvy_z - dvz_y
    wy = dvz_x - dvx_z
    wz = dvx_y - dvy_x
    return wx, wy, wz


def main():
    grid_vSq = np.array([])
    timeseries_vSq = np.array([])
    vx0 = np.array([])
    vy0 = np.array([])
    grid_i = 10
    grid_j = 10
    snap_total = grid_i * grid_j
    vx_all = {}
    vy_all = {}
    vz_all = {}

    for i in range(grid_i):
        row_vSq = np.array([])
        for j in range(grid_j):
            # Load the checkpoint
            snap_num = i * grid_j + j
            restored = async_checkpoint_manager.restore(snap_num)
            vx, vy, vz = restored
            N = vz.shape[0]
            vx_all[snap_num] = vx
            vy_all[snap_num] = vy
            vz_all[snap_num] = vz
            vSq = vx**2 + vy**2 + vz**2
            vSq_mean = np.mean(vSq)
            timeseries_vSq = np.hstack((timeseries_vSq, vSq_mean))
            print(vSq_mean / timeseries_vSq[0])
            vSq = vSq[:, :, N // 2]  # tavSq a slice
            if row_vSq.size == 0:
                row_vSq = vSq
            else:
                row_vSq = np.hstack((row_vSq, vSq))
            if i == 0 and j == 0:
                vx0 = vx
                vy0 = vy
        if grid_vSq.size == 0:
            grid_vSq = row_vSq
        else:
            grid_vSq = np.vstack((grid_vSq, row_vSq))

    # Create a gridspec layout: first subplot (grid_vSq) on the left (half), other two stacked on the right (half)

    fig = plt.figure(figsize=(21, 11), dpi=80)
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

    # imshow grid_vSq (left, spans both rows)
    ax1 = fig.add_subplot(gs[:, 0])
    im = ax1.imshow(
        grid_vSq,
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
    ax2.plot(timeseries_vSq / timeseries_vSq[0])
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

    # Now, plot wz for each snapshot, as an animation
    N = vz.shape[0]
    L = 1.0
    klin = 2.0 * np.pi / L * np.arange(-N / 2, N / 2)
    kx, ky, kz = np.meshgrid(klin, klin, klin, indexing="ij")
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)
    kz = np.fft.ifftshift(kz)

    fig, ax = plt.subplots()
    # Compute global min/max for color scale
    wz_lim = 0.0
    for frame in range(snap_total):
        vx = vx_all[frame]
        vy = vy_all[frame]
        vz = vz_all[frame]
        _, _, wz = curl(vx, vy, vz, kx, ky, kz)
        slice_wz = wz[:, :, N // 2]
        wz_lim = max(wz_lim, np.max(np.abs(slice_wz)))
    im = ax.imshow(
        np.zeros((N, N)), cmap="bwr", origin="lower", vmin=-wz_lim, vmax=wz_lim
    )

    def update(frame):
        vx = vx_all[frame]
        vy = vy_all[frame]
        vz = vz_all[frame]
        _, _, wz = curl(vx, vy, vz, kx, ky, kz)
        im.set_data(wz[:, :, N // 2])
        return [im]

    global ani  # Prevent garbage collection
    ani = animation.FuncAnimation(
        fig, update, frames=snap_total, interval=100, blit=True, repeat=True
    )
    plt.show()


if __name__ == "__main__":
    main()
