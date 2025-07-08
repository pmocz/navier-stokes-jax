import jax
import time
import jax.numpy as jnp
import jax.numpy.fft as jfft
from functools import partial
import chex
from typing import NamedTuple
import optax
import os
import argparse
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

"""
Philip Mocz (2025)

Simulate the 3D Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

Example Usage:

1. Run the simulation with default parameters (N=32, optimize=True):
python navier-stokes-jax.py

2. Run the simulation with a specified grid size (N=64), no optimization:
python navier-stokes-jax.py --N 64 --no-optimize

"""


def poisson_solve(rho, kSq_inv):
    """solve the Poisson equation, given source field rho"""
    V_hat = -(jfft.fftn(rho)) * kSq_inv
    V = jnp.real(jfft.ifftn(V_hat))
    return V


def diffusion_solve(v, dt, nu, kSq):
    """solve the diffusion equation over a timestep dt, given viscosity nu"""
    v_hat = (jfft.fftn(v)) / (1.0 + dt * nu * kSq)
    v = jnp.real(jfft.ifftn(v_hat))
    return v


def grad(v, kx, ky, kz):
    """return gradient of v"""
    v_hat = jfft.fftn(v)
    dvx = jnp.real(jfft.ifftn(1j * kx * v_hat))
    dvy = jnp.real(jfft.ifftn(1j * ky * v_hat))
    dvz = jnp.real(jfft.ifftn(1j * kz * v_hat))
    return dvx, dvy, dvz


def div(vx, vy, vz, kx, ky, kz):
    """return divergence of (vx,vy,vz)"""
    dvx_x = jnp.real(jfft.ifftn(1j * kx * jfft.fftn(vx)))
    dvy_y = jnp.real(jfft.ifftn(1j * ky * jfft.fftn(vy)))
    dvz_z = jnp.real(jfft.ifftn(1j * kz * jfft.fftn(vz)))
    return dvx_x + dvy_y + dvz_z


def curl(vx, vy, vz, kx, ky, kz):
    """return curl of (vx,vy,vz) as (wx, wy, wz)"""
    # wx = dvy/dz - dvz/dy
    # wy = dvz/dx - dvx/dz
    # wz = dvx/dy - dvy/dx
    dvy_z = jnp.real(jfft.ifftn(1j * kz * jfft.fftn(vy)))
    dvz_y = jnp.real(jfft.ifftn(1j * ky * jfft.fftn(vz)))
    dvz_x = jnp.real(jfft.ifftn(1j * kx * jfft.fftn(vz)))
    dvx_z = jnp.real(jfft.ifftn(1j * kz * jfft.fftn(vx)))
    dvx_y = jnp.real(jfft.ifftn(1j * ky * jfft.fftn(vx)))
    dvy_x = jnp.real(jfft.ifftn(1j * kx * jfft.fftn(vy)))
    wx = dvy_z - dvz_y
    wy = dvz_x - dvx_z
    wz = dvx_y - dvy_x
    return wx, wy, wz


def get_ke(vx, vy, vz, dV):
    """Calculate the kinetic energy in the system = 0.5 * integral |v|^2 dV"""
    v2 = vx**2 + vy**2 + vz**2
    ke = 0.5 * jnp.sum(v2) * dV
    return ke


def apply_dealias(f, dealias):
    """apply 2/3 rule dealias to field f"""
    f_hat = dealias * jfft.fftn(f)
    return jnp.real(jfft.ifftn(f_hat))


@partial(jax.jit, static_argnames=["dt", "Nt", "nu"])
def run_simulation(vx, vy, vz, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias):
    """Run the full Navier-Stokes simulation"""

    def update(_, state):
        (vx, vy, vz) = state

        # Advection: rhs = -(v.grad)v
        dvx_x, dvx_y, dvx_z = grad(vx, kx, ky, kz)
        dvy_x, dvy_y, dvy_z = grad(vy, kx, ky, kz)
        dvz_x, dvz_y, dvz_z = grad(vz, kx, ky, kz)

        rhs_x = -(vx * dvx_x + vy * dvx_y + vz * dvx_z)
        rhs_y = -(vx * dvy_x + vy * dvy_y + vz * dvy_z)
        rhs_z = -(vx * dvz_x + vy * dvz_y + vz * dvz_z)

        rhs_x = apply_dealias(rhs_x, dealias)
        rhs_y = apply_dealias(rhs_y, dealias)
        rhs_z = apply_dealias(rhs_z, dealias)

        vx += dt * rhs_x
        vy += dt * rhs_y
        vz += dt * rhs_z

        # Poisson solve for pressure
        div_rhs = div(rhs_x, rhs_y, rhs_z, kx, ky, kz)
        P = poisson_solve(div_rhs, kSq_inv)
        dPx, dPy, dPz = grad(P, kx, ky, kz)

        # Correction (to eliminate divergence component of velocity)
        vx -= dt * dPx
        vy -= dt * dPy
        vz -= dt * dPz

        # Diffusion solve (implicit)
        vx = diffusion_solve(vx, dt, nu, kSq)
        vy = diffusion_solve(vy, dt, nu, kSq)
        vz = diffusion_solve(vz, dt, nu, kSq)

        return (vx, vy, vz)

    (vx, vy, vz) = jax.lax.fori_loop(0, Nt, update, (vx, vy, vz))

    return vx, vy, vz


def run_simulation_and_save_checkpoints(
    vx, vy, vz, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias, folder_name
):
    """Run the full Navier-Stokes simulation and save 100 checkpoints"""

    path = ocp.test_utils.erase_and_create_empty(os.getcwd() + "/" + folder_name)
    async_checkpoint_manager = ocp.CheckpointManager(path)

    num_checkpoints = 100
    snap_interval = max(1, Nt // num_checkpoints)
    checkpoint_id = 0
    for i in range(0, Nt, snap_interval):
        steps = min(snap_interval, Nt - i)
        vx, vy, vz = run_simulation(
            vx, vy, vz, dt, steps, nu, kx, ky, kz, kSq, kSq_inv, dealias
        )
        state = vx, vy, vz
        async_checkpoint_manager.save(checkpoint_id, args=ocp.args.StandardSave(state))
        async_checkpoint_manager.wait_until_finished()
        checkpoint_id += 1

    return vx, vy, vz


class InfoState(NamedTuple):
    iter_num: chex.Numeric


def print_info():
    def init_fn(params):
        del params
        return InfoState(iter_num=0)

    def update_fn(updates, state, params, *, value, grad, **extra_args):
        del params, extra_args

        jax.debug.print(
            "Iteration: {i}, Boost: {v:.2e}, |grad|: {e:.2e}",
            i=state.iter_num,
            v=-value,
            e=optax.tree_utils.tree_l2_norm(grad),
        )
        return updates, InfoState(iter_num=state.iter_num + 1)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def run_opt(init_params, fun, opt, max_iter, tol):
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = optax.tree_utils.tree_get(state, "count")
        grad = optax.tree_utils.tree_get(state, "grad")
        err = optax.tree_utils.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state


def maximize_ke_boost(Ax, Ay, Az, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias, dx):
    """Optimize the initial velocity field maximize the kinetic energy boost"""

    # @jax.jit
    def loss_function(theta):
        Ax, Ay, Az = theta
        Ax -= jnp.mean(Ax)
        Ay -= jnp.mean(Ay)
        Az -= jnp.mean(Az)
        vx, vy, vz = curl(Ax, Ay, Az, kx, ky, kz)
        vx -= jnp.mean(vx)
        vy -= jnp.mean(vy)
        vz -= jnp.mean(vz)

        ke_init = get_ke(vx, vy, vz, dx**3)

        vx, vy, vz = run_simulation(
            vx, vy, vz, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias
        )

        ke_final = get_ke(vx, vy, vz, dx**3)
        ke_boost = ke_final / ke_init

        return -ke_boost

    # opt = optax.lbfgs()
    opt = optax.chain(print_info(), optax.lbfgs())
    init_params = (Ax, Ay, Az)
    print(
        f"Initial value: {-loss_function(init_params):.2e} "
        f"Initial gradient norm: {optax.tree_utils.tree_l2_norm(jax.grad(loss_function)(init_params)):.2e}"
    )
    max_iter = 15  # XXX 100
    final_params, _ = run_opt(
        init_params, loss_function, opt, max_iter=max_iter, tol=1e-3
    )
    print(
        f"Final value: {-loss_function(final_params):.2e}, "
        f"Final gradient norm: {optax.tree_utils.tree_l2_norm(jax.grad(loss_function)(final_params)):.2e}"
    )

    return final_params


def make_plot(vx, vy, vz, kx, ky, kz, N, max_plot_val):
    """Plot the z-component of vorticity in the mid-plane"""
    # _, _, wz = curl(vx, vy, vz, kx, ky, kz)
    ke = 0.5 * (vx**2 + vy**2 + vz**2)
    plt.cla()
    # plt.imshow(jax.device_get(wz[:, :, N // 2]), cmap="RdBu")
    # plt.clim(-20, 20)
    ke_slice = jax.device_get(ke[:, :, N // 2])
    plt.imshow(ke_slice, cmap="RdBu")
    plt.clim(0, max_plot_val)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect("equal")
    plt.show()


def main():
    """3D Navier-Stokes Simulation"""

    print(jax.devices())

    # Simulation parameters
    parser = argparse.ArgumentParser(description="3D Navier-Stokes Simulation")
    parser.add_argument("--N", type=int, default=32, help="Grid size (default: 32)")
    parser.add_argument(
        "--optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flag to perform optimization to maximize kinetic energy boost (default: True)",
    )
    args = parser.parse_args()
    N = args.N
    t_end = 1.0
    dt = 0.001
    nu = 0.001
    plot_flag = False  # True

    # Domain [0,1]^3
    L = 1.0
    dx = L / N
    xlin = jnp.linspace(0, L, num=N + 1)
    xlin = xlin[0:N]
    xx, yy, zz = jnp.meshgrid(xlin, xlin, xlin, indexing="ij")

    # Fourier Space Variables
    klin = 2.0 * jnp.pi / L * jnp.arange(-N / 2, N / 2)
    kmax = jnp.max(klin)
    kx, ky, kz = jnp.meshgrid(klin, klin, klin, indexing="ij")
    kx = jnp.fft.ifftshift(kx)
    ky = jnp.fft.ifftshift(ky)
    kz = jnp.fft.ifftshift(kz)
    kSq = kx**2 + ky**2 + kz**2
    kSq_inv = 1.0 / kSq
    kSq_inv = kSq_inv.at[kSq == 0].set(1.0)

    # dealias with the 2/3 rule
    dealias = (
        (jnp.abs(kx) < (2.0 / 3.0) * kmax)
        & (jnp.abs(ky) < (2.0 / 3.0) * kmax)
        & (jnp.abs(kz) < (2.0 / 3.0) * kmax)
    )

    Nt = int(jnp.ceil(t_end / dt))

    # Initial Condition (simple vortex, divergence free)
    # vx = -jnp.cos(2.0 * jnp.pi * yy) * jnp.cos(2.0 * jnp.pi * zz)
    # vy = jnp.cos(2.0 * jnp.pi * xx) * jnp.cos(2.0 * jnp.pi * zz)
    # vz = jnp.cos(2.0 * jnp.pi * xx) * jnp.cos(2.0 * jnp.pi * yy)
    Ax = jnp.cos(2.0 * jnp.pi * xx) * jnp.sin(2.0 * jnp.pi * yy) / (2.0 * jnp.pi)
    Ay = -jnp.cos(2.0 * jnp.pi * yy) * jnp.sin(2.0 * jnp.pi * zz) / (2.0 * jnp.pi)
    Az = jnp.cos(2.0 * jnp.pi * zz) * jnp.sin(2.0 * jnp.pi * xx) / (2.0 * jnp.pi)
    vx, vy, vz = curl(Ax, Ay, Az, kx, ky, kz)

    # check the divergence of the initial condition
    div_v = div(vx, vy, vz, kx, ky, kz)
    div_error = jnp.max(jnp.abs(div_v))
    assert div_error < 1e-8, f"Initial divergence is too large: {div_error:.6e}"

    ke_init = get_ke(vx, vy, vz, dx**3)

    # Run the simulation
    start_time = time.time()
    state = run_simulation_and_save_checkpoints(
        vx,
        vy,
        vz,
        dt,
        Nt,
        nu,
        kx,
        ky,
        kz,
        kSq,
        kSq_inv,
        dealias,
        f"checkpoints{N}_init",
    )
    jax.block_until_ready(state)
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.6f} seconds")

    vx, vy, vz = state
    ke_final = get_ke(vx, vy, vz, dx**3)
    ke_boost = ke_final / ke_init
    print(f"KE Boost: {ke_boost:.6f}")

    if plot_flag:
        make_plot(vx, vy, vz, kx, ky, kz, N, 1.0)

    # If optimization is enabled, perform the optimization to maximize kinetic energy boost
    if not args.optimize:
        print("Optimization is disabled. Exiting.")
        return

    # Now, carry out the optimization to maximize kinetic energy boost
    print("Starting optimization to maximize kinetic energy boost...")
    # to keep the velocity div-free, we will optimize
    # the potential field A, where v = curl(A)
    Ax = jnp.cos(2.0 * jnp.pi * xx) * jnp.sin(2.0 * jnp.pi * yy) / (2.0 * jnp.pi)
    Ay = -jnp.cos(2.0 * jnp.pi * yy) * jnp.sin(2.0 * jnp.pi * zz) / (2.0 * jnp.pi)
    Az = jnp.cos(2.0 * jnp.pi * zz) * jnp.sin(2.0 * jnp.pi * xx) / (2.0 * jnp.pi)

    start_time = time.time()
    theta = maximize_ke_boost(
        Ax, Ay, Az, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias, dx
    )
    jax.block_until_ready(theta)
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.6f} seconds")

    # Get the optimized velocity field, and re-run the simulation
    print("Running simulation with optimized initial condition...")
    Ax, Ay, Az = theta
    Ax -= jnp.mean(Ax)
    Ay -= jnp.mean(Ay)
    Az -= jnp.mean(Az)
    vx, vy, vz = curl(Ax, Ay, Az, kx, ky, kz)
    vx -= jnp.mean(vx)
    vy -= jnp.mean(vy)
    vz -= jnp.mean(vz)
    ke_init = get_ke(vx, vy, vz, dx**3)
    vx, vy, vz = run_simulation_and_save_checkpoints(
        vx, vy, vz, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias, f"checkpoints{N}"
    )
    ke_final = get_ke(vx, vy, vz, dx**3)
    ke_boost = ke_final / ke_init
    print(f"KE Boost: {ke_boost:.6f}")

    if plot_flag:
        make_plot(vx, vy, vz, kx, ky, kz, N, ke_boost)


if __name__ == "__main__":
    main()
