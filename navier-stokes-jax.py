import jax
import time
import jax.numpy as jnp
import jax.numpy.fft as jfft
from functools import partial
import chex
from typing import NamedTuple
import optax
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

"""
Philip Mocz (2025)

Simulate the 3D Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

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
def run_simulation(vx, vy, vz, t, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias):
    """Run the full Navier-Stokes simulation"""

    def update(_, state):
        (vx, vy, vz, t) = state

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

        t += dt
        return (vx, vy, vz, t)

    (vx, vy, vz, t) = jax.lax.fori_loop(0, Nt, update, (vx, vy, vz, t))

    return vx, vy, vz, t


class InfoState(NamedTuple):
    iter_num: chex.Numeric


def print_info():
    def init_fn(params):
        del params
        return InfoState(iter_num=0)

    def update_fn(updates, state, params, *, value, grad, **extra_args):
        del params, extra_args

        jax.debug.print(
            "Iteration: {i}, Value: {v}, Gradient norm: {e}",
            i=state.iter_num,
            v=value,
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


def maximize_ke_boost(vx, vy, vz, t, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias, dx):
    """Optimize the initial velocity field maximize the kinetic energy boost"""

    # @jax.jit
    def loss_function(theta):
        vx, vy, vz = theta

        ke_init = get_ke(vx, vy, vz, dx**3)

        vx, vy, vz, _ = run_simulation(
            vx, vy, vz, t, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias
        )

        ke_final = get_ke(vx, vy, vz, dx**3)
        ke_boost = ke_final / ke_init

        return -ke_boost

    # opt = optax.lbfgs()
    opt = optax.chain(print_info(), optax.lbfgs())
    init_params = (vx, vy, vz)
    print(
        f"Initial value: {loss_function(init_params):.2e} "
        f"Initial gradient norm: {optax.tree_utils.tree_l2_norm(jax.grad(loss_function)(init_params)):.2e}"
    )
    max_iter = 6  # XXX 100
    final_params, _ = run_opt(
        init_params, loss_function, opt, max_iter=max_iter, tol=1e-3
    )
    print(
        f"Final value: {loss_function(final_params):.2e}, "
        f"Final gradient norm: {optax.tree_utils.tree_l2_norm(jax.grad(loss_function)(final_params)):.2e}"
    )

    return final_params


def plot_vorticity(vx, vy, vz, kx, ky, kz, N):
    """Plot the z-component of vorticity in the mid-plane"""
    _, _, wz = curl(vx, vy, vz, kx, ky, kz)
    plt.cla()
    plt.imshow(jax.device_get(wz[:, :, N // 2]), cmap="RdBu")
    plt.clim(-20, 20)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect("equal")
    plt.show()


def main():
    """3D Navier-Stokes Simulation"""

    # Simulation parameters
    N = 16  #  64  # 32 # 64
    t_end = 1.0
    dt = 0.001
    nu = 0.001

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

    # Initial Condition (simple vortex)
    t = 0.0
    vx = -jnp.cos(2.0 * jnp.pi * yy) * jnp.cos(2.0 * jnp.pi * zz)
    vy = jnp.cos(2.0 * jnp.pi * xx) * jnp.cos(2.0 * jnp.pi * zz)
    vz = jnp.cos(2.0 * jnp.pi * xx) * jnp.cos(2.0 * jnp.pi * yy)

    ke_init = get_ke(vx, vy, vz, dx**3)

    # Run the simulation
    start_time = time.time()
    vx, vy, vz, t = run_simulation(
        vx, vy, vz, t, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias
    )
    jax.block_until_ready(vx)
    jax.block_until_ready(vy)
    jax.block_until_ready(vz)
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.6f} seconds")

    ke_final = get_ke(vx, vy, vz, dx**3)
    ke_boost = ke_final / ke_init
    print(f"KE Boost: {ke_boost:.6f}")

    plot_vorticity(vx, vy, vz, kx, ky, kz, N)

    # reset initial condition for optimization
    t = 0.0
    vx = -jnp.cos(2.0 * jnp.pi * yy) * jnp.cos(2.0 * jnp.pi * zz)
    vy = jnp.cos(2.0 * jnp.pi * xx) * jnp.cos(2.0 * jnp.pi * zz)
    vz = jnp.cos(2.0 * jnp.pi * xx) * jnp.cos(2.0 * jnp.pi * yy)

    start_time = time.time()
    theta = maximize_ke_boost(
        vx, vy, vz, t, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias, dx
    )
    jax.block_until_ready(theta)
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.6f} seconds")

    # Print optimized kenetic energy boost
    vx, vy, vz = theta
    ke_init = get_ke(vx, vy, vz, dx**3)
    vx, vy, vz, _ = run_simulation(
        vx, vy, vz, t, dt, Nt, nu, kx, ky, kz, kSq, kSq_inv, dealias
    )
    ke_final = get_ke(vx, vy, vz, dx**3)
    ke_boost = ke_final / ke_init
    print(f"KE Boost: {ke_boost:.6f}")

    plot_vorticity(vx, vy, vz, kx, ky, kz, N)


if __name__ == "__main__":
    main()
