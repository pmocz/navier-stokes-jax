import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
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


def apply_dealias(f, dealias):
    """apply 2/3 rule dealias to field f"""
    f_hat = dealias * jfft.fftn(f)
    return jnp.real(jfft.ifftn(f_hat))


def main():
    """3D Navier-Stokes Simulation"""

    # Simulation parameters
    N = 64  # Spatial resolution
    t = 0.0
    t_end = 1.0
    dt = 0.001
    t_out = 0.01
    nu = 0.001
    plot_realtime = True

    # Domain [0,1]^3
    L = 1.0
    xlin = jnp.linspace(0, L, num=N + 1)
    xlin = xlin[0:N]
    xx, yy, zz = jnp.meshgrid(xlin, xlin, xlin, indexing="ij")

    # Initial Condition (simple vortex)
    # vx = -jnp.sin(2.0 * jnp.pi * yy) * jnp.sin(2.0 * jnp.pi * zz)
    # vy = jnp.sin(2.0 * jnp.pi * xx) * jnp.sin(2.0 * jnp.pi * zz)
    # vz = jnp.sin(2.0 * jnp.pi * xx) * jnp.sin(2.0 * jnp.pi * yy)

    # Initial Condition (vortex)
    vx = -jnp.sin(2.0 * jnp.pi * yy)
    vy = jnp.sin(2.0 * jnp.pi * xx * 2)
    vz = jnp.zeros_like(xx)

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

    fig = plt.figure(figsize=(5, 4), dpi=80)
    output_count = 1

    for i in range(Nt):
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

        # vorticity (for plotting)
        wx, wy, wz = curl(vx, vy, vz, kx, ky, kz)

        t += dt
        print(float(t))

        # plot in real time (show a 2D slice of wz at z=N//2)
        plot_this_turn = False
        if t + dt > output_count * t_out:
            plot_this_turn = True
        if (plot_realtime and plot_this_turn) or (i == Nt - 1):
            plt.cla()
            plt.imshow(jax.device_get(wz[:, :, N // 2]), cmap="RdBu")
            plt.clim(-20, 20)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")
            plt.pause(0.0001)
            output_count += 1

    plt.show()
    return


if __name__ == "__main__":
    main()
