import jax
import jax.numpy as jnp
import jax.numpy.fft as jfft
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

"""
Philip Mocz (2025)

Simulate the Navier-Stokes equations (incompressible viscous fluid) 
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


def grad(v, kx, ky):
    """return gradient of v"""
    v_hat = jfft.fftn(v)
    dvx = jnp.real(jfft.ifftn(1j * kx * v_hat))
    dvy = jnp.real(jfft.ifftn(1j * ky * v_hat))
    return dvx, dvy


def div(vx, vy, kx, ky):
    """return divergence of (vx,vy)"""
    dvx_x = jnp.real(jfft.ifftn(1j * kx * jfft.fftn(vx)))
    dvy_y = jnp.real(jfft.ifftn(1j * ky * jfft.fftn(vy)))
    return dvx_x + dvy_y


def curl(vx, vy, kx, ky):
    """return curl of (vx,vy)"""
    dvx_y = jnp.real(jfft.ifftn(1j * ky * jfft.fftn(vx)))
    dvy_x = jnp.real(jfft.ifftn(1j * kx * jfft.fftn(vy)))
    return dvy_x - dvx_y


def apply_dealias(f, dealias):
    """apply 2/3 rule dealias to field f"""
    f_hat = dealias * jfft.fftn(f)
    return jnp.real(jfft.ifftn(f_hat))


def main():
    """Navier-Stokes Simulation"""

    # Simulation parameters
    N = 400  # Spatial resolution
    t = 0  # current time of the simulation
    t_end = 1  # time at which simulation ends
    dt = 0.001  # timestep
    t_out = 0.01  # draw frequency
    nu = 0.001  # viscosity
    plot_realtime = True  # switch on for plotting as the simulation goes along

    # Domain [0,1] x [0,1]
    L = 1.0
    xlin = jnp.linspace(0, L, num=N + 1)  # Note: x=0 & x=1 are the same point!
    xlin = xlin[0:N]  # chop off periodic point
    xx, yy = jnp.meshgrid(xlin, xlin, indexing="ij")

    # Initial Condition (vortex)
    vx = -jnp.sin(2.0 * jnp.pi * yy)
    vy = jnp.sin(2.0 * jnp.pi * xx * 2)

    # Fourier Space Variables
    klin = 2.0 * jnp.pi / L * jnp.arange(-N / 2, N / 2)
    kmax = jnp.max(klin)
    kx, ky = jnp.meshgrid(klin, klin, indexing="ij")
    kx = jnp.fft.ifftshift(kx)
    ky = jnp.fft.ifftshift(ky)
    kSq = kx**2 + ky**2
    kSq_inv = 1.0 / kSq
    kSq_inv = kSq_inv.at[kSq == 0].set(1.0)

    # dealias with the 2/3 rule
    dealias = (jnp.abs(kx) < (2.0 / 3.0) * kmax) & (jnp.abs(ky) < (2.0 / 3.0) * kmax)

    # number of timesteps
    Nt = int(jnp.ceil(t_end / dt))

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=80)
    output_count = 1

    # Main Loop
    for i in range(Nt):
        # Advection: rhs = -(v.grad)v
        dvx_x, dvx_y = grad(vx, kx, ky)
        dvy_x, dvy_y = grad(vy, kx, ky)

        rhs_x = -(vx * dvx_x + vy * dvx_y)
        rhs_y = -(vx * dvy_x + vy * dvy_y)

        rhs_x = apply_dealias(rhs_x, dealias)
        rhs_y = apply_dealias(rhs_y, dealias)

        vx = vx + dt * rhs_x
        vy = vy + dt * rhs_y

        # Poisson solve for pressure
        div_rhs = div(rhs_x, rhs_y, kx, ky)
        P = poisson_solve(div_rhs, kSq_inv)
        dPx, dPy = grad(P, kx, ky)

        # Correction (to eliminate divergence component of velocity)
        vx = vx - dt * dPx
        vy = vy - dt * dPy

        # Diffusion solve (implicit)
        vx = diffusion_solve(vx, dt, nu, kSq)
        vy = diffusion_solve(vy, dt, nu, kSq)

        # vorticity (for plotting)
        wz = curl(vx, vy, kx, ky)

        # update time
        t += dt
        print(float(t))

        # plot in real time
        plot_this_turn = False
        if t + dt > output_count * t_out:
            plot_this_turn = True
        if (plot_realtime and plot_this_turn) or (i == Nt - 1):
            plt.cla()
            plt.imshow(jax.device_get(wz), cmap="RdBu")
            plt.clim(-20, 20)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")
            plt.pause(0.001)
            output_count += 1

    # Save figure
    # plt.savefig("navier-stokes-spectral.png", dpi=240)
    plt.show()

    return


if __name__ == "__main__":
    main()
