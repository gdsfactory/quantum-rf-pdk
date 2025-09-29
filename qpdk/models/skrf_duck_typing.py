"""Duck-typing for skrf to make JAX run."""

from functools import partial
from math import pi

import jax
import jax.numpy as jnp
import skrf
from jaxellip import ellipk
from skrf.media import CPW, parse_z0
from skrf.media.cpw import epsilon_0
from skrf.network import renormalize_s

# Duck type scikit-rf functions such that `jax.jit` runs


@partial(jax.jit, inline=True)
def _skin_depth(f, rho: float, mu_r: float):
    """Modified version of :func:`skrf.tlineFunctions.skin_depth` such that jax.jit runs."""
    return jnp.sqrt(rho / (pi * f * mu_r * skrf.mu_0))


@partial(jax.jit, inline=True)
def _surface_resistivity(f, rho, mu_r):
    """Modified version of :func:`skrf.tlineFunctions.surface_resistivity` such that jax.jit runs."""
    return rho / _skin_depth(rho=rho, f=f, mu_r=mu_r)


skrf.tlineFunctions.skin_depth = _skin_depth
skrf.tlineFunctions.surface_resistivity = _surface_resistivity


@classmethod  # pyrefly: ignore
def _from_f(cls, f, unit=None) -> skrf.Frequency:
    """Modified version of :func:`skrf.frequency.Frequency.from_f` such that jax.jit runs."""
    if jnp.isscalar(f):
        f = [f]
    temp_freq = cls(0, 0, 0, unit=unit)
    temp_freq._f = jnp.asarray(f) * temp_freq.multiplier
    # temp_freq.check_monotonic_increasing()

    return temp_freq


def _check_monotonic_increasing(self):  # noqa: ARG001
    """Modified version of :func:`skrf.frequency.Frequency.check_monotonic_increasing` such that jax.jit runs."""
    return True


skrf.frequency.Frequency.from_f = _from_f
skrf.frequency.Frequency.check_monotonic_increasing = _check_monotonic_increasing


def _analyse_dielectric(
    self,  # noqa: ARG001
    ep_r,
    tand,
    f_low,
    f_high,
    f_epr_tand,
    f,
    diel: str,
):
    """Modified version of :func:`skrf.media.CPW.analyse_dielectric` such that jax.jit runs."""
    if diel == "djordjevicsvensson":
        # compute the slope for a log frequency scale, tanD dependent.
        k = jnp.log((f_high + 1j * f_epr_tand) / (f_low + 1j * f_epr_tand))
        fd = jnp.log((f_high + 1j * f) / (f_low + 1j * f))
        ep_d = -tand * ep_r / jnp.imag(k)
        # value for frequency above f_high
        ep_inf = ep_r * (1.0 + tand * jnp.real(k) / jnp.imag(k))
        # compute complex permitivity
        ep_r_f = ep_inf + ep_d * fd
        # get tand
        tand_f = -jnp.imag(ep_r_f) / jnp.real(ep_r_f)
    elif diel == "frequencyinvariant":
        ep_r_f = ep_r - 1j * ep_r * tand
        tand_f = tand
    else:
        raise ValueError("Unknown dielectric dispersion model")

    return ep_r_f, tand_f


@partial(jax.jit, inline=True)
def _ellipa(k):
    """Vectorized version of :func:`skrf.media.cpw.ellipa` compatible with jax.jit and array inputs."""
    k = jnp.asarray(k)
    sqrt_half = jnp.sqrt(0.5)
    kp = jnp.sqrt(1.0 - k * k)

    r_low = skrf.pi / jnp.log(2.0 * (1.0 + jnp.sqrt(kp)) / (1.0 - jnp.sqrt(kp)))
    r_high = jnp.log(2.0 * (1.0 + jnp.sqrt(k)) / (1.0 - jnp.sqrt(k))) / skrf.pi

    return jnp.where(k < sqrt_half, r_low, r_high)


def _analyse_quasi_static(self, ep_r, w, s, h, t, has_metal_backside: bool):  # noqa: ARG001
    """Modified version of :func:`skrf.media.CPW.analyse_quasi_static` such that jax.jit runs."""
    Z0 = jnp.sqrt(skrf.mu_0 / epsilon_0)
    a = w
    b = w + 2.0 * s

    # equation (3a) from [GhNa84] or (6) from [GhNa83]
    k1 = a / b
    kk1 = ellipk(k1)
    kpk1 = ellipk(jnp.sqrt(1.0 - k1 * k1))
    q1 = _ellipa(k1)

    # backside is metal
    if has_metal_backside:
        # equation (4) from [GhNa83]
        # in qucs the 2 coefficient turn to 4 and fit better with ads
        k3 = jnp.tanh(skrf.pi * a / 4.0 / h) / jnp.tanh(skrf.pi * b / 4.0 / h)
        q3 = _ellipa(k3)
        qz = 1.0 / (q1 + q3)
        # equation (7) from [GhNa83]
        # equivalent to e = (q1 + ep_r * q3) / (q1 + q3) and paper
        e = 1.0 + q3 * qz * (ep_r - 1.0)
        # equation (8) from [GhNa83] with later division by sqrt(e)
        zr = Z0 / 2.0 * qz

    # backside is air
    else:
        # equation (3b) from [GhNa84]
        k2 = jnp.sinh((skrf.pi / 4.0) * a / h) / jnp.sinh((skrf.pi / 4.0) * b / h)
        q2 = _ellipa(k2)
        # equation (2) from [GhNa84]
        e = 1.0 + (ep_r - 1.0) / 2.0 * q2 / q1
        # equation (1) from [GhNa84] with later division by sqrt(e)
        zr = Z0 / 4.0 / q1

    # a posteriori effect of strip thickness
    if t is not None and t > 0.0:
        # equation (7.98) from [GGBB96]
        d = 1.25 * t / skrf.pi * (1.0 + jnp.log(4.0 * skrf.pi * w / t))
        # equation between (7.99) and (7.100) from [GGBB96]
        # approx. equal to ke = (w + d) / (w + d + 2 * (s - d))
        ke = k1 + (1.0 - k1 * k1) * d / 2.0 / s
        qe = _ellipa(ke)

        # backside is metal
        if has_metal_backside:
            # equation (8) from [GhNa83] with k1 -> ke
            # but keep q3 unchanged ? (not in papers)
            qz = 1.0 / (qe + q3)
            zr = Z0 / 2.0 * qz
        # backside is air
        else:
            # equation (7.99) from [GGBB96] with later division by sqrt(e)
            zr = Z0 / 4.0 / qe

        # modifies ep_re
        # equation (7.100) of [GGBB96]
        e = e - (0.7 * (e - 1.0) * t / s) / (q1 + (0.7 * t / s))

    ep_reff = e
    # final division of (1) from [GhNa84] and (8) from [GhNa83]
    zl_eff = zr / jnp.sqrt(ep_reff)

    return zl_eff, ep_reff, k1, kk1, kpk1


def _analyse_dispersion(self, zl_eff, ep_reff, ep_r, w, s, h, f):  # noqa: ARG001
    """Modified version of :func:`skrf.media.CPW.analyse_dispersion` such that jax.jit runs."""
    # cut-off frequency of the TE0 mode
    fte = (skrf.c / 4.0) / (h * jnp.sqrt(ep_r - 1.0))

    # dispersion factor G
    p = jnp.log(w / h)
    u = 0.54 - (0.64 - 0.015 * p) * p
    v = 0.43 - (0.86 - 0.54 * p) * p
    G = jnp.exp(u * jnp.log(w / s) + v)

    # add the dispersive effects to ep_reff
    sqrt_ep_reff = jnp.sqrt(ep_reff)
    sqrt_e = sqrt_ep_reff + (jnp.sqrt(ep_r) - sqrt_ep_reff) / (
        1.0 + G * (f / fte) ** (-1.8)
    )

    e = sqrt_e**2

    z = zl_eff * sqrt_ep_reff / sqrt_e

    return z, e


def _analyse_loss(self, ep_r, ep_reff, tand, rho, mu_r, f, w, t, s, k1, kk1, kpk1):  # noqa: ARG001
    """Modified version of :func:`skrf.media.CPW.analyse_loss` such that jax.jit runs."""
    Z0 = jnp.sqrt(skrf.mu_0 / epsilon_0)
    if t is not None and t > 0.0:
        if rho is None:
            raise (
                AttributeError(
                    "must provide values conductivity and conductor thickness to calculate this. "
                    "see initializer help"
                )
            )
        r_s = _surface_resistivity(f=f, rho=rho, mu_r=1)
        # ds = _skin_depth(f=f, rho=rho, mu_r=1.0)
        # if any(t < 3 * ds):
        #     warnings.warn(
        #         "Conductor loss calculation invalid for line"
        #         f"height t ({t})  < 3 * skin depth ({ds[0]})",
        #         RuntimeWarning,
        #         stacklevel=2,
        #     )
        n = (1.0 - k1) * 8.0 * pi / (t * (1.0 + k1))
        a = w / 2.0
        b = a + s
        ac = (pi + jnp.log(n * a)) / a + (pi + jnp.log(n * b)) / b
        a_conductor = (
            r_s * jnp.sqrt(ep_reff) * ac / (4.0 * Z0 * kk1 * kpk1 * (1.0 - k1 * k1))
        )
    else:
        a_conductor = jnp.zeros(f.shape)

    l0 = skrf.c / f
    a_dielectric = (
        pi * ep_r / (ep_r - 1) * (ep_reff - 1) / jnp.sqrt(ep_reff) * tand / l0
    )

    return a_conductor, a_dielectric


@property  # pyrefly: ignore
def _gamma(self):
    """Modified version of :func:`skrf.media.CPW.analyse_loss` such that jax.jit runs."""
    ep_reff, f = jnp.real(self.ep_reff_f), self.frequency.f

    alpha = self.alpha_dielectric.copy()
    if self.rho is not None:
        alpha += self.alpha_conductor

    beta = 2.0 * pi * f * jnp.sqrt(ep_reff) / skrf.c

    return alpha + 1j * beta


CPW.analyse_dielectric = _analyse_dielectric
CPW.analyse_quasi_static = _analyse_quasi_static
CPW.analyse_dispersion = _analyse_dispersion
CPW.analyse_loss = _analyse_loss
CPW.gamma = _gamma


# @z0.setter
def z0(self, z0) -> None:
    """Modified version of :func:`skrf.network.Network.z0` such that jax.jit runs."""
    # cast any array like type (tuple, list) to a np.array
    z0 = jnp.array(z0, dtype=complex)

    # if _z0 is a vector or matrix, we check if _s is already assigned.
    # If not, we cannot proof the correct dimensions and silently accept
    # any vector or fxn array
    if not hasattr(self, "_s"):
        self._z0 = z0
        return

    # if _z0 is a scalar, we broadcast to the correct shape.
    #
    # if _z0 is a vector, we check if the dimension matches with either
    # nports or frequency.npoints. If yes, we accept the value.
    # Note that there can be an ambiguity in theory, if nports == npoints
    #
    # if _z0 is a matrix, we check if the shape matches with _s
    # In any other case raise an Exception
    self._z0 = jnp.empty(self.s.shape[:2], dtype=complex)
    if z0.ndim == 0:
        self._z0[:] = z0
    elif z0.ndim == 1 and z0.shape[0] == self.s.shape[0]:
        self._z0[:] = z0[:, None]
    elif z0.ndim == 1 and z0.shape[0] == self.s.shape[1]:
        self._z0[:] = z0[None, :]
    elif z0.shape == self.s.shape[:2]:
        self._z0 = z0
    else:
        raise AttributeError(
            f"Unable to broadcast z0 shape {z0.shape} to s shape {self.s.shape}."
        )


def _renormalize(self, z_new, s_def=None) -> None:
    """Modified version of :func:`skrf.network.Network.renormalize` such that jax.jit runs."""
    # cast any array like type (tuple, list) to a np.array
    z_new = jnp.array(z_new, dtype=complex)
    # make sure the z_new shape can be compared with self.z0
    z_new = _fix_z0_shape(z_new, self.frequency.npoints, self.nports)
    if s_def is None:
        s_def = self.s_def
    # Try to avoid renormalization if possible since it goes through
    # Z-parameters which can cause numerical inaccuracies.
    # We need to renormalize if z_new is different from z0
    # or s_def is different and there is at least one complex port.
    need_to_renorm = True
    # if jnp.any(self.z0 != z_new):
    #     need_to_renorm = True
    # if s_def != self.s_def and (self.z0.imag != 0).any():
    #     need_to_renorm = True
    if need_to_renorm:
        # We can use s2s if z0 is the same. This is numerically much more
        # accurate.
        # if jnp.all(self.z0 == z_new).item():
        #     self.s = s2s(self.s, self.z0, s_def, self.s_def)
        # else:
        self.s = renormalize_s(self.s, self.z0, z_new, s_def, self.s_def)
    # Update s_def if it was changed
    self.s_def = s_def
    self.z0 = z_new


def _renormalize_s(s, z_old, z_new, s_def=skrf.S_DEF_DEFAULT, s_def_old=None):
    """Modified version of :func:`skrf.network.Network.renormalize_s` such that jax.jit runs."""
    if s_def_old not in skrf.S_DEFINITIONS and s_def_old is not None:
        raise ValueError("s_def_old parameter should be one of:", skrf.S_DEFINITIONS)
    if s_def_old is None:
        s_def_old = s_def
    if s_def not in skrf.S_DEFINITIONS:
        raise ValueError("s_def parameter should be one of:", skrf.S_DEFINITIONS)
    # that's a heck of a one-liner!
    return _z2s(_s2z(s, z0=z_old, s_def=s_def_old), z0=z_new, s_def=s_def)


def _fix_z0_shape(z0, nfreqs, nports):
    """Modified version of :func:`skrf.network.Network.fix_z0_shape` such that jax.jit runs."""
    if jnp.shape(z0) == (nfreqs, nports):
        # z0 is of correct shape. super duper.return it quick.
        return z0.copy()

    if jnp.ndim(z0) == 0:
        # z0 is a single number or np.array without dimensions.
        return jnp.array(nfreqs * [nports * [z0]])

    if len(z0) == nports:
        # assume z0 is a list of impedances for each port,
        # but constant with frequency
        return jnp.array(nfreqs * [z0])

    if len(z0) == nfreqs:
        # assume z0 is a list of impedances for each frequency,
        # but constant with respect to ports
        return jnp.array(nports * [z0]).T

    raise IndexError("z0 is not an acceptable shape")


def _fix_param_shape(p):
    """Modified version of :func:`skrf.network.fix_param_shape` such that jax.jit runs."""
    # Ensure input is numpy array
    p = jnp.array(p, dtype=complex)
    if len(p.shape) == 0:
        # Scalar
        return p.reshape(1, 1, 1)
    if len(p.shape) == 1:
        # One port with many frequencies
        return p.reshape(p.shape[0], 1, 1)
    if p.shape[-1] != p.shape[-2]:
        raise ValueError("Input matrix must be square")
    if len(p.shape) == 2:
        # Many port with one frequency
        return p.reshape(-1, p.shape[0], p.shape[0])
    if len(p.shape) != 3:
        raise ValueError(f"Input array has too many dimensions. Shape: {p.shape}")
    return p


def _z2s(z, z0=50, s_def=skrf.S_DEF_DEFAULT):
    """Modified version of :func:`skrf.network.z2s` such that jax.jit runs."""
    nfreqs, nports, nports = z.shape
    z0 = _fix_z0_shape(z0, nfreqs, nports)

    # Add a small real part in case of pure imaginary char impedance
    # to prevent numerical errors for both pseudo and power waves definitions
    z0 = z0.astype(dtype=complex)
    z0 = jnp.where(z0.real == 0, z0 + skrf.ZERO, z0)

    z = jnp.array(z, dtype=complex)

    if s_def == "power":
        # Power-waves. Eq.(18) from [Kurokawa et al.3]
        # Creating diagonal matrices of shape (nports,nports) for each nfreqs
        F_diag = 1.0 / (2 * jnp.sqrt(z0.real))
        G_diag = z0
        F = jnp.eye(z.shape[-1])[None, :, :] * F_diag[:, :, None]
        G = jnp.eye(z.shape[-1])[None, :, :] * G_diag[:, :, None]
        s = skrf.mf.rsolve(F @ (z + G), F @ (z - jnp.conjugate(G)))

    elif s_def == "pseudo":
        # Pseudo-waves. Eq.(73) from [Marks et al.]
        # Creating diagonal matrices of shape (nports,nports) for each nfreqs
        U_diag = jnp.sqrt(z0.real) / jnp.abs(z0)
        ZR_diag = z0
        U = jnp.eye(z.shape[-1])[None, :, :] * U_diag[:, :, None]
        ZR = jnp.eye(z.shape[-1])[None, :, :] * ZR_diag[:, :, None]
        s = skrf.mf.rsolve(U @ (z + ZR), U @ (z - ZR))

    elif s_def == "traveling":
        # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
        # Creating Identity matrices of shape (nports,nports) for each nfreqs
        Id = jnp.eye(z.shape[-1])[None, :, :] * jnp.ones(
            (z.shape[0], 1, 1)
        )  # (nfreqs, nports, nports)
        sqrty0_diag = jnp.sqrt(1.0 / z0)
        sqrty0 = jnp.eye(z.shape[-1])[None, :, :] * sqrty0_diag[:, :, None]
        s = skrf.mf.rsolve(sqrty0 @ z @ sqrty0 + Id, sqrty0 @ z @ sqrty0 - Id)
    else:
        raise ValueError(f"Unknown s_def: {s_def}")

    return s


def _s2z(s, z0=50, s_def=skrf.S_DEF_DEFAULT):
    """Modified version of :func:`skrf.network.s2z` such that jax.jit runs."""
    nfreqs, nports, nports = s.shape
    z0 = _fix_z0_shape(z0, nfreqs, nports)

    # Add a small real part in case of pure imaginary char impedance
    # to prevent numerical errors for both pseudo and power waves definitions
    z0 = z0.astype(dtype=complex)
    z0 = jnp.where(z0.real == 0, z0 + skrf.ZERO, z0)

    s = jnp.array(s, dtype=complex)

    # The following is a vectorized version of a for loop for all frequencies.
    # # Creating Identity matrices of shape (nports,nports) for each nfreqs
    Id = jnp.eye(s.shape[-1])[None, :, :] * jnp.ones(
        (s.shape[0], 1, 1)
    )  # (nfreqs, nports, nports)

    if s_def == "power":
        # Power-waves. Eq.(19) from [Kurokawa et al.]
        # Creating diagonal matrices of shape (nports,nports) for each nfreqs
        F_diag = 1.0 / (2 * jnp.sqrt(z0.real))
        G_diag = z0
        F = jnp.eye(s.shape[-1])[None, :, :] * F_diag[:, :, None]
        G = jnp.eye(s.shape[-1])[None, :, :] * G_diag[:, :, None]
        z = jnp.linalg.solve(
            skrf.mf.nudge_eig((Id - s) @ F), (s @ G + jnp.conjugate(G)) @ F
        )

    elif s_def == "pseudo":
        # Pseudo-waves. Eq.(74) from [Marks et al.]
        # Creating diagonal matrices of shape (nports,nports) for each nfreqs
        U_diag = jnp.sqrt(z0.real) / jnp.abs(z0)
        ZR_diag = z0
        U = jnp.eye(s.shape[-1])[None, :, :] * U_diag[:, :, None]
        ZR = jnp.eye(s.shape[-1])[None, :, :] * ZR_diag[:, :, None]
        USU = jnp.linalg.solve(U, s @ U)
        z = jnp.linalg.solve(skrf.mf.nudge_eig(Id - USU), (Id + USU) @ ZR)

    elif s_def == "traveling":
        # Traveling-waves definition. Cf.Wikipedia "Impedance parameters" page.
        # Creating diagonal matrices of shape (nports, nports) for each nfreqs
        sqrtz0_diag = jnp.sqrt(z0)
        sqrtz0 = jnp.eye(s.shape[-1])[None, :, :] * sqrtz0_diag[:, :, None]
        z = sqrtz0 @ jnp.linalg.solve(_nudge_eig(Id - s), (Id + s) @ sqrtz0)
    else:
        raise ValueError(f"Unknown s_def: {s_def}")

    return z


skrf.network.Network.z0 = z0
skrf.network.Network.renormalize = _renormalize
skrf.network.fix_z0_shape = _fix_z0_shape
skrf.network.fix_param_shape = _fix_param_shape
skrf.network.renormalize_s = _renormalize_s
skrf.network.z2s = _z2s
skrf.network.s2z = _s2z


@partial(jax.jit, inline=True)
def _nudge_eig(mat, cond=None, min_eig=None):
    """Modified version of :func:`skrf.mathFunctions.nudge_eig` such that jax.jit runs."""
    # use current constants
    if not cond:
        cond = skrf.EIG_COND
    if not min_eig:
        min_eig = skrf.EIG_MIN

    # Eigenvalues and vectors
    eigw, eigv = jnp.linalg.eig(mat)
    # Max eigenvalue for each frequency
    max_eig = jnp.amax(jnp.abs(eigw), axis=1)
    # Calculate mask for positions where problematic eigenvalues are
    mask = jnp.logical_or(
        jnp.abs(eigw) < cond * max_eig[:, None], jnp.abs(eigw) < min_eig
    )

    # Instead of boolean indexing, use where to conditionally update eigenvalues
    corrected_eig = jnp.maximum(cond * max_eig[:, None], min_eig * jnp.ones_like(eigw))
    eigw = jnp.where(mask, corrected_eig, eigw)

    # Now assemble the eigendecomposited matrices back
    e = jnp.eye(mat.shape[-1])[None, :, :] * eigw[:, :, None]
    return _rsolve(eigv, eigv @ e)


@partial(jax.jit, inline=True)
def _rsolve(A, B):
    """Modified version of :func:`skrf.mathFunctions.nudge_eig` such that jax.jit runs."""
    return jnp.transpose(
        jnp.linalg.solve(
            jnp.transpose(A, (0, 2, 1)).conj(), jnp.transpose(B, (0, 2, 1)).conj()
        ),
        (0, 2, 1),
    ).conj()


skrf.mathFunctions.nudge_eig = _nudge_eig
skrf.mathFunctions.rsolve = _rsolve


def _line(self, d, unit, z0=None, embed=False, **kwargs):  # noqa: ARG001
    """Modified version of :func:`skrf.media.Media.line` such that jax.jit runs."""
    if isinstance(z0, str):
        z0 = parse_z0(z0) * self.z0

    if z0 is None:
        z0 = self.z0

    s_def = kwargs.pop("s_def", skrf.S_DEF_DEFAULT)

    # The use of either traveling or pseudo waves s-parameters definition
    # is required here.
    # The definition of the reflection coefficient for power waves has
    # conjugation.
    result = self.match(nports=2, z0=z0, s_def="traveling", **kwargs)
    # breakpoint()

    theta = self.electrical_length(self.to_meters(d=d, unit=unit))
    # breakpoint()

    s11 = jnp.zeros(self.frequency.npoints, dtype=complex)
    s21 = jnp.exp(-1 * theta)
    result.s = jnp.array([[s11, s21], [s21, s11]]).transpose().reshape(-1, 2, 2)
    # breakpoint()

    # renormalize (or embed) into z0_port if required
    if self.z0_port is not None:
        result.renormalize(self.z0_port)
    result.renormalize(result.z0, s_def=s_def)

    return result


skrf.media.Media.line = _line
