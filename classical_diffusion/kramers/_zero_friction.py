# cspell: ignore leggauss legvander chebvander chebint chebval legder legval spherical_jn tanh sinh cosh arccosh arcsinh fftfreq cumprod Fejer Yoshida matmul vander
"""Deterministic ISF and mean square displacement of a 1D periodic system at zero friction.

Hamiltonian dynamics, Boltzmann-distributed initial conditions, natural units of
`classical_diffusion.kramers._model.build_model`: m = k_B T = 1, the period of the
potential is 2 pi (length unit a / 2 pi), the time unit is that length over the
thermal velocity.

Method (action-angle / energy-shell decomposition)
--------------------------------------------------
dx dp = dE dtheta, theta the time along the orbit of energy E.  Each orbit is
periodic with period tau(E) and frequency W = 2 pi / tau: trapped orbits close,
running orbits advance by s 2 pi (s = +-1) per period, at mean velocity s W.
With c_n(E) the Fourier coefficients of f(theta) = exp(i K (x(theta) - s W theta))
along the orbit (trapped: s = 0),

    I(K, t) = sum_families int dW  w(W) sum_n |c_n|^2 exp(i (n + s K) W t),
    w(W) = e^{-E} tau |dE/dW| / Z,  Z = sqrt(2 pi) int e^{-V} dx.

W is the integration variable because the phase is linear in it: on each panel
of W the weight w |c_n|^2 is expanded in Legendre polynomials and
int_{-1}^{1} P_k(u) e^{i nu u} du = 2 i^k j_k(nu) gives the integral exactly
for that polynomial at ANY time, so the cost does not grow with t.  The orbits
near the separatrix (W -> 0) carry the weight e^{-E_s} tau |dE/dW| ~ e^{-c / W},
which is flat: the W < W_cut part is dropped once its weight is below the
tolerance.  Each node needs tau(E) (tanh-sinh quadrature, which resolves the
inverse square root at the turning points and the logarithm near the
separatrix), E(W) by bisection, and the Fourier coefficients along the orbit
(`orbit_from_parameter`): the quarter/half orbit is parametrised by pi - x = u cosh y
(trapped) or r sinh y (running), theta(y) is a Chebyshev cumulative integral and the
c_n are Fejer quadratures in y.  No time stepping (an earlier symplectic sampler,
was unstable near the separatrix).

MSD(t) = <(x(t) - x(0))^2> = sum_families int dW w(W) [ s^2 W^2 t^2
          + 2 sum_{n != 0} |d_n|^2 (1 - cos(n W t)) ],
d_n the Fourier coefficients of x(theta) - s W theta.  D is infinite (running orbits).

Limitations: one minimum and one maximum of V per unit cell (all CosineSystem1D),
and a frequency W(E) monotonic in each family (true for the cosine).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy.special import spherical_jn

if TYPE_CHECKING:
    from classical_diffusion.kramers._model import Model

# ---------------------------------------------------------------- potential


@dataclass(frozen=True)
class CosSeries:
    """V(x) = sum_k a_k (1 - cos(k x)) ... stored as c0 + sum a_k cos(k x), min at x = 0, max at pi."""

    k: np.ndarray
    a: np.ndarray
    c0: float

    def __call__(self, x):
        x = np.asarray(x, float)
        return self.c0 + np.sum(self.a * np.cos(np.multiply.outer(x, self.k)), axis=-1)

    def force(self, x):
        return np.sum(self.a * self.k * np.sin(np.multiply.outer(x, self.k)), axis=-1)

    def drop(self, X, d):
        """V(X) - V(X - d), without cancellation for small d."""
        X = np.asarray(X, float)[..., None]
        d = np.asarray(d, float)[..., None]
        return np.sum(
            -2 * self.a * np.sin(self.k * (2 * X - d) / 2) * np.sin(self.k * d / 2),
            axis=-1,
        )

    @property
    def barrier(self) -> float:
        return float(self(np.pi))

    def log_z(self) -> float:
        """log of Z = sqrt(2 pi) int_0^{2 pi} e^{-V} dx (trapezoid: spectrally exact)."""
        x = np.arange(4096) * 2 * np.pi / 4096
        v = self(x)
        return 0.5 * np.log(2 * np.pi) + np.log(np.mean(np.exp(-v)) * 2 * np.pi)


def series_from_model(model: Model) -> CosSeries:
    """The potential of a 1D model in its natural units, shifted to its minimum at 0."""
    if model.n_dim != 1:
        raise ValueError("1D only")
    coeffs = {n[0]: complex(v) for n, v in model.potential}
    ks = sorted(k for k in coeffs if k > 0)
    a = np.array([2 * coeffs[k].real for k in ks])
    if any(abs(coeffs[k].imag) > 1e-14 for k in ks):
        raise NotImplementedError("sine terms")
    k = np.array(ks, float)
    s = CosSeries(k, a, 0.0)
    if s(np.pi) < s(0.0):  # put the minimum at 0
        s = CosSeries(k, a * (-1.0) ** k, 0.0)
    s = CosSeries(k, s.a, -float(np.sum(s.a)))  # V(0) = 0
    xs = np.linspace(0, np.pi, 4001)
    if np.any(np.diff(s(xs)) <= 0):
        raise NotImplementedError("only one minimum and one maximum per unit cell")
    return s


def omega0(V: CosSeries) -> float:
    return float(np.sqrt(-np.sum(V.a * V.k**2)))


def omega_b(V: CosSeries) -> float:
    return float(np.sqrt(np.sum(V.a * V.k**2 * np.cos(V.k * np.pi))))


# ----------------------------------------------------------- period tau(E)

_GY, _GW = leggauss(160)


def _period_nodes_trapped(V: CosSeries, u):
    """Nodes of the half-orbit integral in y, with pi - x = u cosh y, y in [0, acosh(pi / u)].

    Returns x, the drop V(X) - V(x) (X = pi - u the turning point) and dx/dy, each (..., n).
    The substitution resolves the inverse square root at the turning point and the
    logarithm of the separatrix (the hilltop at pi)."""
    u = np.asarray(u, float)[..., None]
    Y = np.arccosh(np.pi / u)
    y = 0.5 * Y * (1 + _GY)
    w = 0.5 * Y * _GW
    x = np.pi - u * np.cosh(y)
    d = 2 * u * np.sinh(y / 2) ** 2  # X - x
    X = (np.pi - u) * np.ones_like(d)
    drop = V.drop(X, d)
    return x, drop, u * np.sinh(y), w


def period_trapped(V: CosSeries, X):
    """tau of the trapped orbit with turning points +-X: 4 int_0^X dx / sqrt(2 (V(X) - V(x)))."""
    _, drop, dxdy, w = _period_nodes_trapped(V, np.pi - np.asarray(X, float))
    return 4 * np.sum(w * dxdy / np.sqrt(2 * np.maximum(drop, 1e-300)), axis=-1)


def period_trapped_u(V: CosSeries, u):
    _, drop, dxdy, w = _period_nodes_trapped(V, u)
    return 4 * np.sum(w * dxdy / np.sqrt(2 * np.maximum(drop, 1e-300)), axis=-1)


def period_running(V: CosSeries, eps):
    """tau of the running orbit at E = E_s + eps: 2 int_0^pi dx / sqrt(2 (eps + V(pi) - V(x))),
    with pi - x = r sinh y, r = sqrt(2 eps) / omega_b."""
    eps = np.asarray(eps, float)[..., None]
    r = np.sqrt(2 * eps) / omega_b(V)
    Y = np.arcsinh(np.pi / r)
    y = 0.5 * Y * (1 + _GY)
    w = 0.5 * Y * _GW
    d = r * np.sinh(y)
    drop = V.drop(np.pi * np.ones_like(d), d)
    return 2 * np.sum(
        w * r * np.cosh(y) / np.sqrt(2 * (eps + np.maximum(drop, 0.0))), axis=-1
    )


def _bisect(f, lo, hi, target, increasing, n=64):
    """Vectorised bisection of f(z) = target on [lo, hi] (f monotonic)."""
    lo = np.full_like(target, lo, dtype=float)
    hi = np.full_like(target, hi, dtype=float)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        above = f(mid) > target
        if not increasing:
            above = ~above
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return 0.5 * (lo + hi)


def trapped_from_frequency(V: CosSeries, W):
    """Turning point X (and log(pi - X)) of the trapped orbit with frequency W < omega0."""
    W = np.asarray(W, float)
    # parameter z = log(pi - X), decreasing z -> X -> pi -> W -> 0: W increases with z
    f = lambda z: 2 * np.pi / period_trapped_u(V, np.exp(z))
    z = _bisect(f, -40.0, np.log(np.pi), W, increasing=True)
    return np.pi - np.exp(z)


def running_from_frequency(V: CosSeries, W):
    """eps = E - E_s of the running orbit with frequency W."""
    W = np.asarray(W, float)
    f = lambda z: 2 * np.pi / period_running(V, np.exp(z))
    return np.exp(_bisect(f, -80.0, 6.0, W, increasing=True))


# ---------------------------------------------------- nodes in the frequency


@dataclass
class Family:
    """One family of orbits (s = 0 trapped, +1 running right, -1 running left) on panels of W."""

    s: int
    edges: np.ndarray  # panel edges in W
    W: np.ndarray  # (n_panel, M) Gauss nodes
    E: np.ndarray
    tau: np.ndarray
    weight: np.ndarray  # w(W) at the nodes, times the Gauss weight NOT included
    c2: np.ndarray  # (n_panel, M, n_harm) |c_n|^2, n = fftfreq order
    d2: np.ndarray  # |d_n|^2 for the MSD
    n: np.ndarray  # harmonic numbers
    closure: float  # max orbit closure error


@dataclass
class ZeroFrictionModel:
    V: CosSeries
    K: float
    families: list[Family] = field(default_factory=list)
    M: int = 24
    info: dict = field(default_factory=dict)

    # ---------------------------------------------------------- ISF
    def isf(self, times, tol=1e-6):
        times = np.asarray(times, float)
        out = np.zeros(times.shape, complex)
        gx, gw = leggauss(self.M)
        P = legvander(gx, self.M - 1)  # (M, M)
        proj = (P * gw[:, None]).T * ((2 * np.arange(self.M) + 1) / 2)[
            :, None
        ]  # a = proj @ g
        ik = (1j) ** np.arange(self.M)
        for fam in self.families:
            mult = (
                2.0 if fam.s != 0 else 1.0
            )  # left movers = complex conjugate of right movers
            for ip in range(len(fam.edges) - 1):
                lo, hi = fam.edges[ip], fam.edges[ip + 1]
                m, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
                g = fam.weight[ip][:, None] * fam.c2[ip]  # (M, n_harm)
                keep = np.abs(g).max(axis=0) * 2 * h > 1e-4 * tol
                if not np.any(keep):
                    continue
                a = proj @ g[:, keep]  # (M, n_keep)
                nu = np.multiply.outer(
                    times, fam.n[keep] + fam.s * self.K
                )  # (T, n_keep)
                jk = spherical_jn(
                    np.arange(self.M)[:, None, None], np.abs(nu * h)[None]
                )  # (M, T, n)
                sign = np.where(nu < 0, (-1.0) ** np.arange(self.M)[:, None, None], 1.0)
                val = np.einsum("k,ktn,kn->tn", 2 * ik, jk * sign, a)
                contrib = (h * np.exp(1j * nu * m) * val).sum(axis=1)
                out += contrib.real * mult if fam.s != 0 else contrib
        return out.real if np.all(np.abs(out.imag) < 1e-10) else out

    # ---------------------------------------------------------- MSD
    def msd(self, times, tol=1e-6):
        times = np.asarray(times, float)
        out = np.zeros(times.shape)
        gx, gw = leggauss(self.M)
        P = legvander(gx, self.M - 1)
        proj = (P * gw[:, None]).T * ((2 * np.arange(self.M) + 1) / 2)[:, None]
        ik = (1j) ** np.arange(self.M)
        for fam in self.families:
            mult = 2.0 if fam.s != 0 else 1.0
            for ip in range(len(fam.edges) - 1):
                lo, hi = fam.edges[ip], fam.edges[ip + 1]
                m, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
                wq = fam.weight[ip] * gw * h
                # ballistic part
                if fam.s != 0:
                    out += mult * np.sum(wq * fam.W[ip] ** 2) * times**2
                nz = fam.n != 0
                g = 2 * fam.weight[ip][:, None] * fam.d2[ip][:, nz]
                out += mult * np.sum(
                    wq[:, None] * 2 * fam.d2[ip][:, nz]
                )  # constant part
                keep = np.abs(g).max(axis=0) * 2 * h > 1e-4 * tol
                if not np.any(keep):
                    continue
                a = proj @ g[:, keep]
                nu = np.multiply.outer(times, fam.n[nz][keep])
                jk = spherical_jn(
                    np.arange(self.M)[:, None, None], np.abs(nu * h)[None]
                )
                sign = np.where(nu < 0, (-1.0) ** np.arange(self.M)[:, None, None], 1.0)
                val = np.einsum("k,ktn,kn->tn", 2 * ik, jk * sign, a)
                out -= mult * (h * np.exp(1j * nu * m) * val).sum(axis=1).real
        return out


@dataclass(frozen=True)
class _Cheb:
    """Chebyshev (first kind) nodes on [-1, 1], the cumulative integral matrix from -1 and the Fejer weights."""

    y: np.ndarray
    Q: np.ndarray
    w: np.ndarray


def _cheb(n):
    from numpy.polynomial import chebyshev as C

    y = np.cos(np.pi * (np.arange(n) + 0.5) / n)[::-1]
    T = C.chebvander(y, n - 1)
    Tinv = np.linalg.inv(T)
    # cumulative integral of each basis polynomial, lower limit -1, evaluated at the nodes and at +1
    Ic = np.zeros((n + 1, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = 1
        Ic[:, k] = C.chebval(np.append(y, 1.0), C.chebint(e, lbnd=-1))
    Qfull = Ic @ Tinv
    return _Cheb(y=y, Q=Qfull[:-1], w=Qfull[-1])


_CHEB_CACHE: dict[int, _Cheb] = {}


def _get_cheb(n):
    if n not in _CHEB_CACHE:
        _CHEB_CACHE[n] = _cheb(n)
    return _CHEB_CACHE[n]


def _orbit_data(V, s, W, K, n_cheb=512, n_harm=256):
    """E, tau, |c_n|^2, |d_n|^2 (n = -n_harm..n_harm) at the frequencies W (any shape).

    No time stepping: the quarter (trapped) or half (running) orbit is parametrised by y,
    pi - x = u cosh y (trapped, turning point pi - u) or r sinh y (running), which
    resolves the turning point and the hilltop. theta(y) is the Chebyshev cumulative
    integral of dtheta/dy = (dx/dy) / p, and the Fourier coefficients along the orbit are
    Fejer quadratures in y, using the symmetry V(-x) = V(x) for the rest of the orbit.
    """
    shape = W.shape
    W = W.ravel()
    par = (
        np.pi - trapped_from_frequency(V, W) if s == 0 else running_from_frequency(V, W)
    )
    out = orbit_from_parameter(V, s, par, K, n_cheb, n_harm)
    return (
        out["E"].reshape(shape),
        out["tau"].reshape(shape),
        (np.abs(out["c"]) ** 2).reshape(shape + (-1,)),
        (np.abs(out["d"]) ** 2).reshape(shape + (-1,)),
        out["n"],
        out["closure"],
    )


def orbit_from_parameter(V, s, par, K, n_cheb=512, n_harm=256):
    """Orbit quantities at u = pi - X (trapped, s = 0) or eps = E - E_s (running, s = 1), 1D arrays.

    Returns dict with E, tau, action (int p dx over one period), c (signed Fourier
    coefficients of exp(i K x~), n = -n_harm..n_harm), d (of x~), n, closure."""
    par = np.asarray(par, float)
    ch = _get_cheb(n_cheb)
    if s == 0:
        u = par
        Y = np.arccosh(np.pi / u)
        y = (
            0.5 * Y[:, None] * (1 + ch.y)
        )  # increasing y: x from the turning point down to 0
        x = np.pi - u[:, None] * np.cosh(y)
        d = 2 * u[:, None] * np.sinh(y / 2) ** 2
        drop = V.drop((np.pi - u)[:, None] * np.ones_like(d), d)
        dxdy = u[:, None] * np.sinh(y)
        E = V(np.pi - u)
        span = 4
    else:
        eps = par
        r = np.sqrt(2 * eps) / omega_b(V)
        Y = np.arcsinh(np.pi / r)
        y = (
            0.5 * Y[:, None] * (1 + ch.y)
        )  # increasing y: x from the hilltop pi down to 0
        x = np.pi - r[:, None] * np.sinh(y)
        drop = eps[:, None] + V.drop(np.pi * np.ones_like(y), r[:, None] * np.sinh(y))
        dxdy = r[:, None] * np.cosh(y)
        E = V.barrier + eps
        span = 2
    dtdy = (
        dxdy / np.sqrt(2 * np.maximum(drop, 1e-300)) * (0.5 * Y[:, None])
    )  # d theta / d(cheb var)
    # theta measured from x = 0 (y = Y): theta(y) = int_y^Y
    total = dtdy @ ch.w
    theta = total[:, None] - dtdy @ ch.Q.T
    tau = span * total
    Wt = 2 * np.pi / tau
    phi = Wt[:, None] * theta  # in [0, pi/2] (trapped) or [0, pi] (running)
    n = np.arange(-n_harm, n_harm + 1)
    wq = (ch.w[None, :] * dtdy) * Wt[:, None] / (2 * np.pi)  # d phi / (2 pi) per node
    e1 = np.exp(-1j * phi)
    pw = np.empty(phi.shape + (n_harm + 1,), complex)
    pw[..., 0] = 1.0
    pw[..., 1:] = e1[..., None]
    pw = np.cumprod(pw, axis=-1)

    def proj(a):
        """(a . exp(-i n phi), a . exp(+i n phi)) for n = -H..H, a of shape (N, q)."""
        S = np.matmul(a[:, None, :], pw)[:, 0, :]  # a . e^{-i k phi}, k >= 0
        Sb = np.conj(np.matmul(np.conj(a)[:, None, :], pw)[:, 0, :])  # a . e^{+i k phi}
        en = np.concatenate([Sb[:, :0:-1], S], axis=1)
        enc = np.concatenate([S[:, :0:-1], Sb], axis=1)
        return en, enc

    sgn = (-1.0) ** n
    if s == 0:
        # four quarters: (phi, x), (pi - phi, x), (pi + phi, -x), (2 pi - phi, -x)
        fp = np.exp(1j * K * x)
        p1, p1c = proj(wq * fp)
        p2, p2c = proj(wq * np.conj(fp))
        c = p1 + sgn * p1c + sgn * p2 + p2c
        q1, q1c = proj((wq * x).astype(complex))
        dd = q1 + sgn * q1c - sgn * q1 - q1c
    else:
        # two halves: (phi, x) and (-phi, -x); periodic part x - phi
        xt = x - phi
        fp = np.exp(1j * K * xt)
        p1, _ = proj(wq * fp)
        _, p2c = proj(wq * np.conj(fp))
        c = p1 + p2c
        q1, q1c = proj((wq * xt).astype(complex))
        dd = q1 - q1c
    # consistency of the two period quadratures
    tau_g = period_trapped_u(V, u) if s == 0 else period_running(V, eps)
    closure = float(np.max(np.abs(tau / tau_g - 1)))
    action = span * np.sum(ch.w[None, :] * dtdy * 2 * np.maximum(drop, 0.0), axis=1)
    return dict(
        E=E, tau=tau, action=action, c=c, d=dd, n=n, closure=closure, x=x, phi=phi
    )


def _dE_dW(W_nodes, E_nodes, gx, lo, hi):
    """dE/dW on one panel from the Legendre interpolant of E(W)."""
    M = len(gx)
    P = legvander(gx, M - 1)
    coef = np.linalg.solve(P, E_nodes)
    from numpy.polynomial import legendre as L

    dcoef = L.legder(coef)
    return L.legval(gx, dcoef) * 2 / (hi - lo)


def build(
    V: CosSeries,
    K: float,
    tol: float = 1e-6,
    M: int = 24,
    n_cheb: int = 512,
    n_harm: int = 256,
    max_panels: int = 200,
    verbose=False,
) -> ZeroFrictionModel:
    """Adaptive panels in W for the trapped and the right-running family."""
    logz = V.log_z()
    w0 = omega0(V)
    gx, gw = leggauss(M)
    model = ZeroFrictionModel(V=V, K=K, M=M)
    E_max = V.barrier + max(40.0, -np.log(tol) + 30)

    def panel(s, lo, hi):
        W = 0.5 * (lo + hi) + 0.5 * (hi - lo) * gx
        nc, nh = n_cheb // 4, n_harm // 4
        while True:
            E, tau, c2, d2, n, clo = _orbit_data(V, s, W, K, nc, nh)
            edge = max(
                c2[:, :3].max(),
                c2[:, -3:].max(),
                (d2[:, :3].max() + d2[:, -3:].max()) * 1e-2,
            )
            if (edge * (hi - lo) < 1e-6 * tol and clo < 1e-9) or nc >= n_cheb:
                break
            nc, nh = 2 * nc, 2 * nh
        # pad the harmonics to the common size
        pad = n_harm - nh
        c2 = np.pad(c2, ((0, 0), (pad, pad)))
        d2 = np.pad(d2, ((0, 0), (pad, pad)))
        n = np.arange(-n_harm, n_harm + 1)
        dEdW = _dE_dW(W, E, gx, lo, hi)
        weight = np.exp(-E - logz) * tau * np.abs(dEdW)
        return dict(
            W=W, E=E, tau=tau, c2=c2, d2=d2, n=n, clo=clo, weight=weight, edge=edge
        )

    def weight_only(s, W):
        """w(W) by a central difference of E(W), for the separatrix cut."""
        Ws = np.array([W * (1 - 1e-5), W * (1 + 1e-5)])
        if s == 0:
            E = V(trapped_from_frequency(V, Ws))
        else:
            E = V.barrier + running_from_frequency(V, Ws)
        dEdW = (E[1] - E[0]) / (Ws[1] - Ws[0])
        return np.exp(-E.mean() - logz) * 2 * np.pi / W * abs(dEdW)

    def tail(p, lo, hi):
        """Legendre tail of the weight times |c_n|^2 (summed over n) and of E(W), scaled to the integral."""
        P = legvander(gx, M - 1)
        g = p["weight"][:, None] * np.concatenate(
            [p["c2"], p["d2"] / (1 + p["d2"].max())], axis=1
        )
        a = np.linalg.solve(P, g)
        err_g = np.abs(a[-3:]).sum(axis=0).sum() * (hi - lo)
        aE = np.linalg.solve(P, p["E"])
        rel_E = np.abs(aE[-3:]).sum() / max(1e-300, np.abs(aE[1:]).sum())
        return err_g, rel_E

    for s in (0, 1):
        # separatrix cut: shrink W_cut until the weight below it is negligible
        if s == 0:
            top = w0
        else:
            top = np.sqrt(2 * (E_max - V.barrier))
        cut = 0.5 * top
        while True:
            # weight below cut <= w(cut) * cut (w increases with W near the separatrix)
            if weight_only(s, cut) * cut < 1e-3 * tol:
                break
            cut *= 0.8
        edges = list(np.linspace(cut, top, 9))
        done = []
        todo = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        closure = 0.0
        while todo:
            lo, hi = todo.pop()
            p = panel(s, lo, hi)
            err, relE = tail(p, lo, hi)
            if (
                (err > 1e-2 * tol or relE > 1e-10)
                and len(done) + len(todo) < max_panels
                and (hi - lo) > 1e-6
            ):
                mid = 0.5 * (lo + hi)
                todo += [(lo, mid), (mid, hi)]
            else:
                done.append((lo, hi, p))
                closure = max(closure, p["clo"])
                if verbose > 1:
                    print(
                        f"  panel [{lo:.4g},{hi:.4g}] err {err:.1e} relE {relE:.1e} edge {p['edge']:.1e} clo {p['clo']:.1e}",
                        flush=True,
                    )
        done.sort(key=lambda t: t[0])
        fam = Family(
            s=s,
            edges=np.array([d[0] for d in done] + [done[-1][1]]),
            W=np.array([d[2]["W"] for d in done]),
            E=np.array([d[2]["E"] for d in done]),
            tau=np.array([d[2]["tau"] for d in done]),
            weight=np.array([d[2]["weight"] for d in done]),
            c2=np.array([d[2]["c2"] for d in done]),
            d2=np.array([d[2]["d2"] for d in done]),
            n=done[0][2]["n"],
            closure=closure,
        )
        model.families.append(fam)
        if verbose:
            print(
                f"family s={s}: W in [{cut:.4g}, {top:.4g}], {len(done)} panels, closure {closure:.1e}",
                flush=True,
            )
    # normalisation check: I(0) = sum of weights
    total = 0.0
    for fam in model.families:
        h = 0.5 * np.diff(fam.edges)
        total += (2 if fam.s else 1) * np.sum(fam.weight * gw[None, :] * h[:, None])
    model.info["norm"] = total
    model.info["harmonic_tail"] = max(
        float(np.max(np.concatenate([f.c2[..., :4], f.c2[..., -4:]], axis=-1)))
        for f in model.families
    )
    return model
