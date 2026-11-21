"""Steady states of the neoclassical economy when utility is additively separable but not homothetic.

    max  int e^{-rho t} [u(c) + theta v(s)] dt,   s = f(k) - c,   dk/dt = f(k) - delta k - c,   f(k) = A k^alpha,

with u'(c) = (c - cbar)^(-gc) (subsistence cbar >= 0, curvature gc) and v'(s) = s^(-gs) (curvature gs >= 0).
The paper's case (eq:iso) is cbar = 0, gc = gs = gamma: then utility is homothetic and the steady state is unique.

Steady state (eq:ss):  f'(k) = (rho + delta)(1 - tau),  tau = theta v'(s)/u'(c),  c = f(k) - delta k,  s = delta k.
Write x(k) = f'(k)/(rho + delta).  Solving for theta gives the one-to-one map from capital to the weight

    theta*(k) = (1 - x(k)) u'(c(k)) / v'(s(k)),        k_MGR < k < upper end of the feasible range,

and k is a steady state at theta iff theta*(k) = theta.  The number of steady states at theta is the number of
times the graph of theta* crosses the level theta, and the folds (saddle-node points) are its critical points.
With Cobb-Douglas technology, d ln x / d ln k = -(1 - alpha), so

    d ln theta* / d ln k = (1 - alpha) x/(1 - x) + sigma_s(s) - sigma_c(c) eps_c(k),          (*)

where sigma_c = -c u''/u' and sigma_s = -s v''/v' are the local curvatures (inverse elasticities of intertemporal
substitution) and eps_c = k (f' - delta)/(f - delta k) is the elasticity of steady-state consumption in capital,
positive below the golden rule and negative above it.  Since sigma_s >= 0, theta* can fall only where

    sigma_c(c(k)) > Gamma(k) := (1 - alpha) x / ((1 - x) eps_c),       k_MGR < k < k_GR,

a curvature bound that depends on rho, delta and alpha alone (not on A, theta or the utility function).

The stock model (Kurz 1968) with u(c) + theta_K w(k), w'(k) = k^(-gk): steady state f' - delta + theta_K w'(k)/u'(c)
= rho, so theta_K*(k) = (rho + delta)(1 - x) u'(c)/w'(k), and (*) holds with sigma_k = gk in place of sigma_s.
With isoelastic v and w of the same curvature, v'(delta k) = delta^(-gs) w'(k), so theta_K = (rho+delta) delta^(-gs)
theta maps the steady states of one model onto those of the other.
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar


class Technology:
    """f(k) = A k^alpha with depreciation delta and rate of impatience rho."""

    def __init__(self, rho=0.03, delta=0.05, alpha=1 / 3, A=1.0):
        self.rho, self.delta, self.alpha, self.A = rho, delta, alpha, A

    def f(self, k):
        return self.A * k ** self.alpha

    def fp(self, k):
        return self.A * self.alpha * k ** (self.alpha - 1)

    @property
    def k_mgr(self):
        """f'(k) = rho + delta."""
        return (self.A * self.alpha / (self.rho + self.delta)) ** (1 / (1 - self.alpha))

    @property
    def k_gr(self):
        """f'(k) = delta."""
        return (self.A * self.alpha / self.delta) ** (1 / (1 - self.alpha))

    @property
    def k_max(self):
        """f(k) = delta k."""
        return (self.A / self.delta) ** (1 / (1 - self.alpha))

    def c(self, k):
        """Steady-state consumption f(k) - delta k."""
        return self.f(k) - self.delta * k

    def x(self, k):
        """f'(k)/(rho + delta); x < 1 exactly beyond the modified golden rule."""
        return self.fp(k) / (self.rho + self.delta)

    def eps_c(self, k):
        """Elasticity of steady-state consumption with respect to capital, k (f' - delta)/(f - delta k)."""
        return k * (self.fp(k) - self.delta) / self.c(k)

    def Gamma(self, k):
        """Curvature bound (1 - alpha) x/((1 - x) eps_c) on (k_MGR, k_GR)."""
        x = self.x(k)
        return (1 - self.alpha) * x / ((1 - x) * self.eps_c(k))

    def Gamma_of_z(self, z):
        """Gamma as a function of the marginal product z = f'(k) in (delta, rho + delta):
        (1 - alpha)/alpha * z (z - alpha delta) / ((rho + delta - z)(z - delta)). Independent of A."""
        a, d, r = self.alpha, self.delta, self.rho
        return (1 - a) / a * z * (z - a * d) / ((r + d - z) * (z - d))

    def Gamma_min(self):
        """Minimum of Gamma over (k_MGR, k_GR), as (Gamma_min, k at the minimum, tau = 1 - x there).
        Found as the zero of d ln Gamma/dz = 1/z + 1/(z - alpha delta) + 1/(rho + delta - z) - 1/(z - delta),
        which runs from -inf at z = delta to +inf at z = rho + delta."""
        a, d, r = self.alpha, self.delta, self.rho
        g = lambda z: 1 / z + 1 / (z - a * d) + 1 / (r + d - z) - 1 / (z - d)
        eps = 1e-12 * (r + d)
        z = brentq(g, d + eps, r + d - eps, xtol=1e-15, rtol=1e-15)
        k = (self.A * a / z) ** (1 / (1 - a))
        return self.Gamma_of_z(z), k, 1 - z / (r + d)


class Separable:
    """u(c) + theta v(s) (flow, the paper's model) or u(c) + theta w(k) (stock, Kurz's), with
    u'(c) = (c - cbar)^(-gc) and v'(s) = s^(-gs), w'(k) = k^(-gs)."""

    def __init__(self, tech, gc=2.0, gs=2.0, cbar=0.0, model='flow'):
        assert model in ('flow', 'stock')
        self.t, self.gc, self.gs, self.cbar, self.model = tech, gc, gs, cbar, model

    # utility
    def up(self, c):
        return (c - self.cbar) ** (-self.gc)

    def sigma_c(self, c):
        return self.gc * c / (c - self.cbar)

    def arg_v(self, k):
        """Argument of the second term: saving delta k (flow) or capital k (stock)."""
        return self.t.delta * k if self.model == 'flow' else k

    def vp(self, k):
        return self.arg_v(k) ** (-self.gs)

    # range of steady states
    def k_range(self):
        """Open interval of capital on which theta* is defined: beyond k_MGR and with c(k) > cbar."""
        t = self.t
        lo, hi = t.k_mgr, t.k_max
        if self.cbar > 0:
            if t.c(t.k_gr) <= self.cbar:
                return None
            hi = brentq(lambda k: t.c(k) - self.cbar, t.k_gr, t.k_max, xtol=1e-14, rtol=1e-15)
            if t.c(lo) <= self.cbar:
                lo = brentq(lambda k: t.c(k) - self.cbar, lo, t.k_gr, xtol=1e-14, rtol=1e-15)
        return lo, hi

    def theta_star(self, k):
        """The weight theta at which k is a steady state."""
        t = self.t
        scale = 1.0 if self.model == 'flow' else (t.rho + t.delta)
        return scale * (1 - t.x(k)) * self.up(t.c(k)) / self.vp(k)

    def slope(self, k):
        """d ln theta*/d ln k, from (*)."""
        t = self.t
        x = t.x(k)
        return (1 - t.alpha) * x / (1 - x) + self.gs - self.sigma_c(t.c(k)) * t.eps_c(k)

    def grid(self, n=200_001):
        """Log-spaced grid strictly inside k_range()."""
        lo, hi = self.k_range()
        u = np.linspace(0.0, 1.0, n)[1:-1]
        return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))

    def folds(self, n=200_001):
        """Critical points of theta* (zeros of the slope), as a list of (k, theta*) pairs."""
        k = self.grid(n)
        d = self.slope(k)
        out = []
        for i in np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]:
            kk = brentq(self.slope, k[i], k[i + 1], xtol=1e-14, rtol=1e-15)
            out.append((kk, self.theta_star(kk)))
        return out

    def max_slope_deficit(self, n=200_001):
        """max over k of -(d ln theta*/d ln k): positive iff theta* falls somewhere, i.e. iff some theta
        has several steady states."""
        k = self.grid(n)
        return float(np.max(-self.slope(k)))

    def steady_states(self, theta, n=200_001):
        """All k with theta*(k) = theta, from the sign changes of ln theta* - ln theta on the grid."""
        k = self.grid(n)
        h = np.log(self.theta_star(k)) - np.log(theta)
        g = lambda kk: np.log(self.theta_star(kk)) - np.log(theta)
        return [brentq(g, k[i], k[i + 1], xtol=1e-14, rtol=1e-15)
                for i in np.where(np.sign(h[:-1]) * np.sign(h[1:]) < 0)[0]]


def min_gc_for_multiplicity(tech, gs, n=200_001):
    """Smallest constant curvature gc of u (no subsistence) for which theta* falls somewhere:
    min over k in (k_MGR, k_GR) of Gamma(k) + gs/eps_c(k)."""
    k = np.exp(np.linspace(np.log(tech.k_mgr), np.log(tech.k_gr), n)[1:-1])
    with np.errstate(divide='ignore', invalid='ignore'):
        req = tech.Gamma(k) + gs / tech.eps_c(k)
    i = int(np.nanargmin(req))
    kk = minimize_scalar(lambda q: tech.Gamma(q) + gs / tech.eps_c(q), method='bounded',
                         bounds=(k[max(i - 1, 0)], k[min(i + 1, len(k) - 1)]),
                         options=dict(xatol=1e-12)).x
    return tech.Gamma(kk) + gs / tech.eps_c(kk), kk


def min_cbar_for_multiplicity(tech, gc, gs, n=200_001):
    """Smallest subsistence level cbar for which theta* falls somewhere, with curvatures gc and gs.
    The slope (*) is negative at k iff c/(c - cbar) > R(k) := [(1-alpha) x/(1-x) + gs]/(gc eps_c), i.e.
    iff cbar > c(k) (1 - 1/R(k)); the smallest such cbar is the minimum of that product over (k_MGR, k_GR).
    Returns (cbar_min, k at the minimum, cbar_min/c(k) there)."""
    k = np.exp(np.linspace(np.log(tech.k_mgr), np.log(tech.k_gr), n)[1:-1])

    def need(q):
        x = tech.x(q)
        R = ((1 - tech.alpha) * x / (1 - x) + gs) / (gc * tech.eps_c(q))
        return tech.c(q) * (1 - 1 / R)

    with np.errstate(divide='ignore', invalid='ignore'):
        v = need(k)
    i = int(np.nanargmin(v))
    kk = minimize_scalar(need, method='bounded', bounds=(k[max(i - 1, 0)], k[min(i + 1, len(k) - 1)]),
                         options=dict(xatol=1e-12)).x
    return need(kk), kk, need(kk) / tech.c(kk)


def min_rho_for_multiplicity(delta, alpha, gc, gs, lo=1e-4, hi=10.0):
    """Smallest rate of impatience at which power utilities with curvatures gc and gs (no subsistence)
    give several steady states under Cobb-Douglas technology with depreciation delta and share alpha.
    The condition is scale free, so it does not depend on A or theta.  Returns nan if none in (lo, hi)."""
    gap = lambda r: min_gc_for_multiplicity(Technology(rho=r, delta=delta, alpha=alpha), gs, n=20_001)[0] - gc
    if gap(hi) > 0:
        return np.nan
    if gap(lo) < 0:
        return lo
    return brentq(gap, lo, hi, xtol=1e-10)


def flow_rhs(sep, theta, k, c):
    """(dk/dt, dc/dt) of the flow model u(c) + theta v(s), s = f(k) - c, from lambda = u'(c) - theta v'(s) and
    dlambda/dt = (rho + delta) lambda - f'(k) u'(c) (eq:foc).  lambda_c = u''(c) + theta v''(s) and
    lambda_k = -theta v''(s) f'(k), so dc/dt = [dlambda/dt - lambda_k dk/dt]/lambda_c."""
    t = sep.t
    assert sep.model == 'flow'
    s = t.f(k) - c
    kdot = s - t.delta * k
    up = sep.up(c)
    upp = -sep.gc * (c - sep.cbar) ** (-sep.gc - 1)
    vp = s ** (-sep.gs)
    vpp = -sep.gs * s ** (-sep.gs - 1)
    lam = up - theta * vp
    lamdot = (t.rho + t.delta) * lam - t.fp(k) * up
    lam_c = upp + theta * vpp
    lam_k = -theta * vpp * t.fp(k)
    return kdot, (lamdot - lam_k * kdot) / lam_c


def flow_jacobian(sep, theta, k, h=1e-6):
    """Central-difference Jacobian of (dk/dt, dc/dt) in (k, c) at the steady state with capital k."""
    c = sep.t.c(k)
    J = np.zeros((2, 2))
    for j, (dk, dc) in enumerate([(h * k, 0.0), (0.0, h * c)]):
        p = np.array(flow_rhs(sep, theta, k + dk, c + dc))
        m = np.array(flow_rhs(sep, theta, k - dk, c - dc))
        J[:, j] = (p - m) / (2 * (dk + dc))
    return J
