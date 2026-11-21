"""Common functions for the mental-accounting computations of The Joy of Saving (12 November 2026).

Model of Proposition prop:gains: wealth a at market value yields income at rate y and appreciates
at rate x, r = y + x; active saving s = y a - c; da/dt = s + x a.

Joy of saving (JoS): U(c,s) = u(c) + theta w(s), with u, w isoelastic of common curvature gamma,
u(c) = (c^(1-g) - 1)/(1-g) (log if g = 1).  By the proof of prop:gains(ii) the plan is
c = (y - h) a, s = h a, with h the root in (max(0,h_R), y) of
   G(h) = 1 - y/(g h + rho') - theta ((y-h)/h)^g = 0,  rho' = rho + (g-1) x,
   h_R = (y - rho')/g = (r - rho)/g - x.

Constrained Ramsey (the "consume only out of income" rule, c <= y a): c = min(y, z_R) a with
z_R = (rho - (1-g) r)/g (Ramsey consumption-wealth ratio).
"""
import numpy as np
from scipy.optimize import brentq


def u_iso(c, g):
    c = np.asarray(c, dtype=float)
    if g == 1:
        return np.log(c)
    return (c ** (1 - g) - 1) / (1 - g)


def up_iso(c, g):
    return np.asarray(c, dtype=float) ** (-g)


def rho_prime(rho, g, x):
    return rho + (g - 1) * x


def h_ramsey(r, rho, g, x):
    """Active saving per unit of wealth of the unconstrained Ramsey consumer."""
    return (r - rho) / g - x


def z_ramsey(r, rho, g):
    return (rho - (1 - g) * r) / g


def jos_G(h, y, rhop, g, theta):
    return 1 - y / (g * h + rhop) - theta * ((y - h) / h) ** g


def jos_h(theta, y, x, rho, g):
    """Root h* of prop:gains(ii). Solved in log h for accuracy when h is tiny."""
    rhop = rho_prime(rho, g, x)
    hR = (y - rhop) / g
    lo = max(0.0, hR)
    # work with h = lo + y*exp(t) style bracketing on the distance from lo
    span = y - lo

    def f(lt):
        d = span * np.exp(lt)          # h - lo
        h = lo + d
        # LHS 1 - y/(g h + rho') written as g (h - h_R)/(g h + rho') to avoid cancellation
        lhs = g * ((lo - hR) + d) / (g * h + rhop)
        return lhs - theta * ((y - h) / h) ** g

    a, b = -700.0, np.log1p(-1e-15)
    with np.errstate(over='ignore'):
        fa, fb = f(a), f(b)
    if not (fa < 0 < fb):
        raise ValueError(f"no bracket theta={theta} y={y} x={x}: fa={fa}, fb={fb}")
    with np.errstate(over='ignore'):
        lt = brentq(f, a, b, xtol=1e-14, rtol=1e-15, maxiter=500)
    return lo + span * np.exp(lt)


def jos_dh_dx(theta, y, x, rho, g):
    """Analytic dh*/dx at a given return r (y = r - x), from the proof of prop:gains (valid for any g>0
    as an implicit-function formula; the sign result in the paper needs g >= 1)."""
    rhop = rho_prime(rho, g, x)
    h = jos_h(theta, y, x, rho, g)
    D = g * h + rhop
    q = (y - h) / h
    Gh = g * y / D ** 2 + theta * g * q ** (g - 1) * y / h ** 2
    Gy = -1 / D - theta * g * q ** (g - 1) / h
    Grho = y / D ** 2
    return (Gy - (g - 1) * Grho) / Gh


def jos_tau(theta, y, x, rho, g):
    h = jos_h(theta, y, x, rho, g)
    return theta * ((y - h) / h) ** g


def pv_iso(level0, growth, rho, g):
    """int_0^inf e^{-rho t} u(level0 e^{growth t}) dt for isoelastic u (finite when rho > (1-g) growth)."""
    if g == 1:
        return np.log(level0) / rho + growth / rho ** 2
    den = rho - (1 - g) * growth
    assert den > 0
    return level0 ** (1 - g) / ((1 - g) * den) - 1 / ((1 - g) * rho)


def jos_value_parts(theta, a, y, x, rho, g):
    """Return (consumption part, joy part (not multiplied by theta), h) of the JoS value at wealth a."""
    h = jos_h(theta, y, x, rho, g)
    growth = h + x
    vc = pv_iso((y - h) * a, growth, rho, g)
    vs = pv_iso(h * a, growth, rho, g)
    return vc, vs, h


def cramsey_policy(y, x, rho, g):
    """Consumption-wealth ratio and active saving ratio of the constrained Ramsey consumer."""
    r = y + x
    zR = z_ramsey(r, rho, g)
    z = min(y, zR)
    return z, y - z


def cramsey_value(a, y, x, rho, g):
    z, h = cramsey_policy(y, x, rho, g)
    return pv_iso(z * a, h + x, rho, g)


def cramsey_multiplier(y, x, rho, g):
    """Normalized multiplier m = nu/u'(c) on c <= y a along the constrained Ramsey path.
    Binding: m = 1 - y/rho'; slack: 0."""
    rhop = rho_prime(rho, g, x)
    if (y - rhop) / g < 0:
        return 1 - y / rhop
    return 0.0
