"""Linear technology (constant interest rate r) with homothetic isoelastic preferences

    U(c, s) = (c^(1-g) - 1)/(1-g) + theta (s^(1-g) - 1)/(1-g),   s = r a - c,   da/dt = s,

where a is total (financial plus human) wealth. On the optimal path c = z a with z constant, and the common
growth rate g = r - z of consumption, saving and wealth solves

    (gamma g + rho - r)/(gamma g + rho) = theta ((r - g)/g)^gamma  ( = U_s/U_c = tau ).

The left-hand side is increasing in g, the right-hand side decreasing, so the root is unique on
(max(0, g_R), r), g_R = (r - rho)/gamma the Ramsey growth rate.
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar


def ramsey_growth(r, rho, gamma):
    """Ramsey growth rate (r - rho)/gamma: the balanced-growth rate when there is no joy of saving."""
    return (r - rho) / gamma


def tau_lhs(g, r, rho, gamma):
    """Left-hand side of the growth equation, (gamma g + rho - r)/(gamma g + rho), increasing in g.
    At the root it equals tau = U_s/U_c, so the same expression serves as the formula for tau."""
    return (gamma * g + rho - r) / (gamma * g + rho)


def tau_rhs(g, r, theta, gamma):
    """Right-hand side of the growth equation, theta ((r - g)/g)^gamma, decreasing in g: the ratio
    U_s/U_c evaluated on the balanced growth path, where c = (r - g) a and s = g a."""
    return theta * ((r - g) / g) ** gamma


def growth(r, rho, gamma, theta):
    """Growth rate g of consumption and wealth; theta = 0 gives the Ramsey rate."""
    if theta == 0:
        return ramsey_growth(r, rho, gamma)
    lo = max(0.0, ramsey_growth(r, rho, gamma))
    F = lambda g: tau_lhs(g, r, rho, gamma) - tau_rhs(g, r, theta, gamma)
    eps = 1e-12 * r
    return brentq(F, lo + eps, r - eps, xtol=1e-14, rtol=1e-14, maxiter=500)


def solve(r, rho, gamma, theta):
    """All balanced-growth quantities as a dict."""
    g = growth(r, rho, gamma, theta)
    z = r - g                      # consumption-wealth ratio
    tau = tau_lhs(g, r, rho, gamma)  # U_s/U_c
    return dict(g=g, z=z, tau=tau, saving_rate=g / r, r_eff=r / (1 - tau),
                dg_dr=dg_dr(g, r, rho, gamma, theta), dg_dtheta=dg_dtheta(g, r, rho, gamma, theta),
                rho_equiv=r - gamma * g)  # discount rate of the Ramsey consumer with the same growth rate


def _F_partials(g, r, rho, gamma, theta):
    """Partial derivatives of F(g; r, theta) = lhs - rhs."""
    q = (r - g) / g
    F_g = gamma * r / (gamma * g + rho) ** 2 + theta * gamma * q ** (gamma - 1) * r / g ** 2
    F_r = -1.0 / (gamma * g + rho) - theta * gamma * q ** (gamma - 1) / g
    F_theta = -q ** gamma
    F_rho = r / (gamma * g + rho) ** 2
    return F_g, F_r, F_theta, F_rho


def dg_dr(g, r, rho, gamma, theta):
    """Response of the growth rate to the interest rate, by the implicit function theorem on F."""
    F_g, F_r, _, _ = _F_partials(g, r, rho, gamma, theta)
    return -F_r / F_g


def dg_dtheta(g, r, rho, gamma, theta):
    """Response of the growth rate to the strength of the joy of saving; positive."""
    F_g, _, F_theta, _ = _F_partials(g, r, rho, gamma, theta)
    return -F_theta / F_g


def dg_drho(g, r, rho, gamma, theta):
    """Response of the growth rate to the rate of impatience; negative, and -1/gamma when theta = 0."""
    F_g, _, _, F_rho = _F_partials(g, r, rho, gamma, theta)
    return -F_rho / F_g


def u_of_z(z, r, gamma, theta):
    """Felicity per unit a^(1-gamma), without the constants: U(z, r - z)."""
    s = r - z
    if gamma == 1:
        return np.log(z) + theta * np.log(s)
    return (z ** (1 - gamma) + theta * s ** (1 - gamma)) / (1 - gamma)


def hjb_direct(r, rho, gamma, theta):
    """Direct check of the closed form: maximize the Bellman right-hand side over z with the value
    V = A a^(1-gamma) (gamma != 1) and A fixed by the envelope condition. Returns the z that solves the
    fixed point rho A = max_z {u(z) + (1-gamma) A (r - z)} by iterating on A, independently of the formula."""
    assert gamma != 1

    def best(A):
        obj = lambda z: -(u_of_z(z, r, gamma, theta) + (1 - gamma) * A * (r - z))
        res = minimize_scalar(obj, bounds=(1e-9 * r, r * (1 - 1e-9)), method='bounded',
                              options={'xatol': 1e-13})
        return res.x, -res.fun

    G = lambda A: best(A)[1] - rho * A      # zero of G: the Bellman fixed point
    A_R = u_of_z(r - ramsey_growth(r, rho, gamma), r, gamma, 0) / (rho - (1 - gamma) * ramsey_growth(r, rho, gamma))
    lo, hi = (50 * A_R, A_R / 50) if A_R < 0 else (A_R / 50, 50 * A_R)
    A = brentq(G, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
    return best(A)[0], A
