"""The flow formulation against the stock formulation, with linear technology.

Capital gains. Wealth a, an asset valued at market prices, with no labor income, yields income at the
rate y (dividends, rents, interest) and appreciates at the rate x, so that its return is r = y + x. Active saving
is income less consumption, s = y a - c, and da/dt = s + x a. With the flow felicity
U(c, s) = c^(1-gamma)/(1-gamma) + theta s^(1-gamma)/(1-gamma), the substitution a = e^{xt} b turns the
problem into the paper's linear problem with interest rate y and rate of impatience rho + (gamma-1) x, so
active saving per unit of wealth, h = s/a, is the growth rate of that problem:

    (gamma h + rho' - y)/(gamma h + rho') = theta ((y - h)/h)^gamma,    rho' = rho + (gamma - 1) x.

Under any felicity over consumption and the market value of wealth, the Ramsey felicity included, the
problem depends on y and x only through r, so h = g(r) - x: each unit of gain displaces a unit of active
saving.

The additive stock analogue. U(c, a) = c^(1-gamma)/(1-gamma) + theta a^(1-gamma)/(1-gamma), da/dt = r a - c.
On its balanced path c = z a, g = r - z and the Euler equation gives gamma g = r - rho + theta z^gamma.
"""
from scipy.optimize import brentq

from joy import linear


# --- capital gains -------------------------------------------------------------------------------

def active_saving_flow(y, x, rho, gamma, theta):
    """Active saving per unit of wealth, s/a, under the flow felicity when wealth yields y and gains x."""
    return linear.growth(y, rho + (gamma - 1) * x, gamma, theta)


def active_saving_wealth_in_utility(g_of_r, y, x):
    """Active saving per unit of wealth under any felicity over consumption and market wealth:
    the growth rate depends on the return r = y + x only, and gains are subtracted one for one."""
    return g_of_r(y + x) - x


def dh_dx_flow(y, x, rho, gamma, theta):
    """Derivative of active saving per unit of wealth with respect to the gain rate x, holding the
    return r = y + x fixed, from the closed form in the appendix:
        dh/dx = -1 + tau (gamma D - h) / (h D G_h),   D = gamma h + rho',
    G_h the derivative of (lhs - rhs) in h, as in linear._F_partials with r replaced by y."""
    rp = rho + (gamma - 1) * x
    h = active_saving_flow(y, x, rho, gamma, theta)
    D = gamma * h + rp
    tau = 1 - y / D
    G_h, _, _, _ = linear._F_partials(h, y, rp, gamma, theta)
    return -1 + tau * (gamma * D - h) / (h * D * G_h)


# --- the additive stock analogue -----------------------------------------------------------------

def growth_stock(r, rho, gamma, theta):
    """Balanced growth rate under U(c, a) = c^(1-gamma)/(1-gamma) + theta a^(1-gamma)/(1-gamma):
    the root g < r of gamma g = r - rho + theta (r - g)^gamma, unique since the difference of the two
    sides is increasing in g, negative as g -> -inf and positive at g = r when rho > (1-gamma) r."""
    F = lambda g: gamma * g - (r - rho) - theta * (r - g) ** gamma
    lo = -1.0
    while F(lo) > 0:
        lo *= 2
    return brentq(F, lo, r - 1e-14, xtol=1e-15, rtol=1e-14, maxiter=500)


def theta_stock_for_growth(g, r, rho, gamma):
    """The weight on wealth at which the additive stock consumer grows at the rate g."""
    return (gamma * g - (r - rho)) / (r - g) ** gamma


def dg_dr_stock(r, rho, gamma, theta):
    """Response of growth to the interest rate under the additive stock felicity:
        dg/dr = (1 + X)/(gamma + X),   X = gamma theta z^(gamma-1),   z = r - g,
    which exceeds 1/gamma when gamma > 1 for every theta > 0."""
    g = growth_stock(r, rho, gamma, theta)
    X = gamma * theta * (r - g) ** (gamma - 1)
    return (1 + X) / (gamma + X)
