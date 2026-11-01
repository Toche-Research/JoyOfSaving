"""Exploratory computation (not in the paper): a risky return on wealth with the homothetic joy of saving.

    da = (r a - c) dt + varsigma a dW,   s = r a - c  (saving planned out of expected income),
    U = (c^(1-g)-1)/(1-g) + theta (s^(1-g)-1)/(1-g).

Conjecture: V = A a^(1-gamma) + const, c = z a, and the growth rate g = r - z solves the deterministic equation
with rho replaced by rho_hat = rho - gamma (gamma - 1) varsigma^2 / 2.  The value is finite, and the candidate
verified, iff rho_hat > (1 - gamma) g: with ln a_t = ln a_0 + (g - varsigma^2/2) t + varsigma W_t,
E[a_t^(1-gamma)] grows at the rate (1-gamma) g - gamma (1-gamma) varsigma^2 / 2, so e^{-rho t} E[V(a_t)] -> 0
iff rho_hat > (1-gamma) g (the deterministic condition in rho_hat).  When rho_hat < 0 (large varsigma, gamma > 1)
the closed form in rho_hat has a pole at g = -rho_hat/gamma and a second root below it, which violates the
finiteness condition; the root is therefore bracketed in (max(0, g_R_hat), r), g_R_hat = (r - rho_hat)/gamma, which
lies above the pole, and every solution is checked against rho_hat > (1 - gamma) g.  Part 1 checks the conjecture
by evaluating the residual of the HJB equation at the conjectured value function, with the maximization over c
done numerically (a check of the algebra, not an independent solution: a grid solution was attempted and
abandoned because the top boundary, into which the positive drift carries the state, contaminates the whole
grid).  Part 2 tabulates the precautionary response of the growth rate and how the joy of saving damps it.
Nothing in the paper depends on this script: it feeds no figure, no table and no macro, and its output goes
only to out/risk.txt. It is kept because the referee report discusses the risky case.
Run: cd code && PYTHONPATH=. python3 scripts/risk.py   (seconds)."""
import os, time
import numpy as np
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from joy import linear

OUT = os.path.join(os.path.dirname(__file__), '..', 'out', 'risk.txt')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
R, RHO, GAMMA = 0.05, 0.03, 2.0
lines = []


def log(s):
    """Print a line and keep it for out/risk.txt."""
    print(s); lines.append(s)


def rho_hat(rho, gamma, vs):
    """Risk-adjusted rate of impatience rho - gamma (gamma - 1) varsigma^2 / 2: the single change the risky
    return makes to the deterministic equation for the growth rate."""
    return rho - gamma * (gamma - 1) * vs ** 2 / 2


def growth_risky(r, rho, gamma, theta, vs):
    """Growth rate with the risky return: the deterministic root with rho -> rho_hat. linear.growth brackets the
    root in (max(0, g_R), r) with g_R = (r - rho_hat)/gamma; when rho_hat < 0 the pole of the left-hand side,
    g = -rho_hat/gamma, lies below g_R, so the bracket excludes the spurious second root below the pole. The
    finiteness condition rho_hat > (1 - gamma) g is asserted."""
    rh = rho_hat(rho, gamma, vs)
    g = linear.growth(r, rh, gamma, theta)
    if rh < 0:
        assert g > -rh / gamma, (r, rho, gamma, theta, vs, g)       # above the pole
    assert rh > (1 - gamma) * g, ('value not finite', r, rho, gamma, theta, vs, g)
    return g


def hjb_residual(r, rho, gamma, theta, vs, a_values=(0.1, 1.0, 10.0)):
    """Residual of the HJB equation at the conjectured solution V = A a^(1-gamma) + B, with A fixed by the
    envelope condition at the root g of the modified closed form (rho -> rho_hat) and the maximization over c
    done numerically (not by the first-order condition). Returns the largest relative residual and the largest
    gap between the numerical maximizer and z = r - g."""
    g = growth_risky(r, rho, gamma, theta, vs); z = r - g
    A = (z ** (-gamma) - theta * g ** (-gamma)) / (1 - gamma)
    B = -(1 + theta) / ((1 - gamma) * rho)
    V = lambda a: A * a ** (1 - gamma) + B
    Vp = lambda a: (1 - gamma) * A * a ** (-gamma)
    Vpp = lambda a: -gamma * (1 - gamma) * A * a ** (-gamma - 1)
    U = lambda c, s: (c ** (1 - gamma) - 1) / (1 - gamma) + theta * (s ** (1 - gamma) - 1) / (1 - gamma)
    worst_res, worst_z = 0.0, 0.0
    for a in a_values:
        obj = lambda c: -(U(c, r * a - c) + Vp(a) * (r * a - c))
        res = minimize_scalar(obj, bounds=(1e-9 * r * a, r * a * (1 - 1e-9)), method='bounded', options={'xatol': 1e-13})
        rhs = -res.fun + 0.5 * vs ** 2 * a ** 2 * Vpp(a)
        worst_res = max(worst_res, abs(rho * V(a) - rhs) / abs(rho * V(a)))
        worst_z = max(worst_z, abs(res.x / a - z))
    return worst_res, worst_z


if __name__ == '__main__':
    t0 = time.time()
    log('Part 1: residual of the HJB equation at the conjectured solution (r=5%, rho=3%, gamma=2), with the')
    log('        maximization over c done numerically at a = 0.1, 1, 10; z = c/a in %')
    for theta, vs in ((0.2, 0.0), (0.2, 0.10), (0.2, 0.15), (0.05, 0.15), (0.0, 0.15)):
        g = growth_risky(R, RHO, GAMMA, theta, vs)
        res, dz = hjb_residual(R, RHO, GAMMA, theta, vs)
        log(f'  theta={theta:<5g} varsigma={vs:<5g}: z = {100*(R-g):.4f}; max relative HJB residual {res:.1e}; '
            f'max |numerical maximizer - z| = {100*dz:.1e} pp')
    log('')
    log('Part 2: precautionary response of the growth rate, g(varsigma) - g(0), percentage points; Ramsey value (gamma-1) varsigma^2/2')
    for vs in (0.10, 0.15, 0.20):
        row = []
        for theta in (0.0, 0.05, 0.2, 1.0):
            g0 = growth_risky(R, RHO, GAMMA, theta, 0.0); g1 = growth_risky(R, RHO, GAMMA, theta, vs)
            d = linear.solve(R, rho_hat(RHO, GAMMA, vs), GAMMA, theta)
            row.append(f'theta={theta:g}: {100*(g1-g0):.3f} (tau={d["tau"]:.2f})')
        log(f'  varsigma={vs:g}: ' + '; '.join(row))
    log('')
    log('Part 2b: rho_hat, the pole -rho_hat/gamma of the closed form when rho_hat < 0, the root used, and the finiteness margin')
    for vs in (0.10, 0.15, 0.20):
        rh = rho_hat(RHO, GAMMA, vs)
        row = []
        for theta in (0.0, 0.05, 0.2, 1.0):
            g = growth_risky(R, RHO, GAMMA, theta, vs)
            row.append(f'theta={theta:g}: g={100*g:.3f}%, rho_hat-(1-gamma)g={100*(rh-(1-GAMMA)*g):.3f}pp')
        pole = f'pole at g={100*(-rh/GAMMA):.3f}%' if rh < 0 else 'no pole in (0, r)'
        log(f'  varsigma={vs:g}: rho_hat={100*rh:.2f}%; {pole}; ' + '; '.join(row))
    log('')
    log('Part 3: local sensitivity dg/d(rho_hat) = -dg/d(precaution): theta=0 gives -1/gamma')
    for theta in (0.0, 0.05, 0.2, 1.0):
        d = linear.solve(R, RHO, GAMMA, theta)
        log(f'  theta={theta:g}: dg/drho = {linear.dg_drho(d["g"], R, RHO, GAMMA, theta):.3f}')
    log(f'elapsed {time.time()-t0:.0f} s')
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
