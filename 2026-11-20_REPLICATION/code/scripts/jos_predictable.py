"""Predictable income change in the linear joy-of-saving model (independent check of the
fixed-costate formula dc/dw = U_ss/(U_cc+U_ss) = tau q/(1+tau q)).

Linear technology, total wealth a (financial plus human at base w), da/dt = r a + e(t) - c,
s = r a + e(t) - c, with e(t) = E on [t1, t1+T] (announced at 0) and 0 otherwise.
Pontryagin: c^-G - th s^-G = lam, dlam/dt = rho lam - r c^-G.  After t1+T the problem is the
stationary one, V'(a) = u'(z*) a^-G with u'(z) = z^-G - th (r - z)^-G.  Shoot on lam(0).
Also: within-period response to a lump payment P at fixed costate (period accounting).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

r, rho, G = 0.05, 0.03, 2.0

def g_root(th):
    F = lambda g: 1 - r / (G * g + rho) - th * ((r - g) / g) ** G
    return brentq(F, max(0, (r - rho) / G) + 1e-14, r - 1e-14, xtol=1e-16)

def c_of(lam, inc, th):
    f = lambda c: c ** -G - th * (inc - c) ** -G - lam
    return brentq(f, 1e-14 * inc, inc * (1 - 1e-14), xtol=1e-18, rtol=1e-15, maxiter=500)

def run(th, E, t1=1.0, T=0.25, a0=1.0):
    g = g_root(th); z = r - g
    up = z ** -G - th * (r - z) ** -G
    def rhs(t, y, e):
        a, lam = y
        c = c_of(lam, r * a + e, th)
        return [r * a + e - c, rho * lam - r * c ** -G]
    def shoot(lam0, full=False):
        y = [a0, lam0]; out = []
        for (ta, tb, e) in ((0, t1, 0.0), (t1, t1 + T, E)):
            sol = solve_ivp(rhs, (ta, tb), y, args=(e,), rtol=1e-12, atol=1e-15, dense_output=True)
            y = sol.y[:, -1]; out.append(sol)
        a_end, lam_end = y
        return (lam_end - up * a_end ** -G) if not full else out
    lam_base = up * a0 ** -G
    lam0 = brentq(shoot, lam_base * 0.9, lam_base * 1.1, xtol=1e-16, rtol=1e-14)
    s1, s2 = shoot(lam0, full=True)
    a_t1, lam_t1 = s1.y[:, -1]
    c_before = c_of(lam_t1, r * a_t1, th)
    c_after = c_of(lam_t1, r * a_t1 + E, th)
    return (c_after - c_before) / E, c_before, r * a_t1


def main():
    print("== Anticipated temporary income change, full dynamic solution vs fixed-costate formula ==")
    for th in (0.05, 0.2):
        g = g_root(th); q = (r - g) / g; tau = th * q ** G
        formula = tau * q / (1 + tau * q)
        for frac in (0.01, 0.07):        # change as a share of income (Parker: about 7 per cent)
            E = frac * r * 1.0
            jump, c, Y = run(th, E)
            print(f"theta={th} dw/Y={frac:.2f}: jump dc/dw = {jump:.4f}  formula {formula:.4f}  "
                  f"elasticity = {jump * Y / c:.4f} (formula x Y/c = {formula * r / (r - g):.4f})")

    print("\n== Lump payment P within a period, costate fixed: MPC = [c(P) - c(0)]/P ==")
    for th in (0.2,):
        g = g_root(th); c, s = r - g, g        # per unit wealth, per year; homogeneous, so period length scales out
        lam = c ** -G - th * s ** -G
        for Pc in (0.01, 0.25, 0.5, 1.0, 2.0):
            P = Pc * c
            cP = c_of(lam, c + s + P, th)
            print(f"theta={th} P/c={Pc:5.2f}: MPC={(cP - c) / P:.4f}")
    print("max within-period MPC over theta (small P), gamma = 2, 4, 6, 10:")
    for GG in (2.0, 4.0, 6.0, 10.0):
        best = 0.0
        for th in np.logspace(-6, 4, 3001):
            F = lambda gg: 1 - r / (GG * gg + rho) - th * ((r - gg) / gg) ** GG
            try:
                gg = brentq(F, max(0, (r - rho) / GG) + 1e-14, r - 1e-14, xtol=1e-16)
            except ValueError:
                continue
            q = (r - gg) / gg; tau = th * q ** GG
            best = max(best, tau * q / (1 + tau * q))
        print(f"gamma={GG:.0f}: max MPC = {best:.4f}")


if __name__ == '__main__':
    main()
