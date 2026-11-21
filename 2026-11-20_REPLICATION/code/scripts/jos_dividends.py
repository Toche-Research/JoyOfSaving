"""Dividend versus capital gain at a given total return, Joy of Saving (Proposition gains).

Model: wealth a at market value, yield y, gain rate x, r = y + x.  s = y a - c, da/dt = s + x a.
U(c,s) = [c^(1-g) + th s^(1-g)]/(1-g).  On the balanced path c = (y - h) a, s = h a,
h root of (G h + rho' - y)/(G h + rho') = th ((y-h)/h)^G with rho' = rho + (G-1) x.

Outputs:
 1. permanent effect: dc/d(ya) at fixed r = 1 + dh/dx  (per unit of dividend income)
 2. checks: closed-form root vs direct Bellman maximization; analytic dh/dx vs finite difference
 3. transitory effect at fixed costate: dc/d(income) = tau q/(1 + tau q), q = c/s
 4. Ramsey / any U(c,k): zero (shown from the reduced problem: depends on r only)
 5. inflation: nominal asset, cash yield i = r + pi, real gain x = -pi
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar

def h_root(y, x, rho, G, th):
    rp = rho + (G - 1) * x
    F = lambda h: 1 - y / (G * h + rp) - th * ((y - h) / h) ** G
    lo = max(0.0, (y - rp) / G) + 1e-14
    return brentq(F, lo, y - 1e-14, xtol=1e-16, rtol=1e-15, maxiter=500)

def h_bellman(y, x, rho, G, th):
    """Independent check.  V(a) = A a^(1-G).  rho A = max_z {u(z) + (1-G) A (r - z)},
    u(z) = [z^(1-G) + th (y - z)^(1-G)]/(1-G), c = z a, s = (y - z) a, da/dt = (r - z) a.
    Solve for A by root-finding on the Bellman residual; return h = y - z*."""
    r = y + x
    u = lambda z: (z ** (1 - G) + th * (y - z) ** (1 - G)) / (1 - G)
    def zstar(A):
        res = minimize_scalar(lambda z: -(u(z) + (1 - G) * A * (r - z)),
                              bounds=(1e-12, y - 1e-12), method='bounded',
                              options={'xatol': 1e-15, 'maxiter': 2000})
        return res.x, -res.fun
    def resid(A):
        return rho * A - zstar(A)[1]
    # A < 0 for G > 1; bracket
    lo, hi = -1e6, -1e-6
    A = brentq(resid, lo, hi, xtol=1e-14, maxiter=500)
    return y - zstar(A)[0]

def dhdx_analytic(y, x, rho, G, th):
    h = h_root(y, x, rho, G, th)
    rp = rho + (G - 1) * x
    D = G * h + rp
    q = (y - h) / h
    Gh = G * y / D**2 + th * G * q**(G - 1) * y / h**2
    Gy = -1 / D - th * G * q**(G - 1) / h
    Grp = y / D**2
    return (Gy - (G - 1) * Grp) / Gh   # at fixed r: dy/dx = -1, drho'/dx = G-1

def dhdx_fd(r, x, rho, G, th, e=1e-7):
    f = lambda xx: h_root(r - xx, xx, rho, G, th)
    return (f(x + e) - f(x - e)) / (2 * e)

def transitory(y, x, rho, G, th):
    h = h_root(y, x, rho, G, th)
    c, s = y - h, h
    q = c / s
    tau = th * q**G
    return tau * q / (1 + tau * q), tau, q

r, rho, G = 0.05, 0.03, 2.0


def main():
    print("== Calibration r=5%, rho=3%, gamma=2 ==")
    for th in (0.05, 0.2, 1.0):
        print(f"-- theta={th}")
        for x in (0.0, 0.01, 0.02):
            y = r - x
            h = h_root(y, x, rho, G, th)
            hb = h_bellman(y, x, rho, G, th)
            da = dhdx_analytic(y, x, rho, G, th)
            df = dhdx_fd(r, x, rho, G, th)
            print(f"x={x:.2f} y={y:.2f}  h={h*100:.4f}%  c/a={(y-h)*100:.4f}%  |h-h_bellman|={abs(h-hb):.1e}"
                  f"  dh/dx={da:.4f} (fd {df:.4f})  dc/d(ya)|r = 1+dh/dx = {1+da:.4f}")
        # average over x in [0, 0.02] of 1 + dh/dx = 1 - [h(0) - h(0.02)]/0.02
        h0 = h_root(r, 0.0, rho, G, th); h2 = h_root(r - 0.02, 0.02, rho, G, th)
        print(f"   average over 2 points: displacement {(h0-h2)/0.02:.4f}, consumption effect {1-(h0-h2)/0.02:.4f}")

    print("\n== Ramsey and any U(c,k): problem depends on r only; c=(r-gR)a for Ramsey ==")
    for x in (0.0, 0.02):
        gR = (r - rho) / G
        print(f"x={x}: Ramsey c/a={(r-gR)*100:.4f}%, active saving s/a = gR - x = {(gR-x)*100:.4f}%  -> dc/dy|r = 0")

    print("\n== Transitory change in the income flow at fixed costate: tau q/(1+tau q) ==")
    for th in (0.05, 0.2, 1.0):
        m, tau, q = transitory(r, 0.0, rho, G, th)
        Y_over_c = r / (r - h_root(r, 0.0, rho, G, th))
        print(f"theta={th}: tau={tau:.4f} q=c/s={q:.4f}  dc/dY={m:.4f}  elasticity dlnc/dlnY={m*Y_over_c:.4f}")
    print("max over theta of tau q/(1+tau q) and of the elasticity, x=0:")
    for GG in (1.0001, 2.0, 4.0):
        ths = np.logspace(-4, 4, 4001)
        ms, es = [], []
        for th in ths:
            m, tau, q = transitory(r, 0.0, rho, GG, th)
            h = h_root(r, 0.0, rho, GG, th)
            ms.append(m); es.append(m * r / (r - h))
        i, j = int(np.argmax(ms)), int(np.argmax(es))
        print(f"gamma={GG:.0f}: max dc/dY={ms[i]:.4f} at theta={ths[i]:.3g}; max elasticity={es[j]:.4f} at theta={ths[j]:.3g}")
    # analytic bound: tau q/(1+tau q) with tau<1 ... print limits theta->inf
    for GG in (2.0,):
        for th in (1e2, 1e4, 1e6):
            m, tau, q = transitory(r, 0.0, rho, GG, th)
            print(f"  gamma=2 theta={th:g}: dc/dY={m:.4f}, tau={tau:.4f}, q={q:.5f}")

    print("\n== Inflation: nominal bond, real rate r, cash yield i=r+pi, real gain x=-pi ==")
    for th in (0.05, 0.2):
        for pi in (0.0, 0.02, 0.04):
            x = -pi; y = r + pi
            h = h_root(y, x, rho, G, th)
            da = dhdx_analytic(y, x, rho, G, th)
            print(f"theta={th} pi={pi:.2f}: c/a={(y-h)*100:.4f}% (pi=0: {(r-h_root(r,0,rho,G,th))*100:.4f}%), "
                  f"dc/dpi per unit wealth = 1+dh/dx = {1+da:.4f}")

    print("\n== Di Maggio, Kermani and Majlesi (2020), Table III (IV): dividend minus capital-gain coefficient ==")
    groups = ["5-50", "50-70", "70-90", "90-95", "95-100"]
    div = [0.560, 0.601, 0.512, 0.587, 0.386]
    gain = [0.233, 0.068, 0.037, 0.027, 0.028]
    for gname, d, gg in zip(groups, div, gain):
        print(f"{gname}: {d:.3f} - {gg:.3f} = {d - gg:.3f}")


if __name__ == '__main__':
    main()
