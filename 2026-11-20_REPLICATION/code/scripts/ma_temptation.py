"""Temptation with a mental account (Gul and Pesendorfer 2004, DSC preferences) against the joy of saving.

Setting of prop:gains: yield y, gain rate x, r = y + x; active saving s = y a - c; da/dt = r a - c.
Mental-accounting rule: the principal is illiquid, so the menu at each date is c in (0, y a]
(only current income is liquid). GP's recursion for a deterministic consumption problem is
W(a) = max_{c in menu} {u(c) + v(c) + delta W(a')} - v(max menu)  (GP 2004, pp. 119, 134).
Continuous-time analogue used here (flow menu): max int e^{-rho t}[u(c) - phi (v(y a) - v(c))] dt.

Parts:
 A. homothetic temptation v = lambda u (closed form) in continuous time;
 B. linear temptation v(c) = lambda c (satisfies GP's Section 6 assumptions: v increasing, convex,
    u + v concave), solved by a finite-difference HJB scheme; the scheme is first validated on the
    joy of saving against its closed form;
 C. discrete-time GP recursion: value of the rule (illiquid principal) against a liquid principal
    (C2, linear v: value function iteration with continuous choice of consumption).
Output: code/out/ma/ma_temptation.txt
"""
import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, os.path.dirname(__file__))
from ma_hjb import solve_hjb, up_inv
from ma_common import (jos_h, jos_dh_dx, pv_iso, u_iso, z_ramsey, cramsey_policy, cramsey_value)

OUT = os.path.join(os.path.dirname(__file__), '..', 'out', 'ma', 'ma_temptation.txt')
lines = []


def P(s=''):
    print(s)
    lines.append(s)


G, RHO, R = 2.0, 0.03, 0.05
THETA = 0.2

# =====================================================================================  A. homothetic
P('A. Temptation with v = psi*u (homothetic), continuous time, gamma=2, rho=3%, r=5%.')
P('   Plan c = z a; slack: gamma z - kappa y^(1-gamma) z^gamma = rho - (1-gamma) r, kappa = psi/(1+psi);')
P('   binding (z > y): z = y. Checked against a direct maximization of the reduced HJB.')


def gp_hom_z(y, r, rho, g, psi):
    kap = psi / (1 + psi)
    target = rho - (1 - g) * r
    Phi = lambda z: g * z - kap * y ** (1 - g) * z ** g - target
    # Phi convex for g<1, increasing on (0,y] for g>=1; root in (0, y] or none (then binding)
    if Phi(y) <= 0:
        return y, True
    zlo = 1e-12
    if Phi(zlo) > 0:  # cannot happen for target > 0
        raise ValueError
    return brentq(Phi, zlo, y, xtol=1e-15, rtol=1e-15), False


def gp_hom_value(a, y, r, rho, g, psi):
    z, _ = gp_hom_z(y, r, rho, g, psi)
    grow = r - z
    return (1 + psi) * pv_iso(z * a, grow, rho, g) - psi * pv_iso(y * a, grow, rho, g)


def gp_hom_check(y, r, rho, g, psi):
    """Direct maximization of the reduced HJB over z for V = K a^(1-g)/(1-g) (fixed point in K)."""
    def best(K):
        f = lambda z: -(((1 + psi) * z ** (1 - g) - psi * y ** (1 - g)) / (1 - g) + K * (r - z))
        res = minimize_scalar(f, bounds=(1e-10, y), method='bounded', options=dict(xatol=1e-14))
        return res.x, -res.fun
    # rho K/(1-g) = max_z {...}: solve for K by bisection on the residual
    def resid(K):
        return rho * K / (1 - g) - best(K)[1]
    Ks = np.logspace(-3, 6, 400)
    vals = [resid(K) for K in Ks]
    for i in range(len(Ks) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]):
            K = brentq(resid, Ks[i], Ks[i + 1], xtol=1e-13, rtol=1e-14)
            return best(K)[0]
    return np.nan


P(f'{"psi":>6} {"y":>6} {"z_GP":>10} {"bind":>5} {"z check":>10} {"dz/dy|r":>9} {"z_JoS":>9} {"MPCform JoS":>12}')
for psi in [0.05, 0.2, 1.0]:
    for y in [0.01, 0.02, 0.03, 0.04, 0.045, 0.05]:
        z, b = gp_hom_z(y, R, RHO, G, psi)
        zc = gp_hom_check(y, R, RHO, G, psi)
        dy = 1e-6
        if y + dy <= R:
            dzdy = (gp_hom_z(y + dy, R, RHO, G, psi)[0] - gp_hom_z(y - dy, R, RHO, G, psi)[0]) / (2 * dy)
        else:
            dzdy = (gp_hom_z(y, R, RHO, G, psi)[0] - gp_hom_z(y - dy, R, RHO, G, psi)[0]) / dy
        x = R - y
        hj = jos_h(THETA, y, x, RHO, G)
        mj = 1 + jos_dh_dx(THETA, y, x, RHO, G)
        P(f'{psi:6.2f} {y:6.3f} {z:10.6f} {str(b):>5} {zc:10.6f} {dzdy:9.4f} {y-hj:9.6f} {mj:12.4f}')
P('   (dz/dy at a given r is the MPC out of a shift of the return from gains to yield; JoS at theta=0.2.)')

P()
P('   Sign of dz/dy|r when the rule is slack: sign(1 - gamma) for v = psi*u. Check with gamma = 0.5 and 2:')
for g in [0.5, 2.0]:
    psi = 0.2
    rho = 0.03
    for y in [0.045, 0.05]:
        z, b = gp_hom_z(y, R, rho, g, psi)
        dy = 1e-6
        dzdy = (gp_hom_z(min(y + dy, R), R, rho, g, psi)[0] - gp_hom_z(y - dy, R, rho, g, psi)[0]) / \
            (min(y + dy, R) - (y - dy))
        P(f'   gamma={g}: y={y}: z={z:.6f} binding={b} dz/dy|r={dzdy:+.5f}')

P()
P('   Welfare against the split of a given return r=5% (a=1): GP-MA W(y) and JoS V(y) (theta=0.2).')
P('   Bound: W(y) <= constrained Ramsey V0(y) <= Ramsey V_R, equality iff y = z_R = 4%.')
VR = pv_iso(z_ramsey(R, RHO, G), R - z_ramsey(R, RHO, G), RHO, G)
P(f'   Ramsey V_R(1) = {VR:.6f}')
P(f'{"y":>7} {"W psi=.05":>12} {"W psi=.2":>12} {"W psi=1":>12} {"V0 (rule)":>12} {"JoS V":>12}')
ys = np.round(np.arange(0.005, 0.0501, 0.0025), 4)
best = {}
for y in ys:
    x = R - y
    row = [gp_hom_value(1.0, y, R, RHO, G, psi) for psi in [0.05, 0.2, 1.0]]
    v0 = cramsey_value(1.0, y, x, RHO, G)
    vc, vs, h = (lambda t: t)(__import__('ma_common').jos_value_parts(THETA, 1.0, y, x, RHO, G))
    P(f'{y:7.4f} ' + ' '.join(f'{w:12.4f}' for w in row) + f' {v0:12.4f} {vc + THETA * vs:12.4f}')
for psi in [0.05, 0.2, 1.0]:
    res = minimize_scalar(lambda y: -gp_hom_value(1.0, y, R, RHO, G, psi), bounds=(0.001, R), method='bounded',
                          options=dict(xatol=1e-12))
    P(f'   argmax_y W for psi={psi}: y*={res.x:.6f} (z_R={z_ramsey(R, RHO, G):.6f}); W(y*)={-res.fun:.6f}, '
      f'V_R={VR:.6f}')
P('   Shape of W(y) on a 401-point grid of y in [1%, 5%] (numerical; Prop. (ii) proves only the maximum,')
P('   and the increase below z_R holds in general because c = y a attains V0(y) there):')
ygr = np.linspace(0.01, 0.05, 401)
for psi in [0.05, 0.2, 1.0]:
    Wg = np.array([gp_hom_value(1.0, yy, R, RHO, G, psi) for yy in ygr])
    k = int(np.argmax(Wg))
    up = bool(np.all(np.diff(Wg[:k + 1]) > 0)); down = bool(np.all(np.diff(Wg[k:]) < 0))
    P(f'   psi={psi}: peak at y={ygr[k]:.5f}, W={Wg[k]:.6f}; strictly increasing below: {up}; '
      f'strictly decreasing above: {down}')

# =====================================================================================  B. HJB, linear v
P()
P('B. Finite-difference HJB: rho V = max_{0<c<=y a} F(c,a) + V\'(a)(r a - c), upwind (drift >= 0),')
P('   implicit iteration, log grid, Dirichlet value at the top node from the asymptotic plan.')


# ---- validation on the joy of saving (closed form)
def jos_policy_factory(theta, y, g):
    def pol(dV, a):
        # solve u'(c) - theta w'(y a - c) = dV for c in (0, y a); LHS decreasing from +inf to -inf
        lo = np.full_like(a, 1e-300); hi = y * a * (1 - 1e-14)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f = mid ** (-g) - theta * (y * a - mid) ** (-g) - dV
            lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
        return 0.5 * (lo + hi)
    return pol


def jos_value_factory(theta, y, g):
    return lambda c, a: u_iso(c, g) + theta * u_iso(y * a - c, g)


for (y, x) in [(0.05, 0.0), (0.03, 0.02)]:
    vc, vs, h = __import__('ma_common').jos_value_parts(THETA, 1.0, y, x, RHO, G)
    Vex = lambda aa: (vc + 1 / ((1 - G) * RHO) + THETA * (vs + 1 / ((1 - G) * RHO))) * aa ** (1 - G) \
        - (1 + THETA) / ((1 - G) * RHO)
    for N in [1500, 3000, 6000]:
        a = np.exp(np.linspace(np.log(0.1), np.log(1e4), N))
        V0 = Vex(a) * 1.05
        V, c, it, err = solve_hjb(jos_policy_factory(THETA, y, G), jos_value_factory(THETA, y, G), y, R, RHO, a,
                                  Vex(a[-1]), V0)
        i1 = np.searchsorted(a, 1.0)
        m = a <= 10
        P(f'   JoS validation y={y}, x={x}, N={N}: c/a at a={a[i1]:.3f}: HJB {c[i1]/a[i1]:.6f}, closed form '
          f'{y-h:.6f}; max |V/Vex - 1| on a<=10: {np.max(np.abs(V[m]/Vex(a[m]) - 1)):.2e}; iterations {it}')

# ---- linear temptation


def gp_lin_policy_factory(lam, g):
    def pol(dV, a):
        m = dV - lam
        return np.where(m > 0, np.maximum(m, 1e-300) ** (-1 / g), np.inf)
    return pol


def gp_lin_value_factory(lam, y, g):
    return lambda c, a: u_iso(c, g) + lam * c - lam * y * a


def solve_gp_lin(y, lam, a):
    x = R - y
    Vtop = pv_iso(y * a[-1], x, RHO, G)  # binding plan at the top: c = y a, a grows at x, no temptation cost
    V0 = np.array([pv_iso(y * ai, x, RHO, G) for ai in a])
    V, c, it, err = solve_hjb(gp_lin_policy_factory(lam, G), gp_lin_value_factory(lam, y, G), y, R, RHO, a,
                              Vtop, V0)
    return V, c, it, err


for LAM in [2.0, 40.0]:
    P()
    P(f'   Linear temptation v(c) = lambda c, lambda = {LAM:g} (u\'(c) = c^-2 equals lambda at c = {LAM**-0.5:.4f}).')
    P('   Rule: c <= y a. MPC left/right = one-sided d(c/a)/dy at given r (dy = 0.0005); JoS at theta = 0.2.')
    a = np.exp(np.linspace(np.log(0.05), np.log(2e3), 1600))
    sol = {}
    for y in [0.0295, 0.03, 0.0305, 0.0395, 0.04, 0.0405, 0.0495, 0.05]:
        sol[y] = solve_gp_lin(y, LAM, a)
        P(f'   solved y={y}: iterations {sol[y][2]}, last change {sol[y][3]:.1e}; binding at top node: '
          f'{abs(sol[y][1][-2]/(y*a[-2]) - 1) < 1e-9}')
    # robustness to the top of the grid
    a2 = np.exp(np.linspace(np.log(0.05), np.log(2e4), 1900))
    V2, c2, _, _ = solve_gp_lin(0.04, LAM, a2)
    V1, c1 = sol[0.04][0], sol[0.04][1]
    for aa in [0.2, 1.0, 5.0, 50.0]:
        i, j = np.searchsorted(a, aa), np.searchsorted(a2, aa)
        P(f'   grid top 2e3 vs 2e4 at a~{aa}: c/a {c1[i]/a[i]:.6f} vs {c2[j]/a2[j]:.6f}')
    P(f'{"y":>7} {"a":>8} {"c/a":>9} {"bind":>5} {"MPC left":>9} {"MPC right":>9} {"JoS c/a":>9} {"JoS MPC":>9}')
    for yc in [0.03, 0.04, 0.05]:
        for aa in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
            i = np.searchsorted(a, aa)
            ca = sol[yc][1][i] / a[i]
            bind = abs(ca / yc - 1) < 1e-9
            yp, ym = round(yc + 0.0005, 4), round(yc - 0.0005, 4)
            mleft = (sol[yc][1][i] / a[i] - sol[ym][1][i] / a[i]) / (yc - ym)
            mright = (sol[yp][1][i] / a[i] - sol[yc][1][i] / a[i]) / (yp - yc) if yp in sol else np.nan
            hj = jos_h(THETA, yc, R - yc, RHO, G)
            mj = 1 + jos_dh_dx(THETA, yc, R - yc, RHO, G)
            P(f'{yc:7.4f} {a[i]:8.3f} {ca:9.6f} {str(bind):>5} {mleft:9.4f} {mright:9.4f} {yc-hj:9.6f} {mj:9.4f}')
    # threshold wealth
    for yc in [0.03, 0.04, 0.05]:
        c = sol[yc][1]
        nb = np.where(np.abs(c / (yc * a) - 1) > 1e-9)[0]
        thr = a[nb[-1] + 1] if len(nb) else a[0]
        P(f'   y={yc}: rule binds for all a >= {thr:.4f} on the grid (slack below)')

    # welfare against y at fixed r, linear v
    P()
    P('   Welfare W(a; y) at a given r (linear v): argmax over the split, and the bound W <= V0(rule) <= V_R.')
    ygrid = np.round(np.arange(0.01, 0.0501, 0.005), 4)
    Wy = {}
    for y in ygrid:
        if y not in sol:
            sol[y] = solve_gp_lin(y, LAM, a)
        Wy[y] = sol[y][0]
    for aa in [0.3, 1.0, 10.0]:
        i = np.searchsorted(a, aa)
        vals = [Wy[y][i] for y in ygrid]
        k = int(np.argmax(vals))
        v0s = [cramsey_value(a[i], y, R - y, RHO, G) for y in ygrid]
        ok = all(Wy[y][i] <= v0 + 1e-6 * abs(v0) for y, v0 in zip(ygrid, v0s))
        P(f'   a={a[i]:.3f}: W(y) for y=' + ', '.join(f'{y:.3f}:{v:.3f}' for y, v in zip(ygrid, vals)))
        P(f'          argmax on grid y={ygrid[k]:.3f}; W<=V0 at every y: {ok}')

# =====================================================================================  C. discrete GP
P()
P('C. Discrete-time GP recursion (annual), value of the rule (illiquid principal) against a liquid principal.')
P('   W(a) = max_{c<=m(a)} {u(c)+v(c)+delta W(R a - c)} - v(m(a)), R = 1+r, delta = exp(-rho);')
P('   rule: m(a) = y a; liquid: m(a) = R a. Homothetic v = psi*u: W = K a^(1-g)/(1-g) + B, K by fixed point.')
P('   omega = wealth multiple that makes the liquid consumer as well off as the rule-follower (omega>1: rule valuable).')
Rg = 1 + R
delta = np.exp(-RHO)


def disc_hom(m, psi, g=G):
    # K solves K = (1-g) max_{z<=m} {[(1+psi) z^(1-g) + delta K (Rg - z)^(1-g)]/(1-g)} - psi m^(1-g)
    K = 1.0
    for _ in range(20000):
        f = lambda z: -(((1 + psi) * z ** (1 - g) + delta * K * (Rg - z) ** (1 - g)) / (1 - g))
        res = minimize_scalar(f, bounds=(1e-10, min(m, Rg - 1e-10)), method='bounded', options=dict(xatol=1e-14))
        Kn = (1 - g) * (-res.fun) - psi * m ** (1 - g)
        if abs(Kn - K) < 1e-13 * abs(K):
            K = Kn
            break
        K = Kn
    return K, res.x


for y in [0.02, 0.03, 0.04, 0.05]:
    for psi in [0.0, 0.05, 0.2, 1.0]:
        Kr, zr = disc_hom(y, psi)
        Kl, zl = disc_hom(Rg, psi)
        Kc, zc = disc_hom(Rg, 0.0)  # commitment benchmark (no temptation, liquid)
        omega = (Kr / Kl) ** (1 / (1 - G))
        P(f'   y={y:.2f} psi={psi:4.2f}: c/a rule={zr:.6f} (binds: {abs(zr-y)<1e-7}), c/a liquid={zl:.6f}, '
          f'Ramsey c/a={zc:.6f}; omega={omega:.6f}')
P()
P('   C2. Linear temptation v(c) = lambda c (GP Sec. 6 assumptions hold), whole return paid as yield (y = r),')
P('   so the rule is "never sell principal": a\' >= a. Value function iteration with CONTINUOUS choice of c:')
P('   golden-section search on a natural cubic spline of W in log wealth, extended linearly (with the')
P('   spline\'s end slopes) beyond the grid, Howard steps, three grids. Policies and values are evaluated at the exact')
P('   wealth levels below from the converged spline. (The earlier version restricted next-period wealth to')
P('   grid nodes, so c/a could take only the values 0.0500, 0.0423, 0.0346, ...; its consumption figures')
P('   were a grid artefact.) omega = liquid wealth / a that gives the liquid consumer the rule-follower\'s value.')
from scipy.interpolate import CubicSpline

GOLD = (np.sqrt(5) - 1) / 2


def golden_max(f, lo, hi, n=100):
    a_, b_ = lo.copy(), hi.copy()
    x1 = b_ - GOLD * (b_ - a_); x2 = a_ + GOLD * (b_ - a_)
    f1, f2 = f(x1), f(x2)
    for _ in range(n):
        left = f1 > f2
        b_ = np.where(left, x2, b_); a_ = np.where(left, a_, x1)
        nx1 = np.where(left, b_ - GOLD * (b_ - a_), x2); nx2 = np.where(left, x1, a_ + GOLD * (b_ - a_))
        nf1 = np.where(left, f(nx1), f2); nf2 = np.where(left, f1, f(nx2))
        x1, x2, f1, f2 = nx1, nx2, nf1, nf2
    c = 0.5 * (a_ + b_)
    fc, fh = f(c), f(hi)
    return np.where(fh >= fc, hi, c), np.maximum(fc, fh)


def spline_lin_ext(lg, Wv):
    """Natural cubic spline of W in log wealth inside the grid, linear outside with the end slopes."""
    cs = CubicSpline(lg, Wv, bc_type='natural')
    d0, d1 = cs(lg[0], 1), cs(lg[-1], 1)

    def f(x):
        x = np.asarray(x, float)
        out = cs(np.clip(x, lg[0], lg[-1]))
        return np.where(x < lg[0], Wv[0] + d0 * (x - lg[0]), np.where(x > lg[-1], Wv[-1] + d1 * (x - lg[-1]), out))
    return f


def disc_lin_vfi(lam, rule, agrid, tol=1e-12, maxit=3000):
    """Continuous-choice VFI for W(a) = max_{0<c<=m(a)} {u(c) + lam c + delta W(Rg a - c)} - lam m(a)."""
    lg = np.log(agrid)
    m = R * agrid if rule else Rg * agrid
    cmax = m if rule else Rg * agrid - agrid[0]   # liquid: a' stays on the grid
    cmin = np.full_like(agrid, 1e-8)
    W = u_iso(R * agrid, G) / (1 - delta)          # start: consume the return forever

    def obj(Wv, aa):
        cs = spline_lin_ext(lg, Wv)
        return lambda c: u_iso(c, G) + lam * c + delta * cs(np.log(Rg * aa - c))
    d = np.nan
    for it in range(maxit):
        c, fmax = golden_max(obj(W, agrid), cmin, cmax)
        Wn = fmax - lam * m
        for _ in range(30):                        # Howard steps at the current policy
            Wn = obj(Wn, agrid)(c) - lam * m
        d = np.max(np.abs(Wn - W) / (1 + np.abs(W)))
        W = Wn
        if d < tol:
            break
    cs = spline_lin_ext(lg, W)

    def at(apts):
        apts = np.asarray(apts, float)
        mm = R * apts if rule else Rg * apts
        hi = mm if rule else Rg * apts - agrid[0]
        c, fmax = golden_max(obj(W, apts), np.full_like(apts, 1e-8), hi)
        return c, fmax - lam * mm
    return W, cs, at, it, d


APTS = np.array([0.3, 1.0, 3.0, 10.0])
C2 = {}
for (alo, ahi, NN) in [(1e-3, 1e4, 800), (1e-3, 1e4, 1600), (1e-3, 1e5, 1600)]:
    ag = np.exp(np.linspace(np.log(alo), np.log(ahi), NN))
    P(f'   grid [{alo:g}, {ahi:g}], {NN} points (log step {np.log(ahi / alo) / (NN - 1):.5f}):')
    for lam in [0.0, 2.0, 10.0, 40.0]:
        Wr, csr, atr, itr, dr = disc_lin_vfi(lam, True, ag)
        Wl, csl, atl, itl, dl = disc_lin_vfi(lam, False, ag)
        cr, vr = atr(APTS)
        cl, vl = atl(APTS)
        inc = bool(np.all(np.diff(Wl) > 0))
        P(f'   lambda={lam:g}: iterations {itr},{itl}, last relative change {dr:.1e},{dl:.1e}; W_liq increasing on '
          f'the grid: {inc}; W_liq at the top node {Wl[-1]:.3f}')
        for k, aa in enumerate(APTS):
            if vr[k] <= Wl[-1]:
                ae = brentq(lambda la: csl(la) - vr[k], np.log(ag[0]), np.log(ag[-1]), xtol=1e-13)
                oms = f'{np.exp(ae) / aa:.4f}'
            else:
                oms = 'none on the grid (W_rule(a) > W_liq at the top node)'
            binds = abs(cr[k] / (R * aa) - 1) < 1e-7
            P(f'      a={aa:g}: c/a rule {cr[k]/aa:.5f} (binds {binds}), c/a liquid {cl[k]/aa:.5f}, omega {oms}')
            C2[(NN, ahi, lam, aa)] = (cr[k] / aa, cl[k] / aa, oms)
P(f'   Ramsey c/a in discrete time (lambda = 0, exact): Rg - (delta Rg)^(1/gamma) = '
  f'{Rg - (delta * Rg) ** (1 / G):.5f}')
P('   Note: with psi > 0 and a slack rule (y=0.05), the rule changes consumption although it does not bind:')
P('   removing a non-binding constraint changes behavior and welfare (GP 2004, abstract and Sec. 6.3).')
P('   Under the joy of saving with w\'(0)=inf the rule s>=0 never binds and its value is exactly zero.')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print('written', OUT)
