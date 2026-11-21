"""Limit theta -> 0 (and theta -> infinity) of the joy-of-saving plan with yield y and gain rate x.

Checks the limit result of the section on mental accounting (the rule of mental accounting as theta -> 0):
 - h*(theta) -> max(0, h_R): the plan converges to the Ramsey plan under the rule c <= y a;
 - rates: O(theta) when the rule is slack, O(theta^(1/g)) when it binds, O(theta^(1/(1+g))) at the knife edge;
 - tau(theta) -> normalized multiplier of the rule (barrier analogy);
 - MPC out of a shift of the return from gains to yield, 1 + dh/dx, -> 1 (binding) or 0 (slack);
 - within-period MPC out of extra income at a fixed costate, tau q/(1+tau q), -> 1 or 0;
 - values converge; consumption-utility gap O(theta^2) slack, O(theta^(1/g)) binding;
 - independent check: direct numerical maximization over piecewise-constant (non-stationary) policies.
Output: code/out/ma/ma_limit.txt
"""
import os
import sys
import numpy as np
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, os.path.dirname(__file__))
from ma_common import (jos_h, jos_dh_dx, jos_tau, jos_value_parts, cramsey_policy, cramsey_value,
                       cramsey_multiplier, h_ramsey, z_ramsey, rho_prime, pv_iso, u_iso)

OUT = os.path.join(os.path.dirname(__file__), '..', 'out', 'ma', 'ma_limit.txt')
lines = []


def P(s=''):
    print(s)
    lines.append(s)


CASES = [
    # name, gamma, rho, r, x
    ('S  slack, all yield (paper calibration)', 2.0, 0.03, 0.05, 0.00),
    ('B1 binding, 2 points of gains (paper calibration)', 2.0, 0.03, 0.05, 0.02),
    ('B2 binding, impatient rho > r', 2.0, 0.05, 0.03, 0.00),
    ('K  knife edge z_R = y', 2.0, 0.03, 0.05, 0.01),
    ('K1 knife edge, gamma = 1 (z_R = y)', 1.0, 0.03, 0.05, 0.02),
    ('B1g4 binding, gamma = 4', 4.0, 0.03, 0.05, 0.02),
    ('Sg4 slack, gamma = 4', 4.0, 0.03, 0.05, 0.00),
]
THETAS = [1.0, 0.2, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


def mpc_form_fd(theta, r, x, rho, g, dx=1e-7):
    """1 + dh/dx at a given r by central finite difference (independent of the analytic formula)."""
    hp = jos_h(theta, r - (x + dx), x + dx, rho, g)
    hm = jos_h(theta, r - (x - dx), x - dx, rho, g) if x - dx >= 0 or True else None
    return 1 + (hp - hm) / (2 * dx)


P('Limit theta -> 0 of the joy of saving with yield y and gain rate x (prop:gains model).')
P('Columns: theta; h* (active saving/wealth); h* - h0 with h0 = max(0,h_R); tau = U_s/U_c;')
P('MPCform = 1 + dh/dx at given r (analytic | finite difference); MPCtr = tau q/(1+tau q), q = c/s;')
P('V = JoS value at a=1; V0 = constrained-Ramsey value at a=1; gapC = V0 - (consumption part of V) >= 0;')
P('thJ = theta * (joy part of V).')
summary = {}
for name, g, rho, r, x in CASES:
    y = r - x
    rhop = rho_prime(rho, g, x)
    hR = h_ramsey(r, rho, g, x)
    zR = z_ramsey(r, rho, g)
    z0, h0 = cramsey_policy(y, x, rho, g)
    m0 = cramsey_multiplier(y, x, rho, g)
    V0 = cramsey_value(1.0, y, x, rho, g)
    P()
    P(f'=== Case {name}: gamma={g}, rho={rho}, r={r}, x={x}, y={y:.4f}, rho\'={rhop:.4f}')
    P(f'    Ramsey c/a z_R={zR:.6f}, h_R={hR:.6f}; rule binds iff z_R > y: {zR > y}')
    P(f'    constrained Ramsey: c/a={z0:.6f}, h0={h0:.6f}, normalized multiplier m0={m0:.6f}, V0(1)={V0:.6f}')
    P(f'{"theta":>8} {"h*":>12} {"h*-h0":>11} {"tau":>9} {"MPCform":>9} {"(fd)":>9} {"MPCtr":>9} '
      f'{"V":>11} {"V-V0":>11} {"gapC":>10} {"thJ":>11}')
    rows = []
    for th in THETAS:
        vc, vs, h = jos_value_parts(th, 1.0, y, x, rho, g)
        tau = jos_tau(th, y, x, rho, g)
        mf = 1 + jos_dh_dx(th, y, x, rho, g)
        mfd = mpc_form_fd(th, r, x, rho, g) if x > 1e-6 else np.nan
        q = (y - h) / h
        mtr = tau * q / (1 + tau * q)
        V = vc + th * vs
        rows.append((th, h, h - h0, tau, mf, mtr, V - V0, V0 - vc, th * vs))
        P(f'{th:8.0e} {h:12.6e} {h-h0:11.3e} {tau:9.6f} {mf:9.6f} {mfd:9.6f} {mtr:9.6f} '
          f'{V:11.6f} {V-V0:11.3e} {V0-vc:10.3e} {th*vs:11.3e}')
    # log-log slopes over the last three decades
    th = np.array([rw[0] for rw in rows]); dh = np.array([abs(rw[2]) for rw in rows])
    gc = np.array([rw[7] for rw in rows]); dv = np.array([abs(rw[6]) for rw in rows])
    k = slice(-4, None)
    sl_h = np.polyfit(np.log(th[k]), np.log(dh[k]), 1)[0]
    sl_gc = np.polyfit(np.log(th[k]), np.log(np.maximum(gc[k], 1e-300)), 1)[0]
    sl_v = np.polyfit(np.log(th[k]), np.log(np.maximum(dv[k], 1e-300)), 1)[0]
    P(f'    log-log slopes over theta in [1e-8,1e-5]: |h*-h0| ~ theta^{sl_h:.4f}; gapC ~ theta^{sl_gc:.4f}; '
      f'|V-V0| ~ theta^{sl_v:.4f}')
    P(f'    predicted exponent for |h*-h0|: slack 1, binding 1/gamma = {1/g:.4f}, knife edge 1/(1+gamma) = {1/(1+g):.4f}')
    P(f'    tau(1e-8) = {rows[-1][3]:.6f} against m0 = {m0:.6f}; MPCform(1e-8) = {rows[-1][4]:.6f}; '
      f'MPCtr(1e-8) = {rows[-1][5]:.6f}')
    summary[name] = dict(sl_h=sl_h, sl_gc=sl_gc, sl_v=sl_v)

# ---------------------------------------------------------------- paths: uniform on compacts, not on [0, inf)
P()
P('=== Mode of convergence of paths (case B1): a^theta_t / a^0_t = exp((h*-h0) t)')
g, rho, r, x = 2.0, 0.03, 0.05, 0.02
y = r - x
z0, h0 = cramsey_policy(y, x, rho, g)
for th in [1e-2, 1e-4, 1e-6]:
    h = jos_h(th, y, x, rho, g)
    for T in [50, 1000]:
        P(f'  theta={th:.0e}: sup_[0,{T}] |a^th/a^0 - 1| = {np.expm1((h-h0)*T):.4e}; '
          f'sup_[0,{T}] |c^th/c^0 - 1| = {max(abs((y-h)/(y-h0)*np.exp((h-h0)*t)-1) for t in (0, T)):.4e}')
    eps = 0.01
    # weighted sup_t e^{-eps t}|a^th_t - a^0_t| / a_0 = sup_t e^{-eps t} e^{(h0+x) t}(e^{(h-h0)t}-1); finite iff
    # eps > h + x; report with eps = 0.05 > h + x
    eps = 0.05
    t = np.linspace(0, 2000, 200001)
    wsup = np.max(np.exp(-eps * t) * np.exp((h0 + x) * t) * np.expm1((h - h0) * t))
    P(f'  theta={th:.0e}: sup_t e^(-0.05 t)|a^th_t - a^0_t|/a_0 (t<=2000) = {wsup:.4e}')

# ---------------------------------------------------------------- theta -> infinity
P()
P('=== Limit theta -> infinity (case S and B1): h*/y -> 1, (c/a) theta^(1/g) -> y tau_inf^(1/g), MPCform -> 0')
for name, g, rho, r, x in [CASES[0], CASES[1]]:
    y = r - x
    rhop = rho_prime(rho, g, x)
    tinf = 1 - y / (g * y + rhop)
    for th in [1e2, 1e4, 1e6, 1e8]:
        h = jos_h(th, y, x, rho, g)
        mf = 1 + jos_dh_dx(th, y, x, rho, g)
        P(f'  {name[:3]} theta={th:.0e}: h*/y={h/y:.8f}  (c/a) theta^(1/g)={(y-h)*th**(1/g):.6f} '
          f'(limit {y*tinf**(1/g):.6f})  MPCform={mf:.6f}')

# ---------------------------------------------------------------- independent check: non-stationary policies
P()
P('=== Independent check: maximize the objective over piecewise-constant consumption ratios z_k on')
P('    [10k, 10(k+1)), k = 0..39, then constant; objective by exact integration piece by piece;')
P('    L-BFGS-B from z_k = half the upper bound; objective gap <= 0 means no policy beats the closed form.')


def objective_pw(zs, y, x, rho, g, theta, tail_years=10.0):
    """Exact objective of the policy c = z_k a on piece k (length tail_years), last ratio kept forever."""
    val, loga, t0 = 0.0, 0.0, 0.0
    n = len(zs)
    for k, z in enumerate(zs):
        h = y - z
        grow = h + x
        L = np.inf if k == n - 1 else tail_years
        # int_{t0}^{t0+L} e^{-rho t} [u(z a_t) + theta w(h a_t)] dt with a_t = exp(loga + grow (t - t0))
        def piece(level):
            if level <= 0:
                return -np.inf
            if g == 1:
                # int e^{-rho t}(ln level + loga + grow(t - t0)) dt
                f = lambda T: -np.exp(-rho * T) / rho * (np.log(level) + loga + grow * (T - t0)) \
                    - grow * np.exp(-rho * T) / rho ** 2
                upper = 0.0 if L == np.inf else f(t0 + L)
                return upper - f(t0)
            kk = rho - (1 - g) * grow
            A = (level * np.exp(loga)) ** (1 - g) / (1 - g)
            # int e^{-rho t} A e^{(1-g)grow (t-t0)} dt - int e^{-rho t}/(1-g) dt
            I1 = A * np.exp(-rho * t0) * ((1 - (np.exp(-kk * L) if L != np.inf else 0.0)) / kk)
            I2 = np.exp(-rho * t0) * ((1 - (np.exp(-rho * L) if L != np.inf else 0.0)) / rho) / (1 - g)
            return I1 - I2
        val += piece(z)
        if theta > 0:
            val += theta * piece(h)
        if L != np.inf:
            loga += grow * L
            t0 += L
    return val


for name, g, rho, r, x in [CASES[0], CASES[1], CASES[2]]:
    y = r - x
    for theta in [0.0, 1e-3, 0.2]:
        if theta == 0.0:
            z_th = cramsey_policy(y, x, rho, g)[0]
            ub = y  # the rule
        else:
            z_th = y - jos_h(theta, y, x, rho, g)
            ub = y * (1 - 1e-12)
        n = 40
        f = lambda zs: -objective_pw(np.clip(zs, 1e-9, ub), y, x, rho, g, theta)
        z_start = np.full(n, 0.5 * ub)
        res = minimize(f, z_start, method='L-BFGS-B', bounds=[(1e-6, ub)] * n,
                       options=dict(maxiter=20000, ftol=1e-15, gtol=1e-12))
        zs = res.x
        P(f'  {name[:3]} theta={theta:g}: closed-form c/a={z_th:.6f}; numerical z_0={zs[0]:.6f}, '
          f'z_9={zs[9]:.6f}, max_(k<10)|z_k - closed form|={np.max(np.abs(zs[:10] - z_th)):.2e} '
          f'(z_39={zs[39]:.6f}: weight of t>=390 is negligible); '
          f'objective gap (numerical - closed form) = {-res.fun - objective_pw(np.full(n, z_th), y, x, rho, g, theta):.2e}')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print('written', OUT)
