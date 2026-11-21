"""Financial risk in the joy of saving model (draft section for the working paper).

Model. Wealth a > 0 yields income at the safe rate y > 0 and earns a risky capital gain,

    da = (y a - c) dt + a (x dt + sigma dZ),        s = y a - c  (active saving),

with Z a standard Brownian motion and the utility (iso) of the paper,
    U(c, s) = (c^(1-gamma) - 1)/(1-gamma) + theta (s^(1-gamma) - 1)/(1-gamma).
Only active saving gives joy; a capital gain does not.

Result checked here (Proposition R of risk_section.tex). With xhat = x - gamma sigma^2/2 (the certainty
equivalent of the gain) and rho' = rho + (gamma - 1) xhat, and under rho > (1-gamma)(y + xhat), the optimal plan
is c = (y - h*) a, s = h* a, where h* is the unique root in (max{0, h_R}, y) of

    (gamma h + rho' - y)/(gamma h + rho') = theta ((y - h)/h)^gamma,          h_R = (y - rho')/gamma,

h_R being the active saving rate of the consumer with theta = 0 (the fixed-portfolio Merton rule). Then

    dh*/d(sigma^2) = (gamma - 1)/2 * M,   M = h(y-h) / (h(y-h) + tau D^2),  D = gamma h + rho',  tau = 1 - y/D,

with M in (0,1), against (gamma - 1)/2 for theta = 0.

Every number in risk_section.tex comes from this script. Outputs, all in the folder of this script:
    risk_output.txt   the log of every check and every tabulated number
    risk_macros.tex   LaTeX macros for the numbers quoted in the text
    tbl_risk.tex      body of the table
    tbl_portfolio.tex body of the portfolio table
    fig_risk.pdf      the figure
Run: python3 risk.py    (about ten seconds; needs numpy, scipy, matplotlib and a LaTeX installation)
No pytest: the checks are plain assertions, run every time.
"""
import os
import time

import figstyle  # must precede pyplot (pgf backend)
import numpy as np
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GAMMA, RHO, THETA, R = 2.0, 0.03, 0.2, 0.05
SIGMAS_TABLE = (0.0, 0.05, 0.10, 0.15, 0.20)
lines = []
macros = []


def num(v, d):
    """A number for a table cell, rounded half away from zero, with a math minus sign."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(repr(round(float(v), 10))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    t = '%s' % q
    return '$-%s$' % t[1:] if t.startswith('-') and q != 0 else t.lstrip('-')


def log(s=''):
    print(s)
    lines.append(s)


def macro(name, value, comment):
    assert name.isalpha(), name
    macros.append('\\newcommand{\\%s}{%s}%% %s' % (name, value, comment))


# ---------------------------------------------------------------------------------------------------------------
# The joy of saving
# ---------------------------------------------------------------------------------------------------------------
def rho_prime(rho, gamma, x, sigma):
    """Effective rate of impatience in units of wealth net of the gain: rho + (gamma-1)(x - gamma sigma^2/2)."""
    return rho + (gamma - 1.0) * (x - 0.5 * gamma * sigma ** 2)


def condition(rho, gamma, y, x, sigma):
    """Finiteness condition rho > (1-gamma)(y + x - gamma sigma^2/2), equivalently rho' > (1-gamma) y."""
    return rho > (1.0 - gamma) * (y + x - 0.5 * gamma * sigma ** 2)


def h_merton(y, x, sigma, rho, gamma):
    """Active saving per unit of wealth with theta = 0: y - c/a, c/a = (rho - (1-gamma)(y + xhat))/gamma."""
    return (y - rho_prime(rho, gamma, x, sigma)) / gamma


def F(h, y, rp, gamma, theta):
    """Root function: required tau less delivered tau."""
    return 1.0 - y / (gamma * h + rp) - theta * ((y - h) / h) ** gamma


def h_star(y, x, sigma, rho, gamma, theta):
    """Unique root of F in (max{0, h_R}, y); theta = 0 gives h_R."""
    assert condition(rho, gamma, y, x, sigma), (y, x, sigma, rho, gamma)
    rp = rho_prime(rho, gamma, x, sigma)
    hR = (y - rp) / gamma
    if theta == 0:
        return hR
    lo = max(0.0, hR)
    eps = 1e-13 * y
    return brentq(F, lo + eps, y - eps, args=(y, rp, gamma, theta), xtol=1e-16, rtol=1e-14, maxiter=1000)


def damping(h, y, rp, gamma):
    """M = h(y-h)/(h(y-h) + tau D^2): the response of h* to risk relative to the theta = 0 response."""
    D = gamma * h + rp
    tau = 1.0 - y / D
    return h * (y - h) / (h * (y - h) + tau * D ** 2)


def solve(y, x, sigma, rho=RHO, gamma=GAMMA, theta=THETA):
    rp = rho_prime(rho, gamma, x, sigma)
    h = h_star(y, x, sigma, rho, gamma, theta)
    D = gamma * h + rp
    tau = 1.0 - y / D
    M = damping(h, y, rp, gamma) if theta > 0 else 1.0
    return dict(h=h, z=y - h, rp=rp, tau=tau, D=D, M=M, dh=(gamma - 1) / 2 * M, hR=(y - rp) / gamma)


def J_constant_ratio(z, y, x, sigma, rho, gamma, theta):
    """Objective (homogeneous part, a0 = 1) of the plan c = z a, computed from E[a_t^(1-gamma)] for geometric
    Brownian motion, independently of the root equation: E a_t^(1-g) = exp((1-g)(y - z + x - g sigma^2/2) t)."""
    u = (z ** (1 - gamma) + theta * (y - z) ** (1 - gamma)) / (1 - gamma)
    den = rho - (1 - gamma) * (y - z + x - 0.5 * gamma * sigma ** 2)
    return u / den if den > 0 else (-np.inf if gamma > 1 else np.inf)


def hjb_residual(y, x, sigma, rho, gamma, theta, avals=(0.1, 1.0, 10.0)):
    """Residual of rho V = max_c {U(c, ya-c) + V'(ya - c + xa) + V'' sigma^2 a^2/2} at V = K a^(1-g)/(1-g) + C0,
    K = z^-g - theta h^-g, C0 = -(1+theta)/((1-g) rho), maximizing over c numerically (not by the FOC)."""
    S = solve(y, x, sigma, rho, gamma, theta)
    h, z = S['h'], S['z']
    K = z ** (-gamma) - theta * h ** (-gamma)
    C0 = -(1 + theta) / ((1 - gamma) * rho)
    V = lambda a: K * a ** (1 - gamma) / (1 - gamma) + C0
    Vp = lambda a: K * a ** (-gamma)
    Vpp = lambda a: -gamma * K * a ** (-gamma - 1)
    U = lambda c, s: (c ** (1 - gamma) - 1) / (1 - gamma) + theta * (s ** (1 - gamma) - 1) / (1 - gamma)
    worst_res, worst_z = 0.0, 0.0
    for a in avals:
        obj = lambda c: -(U(c, y * a - c) + Vp(a) * (y * a - c + x * a))
        res = minimize_scalar(obj, bounds=(1e-10 * y * a, y * a * (1 - 1e-10)), method='bounded',
                              options={'xatol': 1e-14 * a})
        rhs = -res.fun + 0.5 * sigma ** 2 * a ** 2 * Vpp(a)
        worst_res = max(worst_res, abs(rho * V(a) - rhs) / abs(rho * V(a)))
        worst_z = max(worst_z, abs(res.x / a - z))
    return worst_res, worst_z, K


# ---------------------------------------------------------------------------------------------------------------
# The spirit of capitalism counterpart: U(c, a) = (c^(1-g)-1)/(1-g) + theta_S (a^(1-g)-1)/(1-g), same wealth process.
# HJB with V = K a^(1-g)/(1-g): c = z a with gamma z + theta_S z^gamma = m, m = rho - (1-gamma)(y + x - gamma sigma^2/2).
# ---------------------------------------------------------------------------------------------------------------
def m_soc(y, x, sigma, rho, gamma):
    return rho - (1 - gamma) * (y + x - 0.5 * gamma * sigma ** 2)


def z_soc(y, x, sigma, rho, gamma, thetaS):
    m = m_soc(y, x, sigma, rho, gamma)
    assert m > 0
    return brentq(lambda z: gamma * z + thetaS * z ** gamma - m, 1e-14, m / gamma, xtol=1e-16, rtol=1e-14)


def theta_soc_matching(z, y, x, sigma, rho, gamma):
    """theta_S at which the stock consumer has consumption-wealth ratio z at (y, x, sigma)."""
    return (m_soc(y, x, sigma, rho, gamma) - gamma * z) / z ** gamma


def damping_soc(z, thetaS, gamma):
    return 1.0 / (1.0 + thetaS * z ** (gamma - 1))


def hjb_residual_soc(y, x, sigma, rho, gamma, thetaS, avals=(0.1, 1.0, 10.0)):
    z = z_soc(y, x, sigma, rho, gamma, thetaS)
    K = z ** (-gamma)
    C0 = -(1 + thetaS) / ((1 - gamma) * rho)
    V = lambda a: K * a ** (1 - gamma) / (1 - gamma) + C0
    Vp = lambda a: K * a ** (-gamma)
    Vpp = lambda a: -gamma * K * a ** (-gamma - 1)
    worst_res, worst_z = 0.0, 0.0
    for a in avals:
        obj = lambda c: -((c ** (1 - gamma) - 1) / (1 - gamma) + Vp(a) * ((y + x) * a - c))
        res = minimize_scalar(obj, bounds=(1e-10 * a, a), method='bounded', options={'xatol': 1e-14 * a})
        rhs = -res.fun + thetaS * (a ** (1 - gamma) - 1) / (1 - gamma) + 0.5 * sigma ** 2 * a ** 2 * Vpp(a)
        worst_res = max(worst_res, abs(rho * V(a) - rhs) / abs(rho * V(a)))
        worst_z = max(worst_z, abs(res.x / a - z))
    return worst_res, worst_z


# ---------------------------------------------------------------------------------------------------------------
# Portfolio: a safe asset with income yield yf, and a risky asset with income yield yr, expected gain xr, volatility
# sr. Share pi in the risky asset, held constant. Then y(pi) = yf + pi (yr - yf), x(pi) = pi xr, sigma(pi) = |pi| sr.
# ---------------------------------------------------------------------------------------------------------------
def value_K(pi, yf, yr, xr, sr, rho, gamma, theta):
    """K(pi) = U_c - U_s (in units of a^-gamma) for the constant share pi; the value is K a^(1-g)/(1-g) + const."""
    y = yf + pi * (yr - yf)
    if y <= 0 or not condition(rho, gamma, y, pi * xr, abs(pi) * sr):
        return np.nan
    S = solve(y, pi * xr, abs(pi) * sr, rho, gamma, theta)
    if theta == 0:
        z = S['z']
        return z ** (-gamma)
    return S['z'] ** (-gamma) - theta * S['h'] ** (-gamma)


def best_pi(yf, yr, xr, sr, rho, gamma, theta, lo=-1.0, hi=3.0):
    """Best constant share: maximizes K/(1-gamma), the value per unit of a^(1-gamma)."""
    sgn = 1.0 / (1 - gamma)
    obj = lambda p: -sgn * value_K(p, yf, yr, xr, sr, rho, gamma, theta)
    grid = np.linspace(lo, hi, 801)
    vals = np.array([obj(p) for p in grid])
    vals = np.where(np.isnan(vals), np.inf, vals)
    i = int(np.argmin(vals))
    # unimodality on the grid (reported, not assumed)
    finite = vals[np.isfinite(vals)]
    d = np.diff(finite)
    turns = int(np.sum(np.diff(np.sign(d[d != 0])) != 0))
    res = minimize_scalar(obj, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]), method='bounded',
                          options={'xatol': 1e-12})
    return res.x, turns


def main():
    t0 = time.time()
    y0, x0 = R, 0.0  # baseline: the whole expected return is income, the gain has mean zero
    log('Financial risk and the joy of saving. gamma=%g, rho=%g, theta=%g, r=%g.' % (GAMMA, RHO, THETA, R))
    log('Wealth process da = (y a - c) dt + a (x dt + sigma dZ); active saving s = y a - c; h = s/a.')
    log()

    # ---------------- 1. identities ----------------
    log('1. Identities on computed solutions')
    # (a) sigma = 0 reproduces Proposition 3 (numbers in macros/evidence.tex: evHzero 2.20, evHgain 1.11)
    h00 = solve(0.05, 0.0, 0.0)['h']
    h32 = solve(0.03, 0.02, 0.0)['h']
    log('  (a) sigma=0: h* = %.6f%% (all yield; paper evHzero = 2.20), %.6f%% (y=3%%, x=2%%; paper evHgain = 1.11)'
        % (100 * h00, 100 * h32))
    assert abs(100 * h00 - 2.20) < 0.005 and abs(100 * h32 - 1.11) < 0.005
    # direct check against equation (closed) of Proposition 2 with r=5%, rho=3%
    g2 = brentq(lambda g: (GAMMA * g + RHO - R) / (GAMMA * g + RHO) - THETA * ((R - g) / g) ** GAMMA, 1e-9, R - 1e-12,
                xtol=1e-16)
    log('      equation (closed) of Proposition 2 solved directly: g* = %.10f, h*(sigma=0, x=0) = %.10f, gap %.1e'
        % (g2, h00, abs(g2 - h00)))
    assert abs(g2 - h00) < 1e-12
    # (b) theta -> 0 gives h* -> max{0, h_R}: the Merton rule only when h_R > 0
    worst = 0.0
    for s in SIGMAS_TABLE:
        hm = h_merton(y0, x0, s, RHO, GAMMA)
        ht = solve(y0, x0, s, theta=1e-12)['h']
        worst = max(worst, abs(ht - max(hm, 0.0)))
        zM = (RHO - (1 - GAMMA) * (y0 + x0 - 0.5 * GAMMA * s ** 2)) / GAMMA
        assert abs((y0 - zM) - hm) < 1e-15
    log('  (b) theta=1e-12, y=5%%, x=0 (h_R > 0 at every sigma): max |h* - h_Merton| over sigma in %s = %.2e'
        % (list(SIGMAS_TABLE), worst))
    assert worst < 1e-5
    hs_neg = [(h_merton(0.03, 0.02, s, RHO, GAMMA), solve(0.03, 0.02, s, theta=1e-12)['h']) for s in (0.0, 0.05, 0.10)]
    log('      y=3%%, x=2%%, sigma=0, 0.05, 0.1 (h_R < 0): h_R = %s%%, h*(theta=1e-12) = %s%%: h* -> 0, not h_R'
        % ([round(100 * a, 3) for a, b in hs_neg], ['%.1e' % (100 * b) for a, b in hs_neg]))
    assert all(a < 0 and b < 1e-4 for a, b in hs_neg)
    # (c) risk = a lower certainty-equivalent gain: h*(y, x, sigma) = h*(y, x - gamma sigma^2/2, 0)
    worst = 0.0
    for s in SIGMAS_TABLE:
        for (y, x) in ((0.05, 0.0), (0.03, 0.02), (0.02, 0.04)):
            worst = max(worst, abs(solve(y, x, s)['h'] - solve(y, x - 0.5 * GAMMA * s ** 2, 0.0)['h']))
    log('  (c) h*(y,x,sigma) - h*(y, x - gamma sigma^2/2, 0): max abs %.1e' % worst)
    assert worst < 1e-14
    # (d) gamma = 1: risk has no effect on h*
    hs = [solve(y0, x0, s, gamma=1.0)['h'] for s in SIGMAS_TABLE]
    log('  (d) gamma=1: h* over sigma = %s (%%), spread %.1e' % (['%.6f' % (100 * v) for v in hs], max(hs) - min(hs)))
    assert max(hs) - min(hs) < 1e-15
    # (e) damping at sigma = 0, x = 0 equals gamma times the paper's -dh*/drho' (wpHrTwo = 0.128811)
    S0 = solve(y0, x0, 0.0)
    log('  (e) M at sigma=0, all yield = %.6f; gamma * 0.128811 (wp_numbers wpHrTwo) = %.6f' % (S0['M'], 2 * 0.128811))
    assert abs(S0['M'] - 2 * 0.128811) < 5e-6
    log()

    # ---------------- 2. independent checks of optimality and of the derivative ----------------
    log('2. Independent checks (y=5%, x=0)')
    for s in SIGMAS_TABLE:
        S = solve(y0, x0, s)
        res, zgap, K = hjb_residual(y0, x0, s, RHO, GAMMA, THETA)
        # direct maximization of the objective over constant consumption-wealth ratios
        o = minimize_scalar(lambda z: -J_constant_ratio(z, y0, x0, s, RHO, GAMMA, THETA),
                            bounds=(1e-9, y0 - 1e-9), method='bounded', options={'xatol': 1e-14})
        Jstar = J_constant_ratio(S['z'], y0, x0, s, RHO, GAMMA, THETA)
        # derivative by central differences in sigma^2
        e = 1e-6
        s2 = s ** 2
        hp = solve(y0, x0, np.sqrt(s2 + e))['h']
        hm = solve(y0, x0, np.sqrt(max(s2 - e, 0.0)))['h'] if s2 >= e else None
        fd = (hp - hm) / (2 * e) if hm is not None else (hp - S['h']) / e
        log('  sigma=%.2f: HJB rel. residual %.1e, |argmax_c/a - z*| %.1e; argmax of objective over c/a %.10f vs z* %.10f;'
            ' value K/(1-g) %.8f vs objective %.8f; dh/dsigma^2 analytic %.6f, finite difference %.6f'
            % (s, res, zgap, o.x, S['z'], K / (1 - GAMMA), Jstar, S['dh'], fd))
        assert res < 1e-13 and zgap < 1e-7 and abs(o.x - S['z']) < 1e-7
        assert abs(K / (1 - GAMMA) - Jstar) < 1e-9 * abs(Jstar)
        assert abs(fd - S['dh']) < 2e-4 * max(1.0, abs(S['dh']))
    log()

    # ---------------- 3. table: joy of saving, Merton, spirit of capitalism matched at sigma = 0 ----------------
    log('3. Response to risk, gamma=2, rho=3%, theta=0.2 (percentages of wealth)')
    tbl = []
    blocks = (('the whole expected return is income', 0.05, 0.0),
              ('part of the expected return is a gain', 0.03, 0.02))
    store = {}
    for title, y, x in blocks:
        z0 = solve(y, x, 0.0)['z']
        thS = theta_soc_matching(z0, y, x, 0.0, RHO, GAMMA)
        log('  block y=%g, x=%g: spirit of capitalism weight matched at sigma=0: theta_S = %.6f' % (y, x, thS))
        tbl.append('\\SetCell[c=8]{l}\\emph{%s: $y=%g\\%%$, $x=%g\\%%$, $\\theta_S=%.2f$} &&&&&&&\\\\'
                   % (title, 100 * y, 100 * x, thS))
        for s in SIGMAS_TABLE:
            S = solve(y, x, s)
            hM = h_merton(y, x, s, RHO, GAMMA)
            zS = z_soc(y, x, s, RHO, GAMMA, thS)
            MS = damping_soc(zS, thS, GAMMA)
            resS, zgS = hjb_residual_soc(y, x, s, RHO, GAMMA, thS)
            assert resS < 1e-13 and zgS < 1e-7
            # finite-difference check of the stock consumer's response
            e = 1e-6
            fdS = -(z_soc(y, x, np.sqrt(s ** 2 + e), RHO, GAMMA, thS) - zS) / e
            assert abs(fdS - (GAMMA - 1) / 2 * MS) < 2e-4
            store[(y, x, s)] = (S, hM, zS, MS, thS)
            log('    sigma=%.2f rho\'=%.3f%%  Merton h=%.3f%%  JoS h*=%.3f%% c/a=%.3f%% s/(ya)=%.1f%% tau=%.3f M=%.4f'
                ' dh/dsig2=%.4f | SoC h=%.3f%% M_S=%.4f (HJB res %.1e)'
                % (s, 100 * S['rp'], 100 * hM, 100 * S['h'], 100 * S['z'], 100 * S['h'] / y, S['tau'], S['M'],
                   S['dh'], 100 * (y - zS), MS, resS))
            tbl.append(' & '.join(num(v, d) for v, d in ((s, 2), (100 * S['rp'], 2), (100 * hM, 2), (100 * S['h'], 2),
                                                           (100 * S['h'] / y, 0), (S['M'], 3), (100 * (y - zS), 2),
                                                           (MS, 3))) + '\\\\')
        tbl.append('\\midrule')
    tbl.pop()
    with open(os.path.join(HERE, 'tbl_risk.tex'), 'w') as f:
        f.write('% generated by risk.py; do not edit\n' + '\n'.join(tbl) + '\n')
    log()

    # matched comparison at every sigma (the proposition: M < M_S at the same c/a, ratio of excess terms D/h)
    log('4. Matched comparison at each point: theta_S chosen so the stock consumer has the same c/a at that sigma')
    for s in SIGMAS_TABLE:
        S = solve(y0, x0, s)
        thS = theta_soc_matching(S['z'], y0, x0, s, RHO, GAMMA)
        MS = damping_soc(S['z'], thS, GAMMA)
        RS = 1 / MS - 1
        RJ = 1 / S['M'] - 1
        log('  sigma=%.2f: M_JoS=%.4f, M_SoC=%.4f, (1/M_JoS-1)/(1/M_SoC-1)=%.6f, D/h=%.6f'
            % (s, S['M'], MS, RJ / RS, S['D'] / S['h']))
        assert abs(RJ / RS - S['D'] / S['h']) < 1e-9 * S['D'] / S['h']
    log()

    # numbers for the text
    a0, a2 = store[(0.05, 0.0, 0.0)], store[(0.05, 0.0, 0.2)]
    b0, b2 = store[(0.03, 0.02, 0.0)], store[(0.03, 0.02, 0.2)]
    dJ = a2[0]['h'] - a0[0]['h']
    dM = a2[1] - a0[1]
    dS = (0.05 - a2[2]) - (0.05 - a0[2])
    log('  y=5%%, x=0: from sigma=0 to 0.2, active saving rises by %.3f pp (JoS), %.3f pp (Merton), %.3f pp (SoC matched)'
        % (100 * dJ, 100 * dM, 100 * dS))
    log('              ratio JoS/Merton %.3f, SoC/Merton %.3f' % (dJ / dM, dS / dM))
    dJb = b2[0]['h'] - b0[0]['h']
    dSb = (0.03 - b2[2]) - (0.03 - b0[2])
    log('  y=3%%, x=2%%: rises by %.3f pp (JoS), %.3f pp (Merton), %.3f pp (SoC matched)'
        % (100 * dJb, 100 * (b2[1] - b0[1]), 100 * dSb))
    macro('rkHzero', '%.2f' % (100 * a0[0]['h']), 'h* %, sigma=0, y=5, x=0')
    macro('rkHtwo', '%.2f' % (100 * a2[0]['h']), 'h* %, sigma=0.2, y=5, x=0')
    macro('rkHMzero', '%.2f' % (100 * a0[1]), 'Merton h %, sigma=0')
    macro('rkHMtwo', '%.2f' % (100 * a2[1]), 'Merton h %, sigma=0.2')
    macro('rkHSzero', '%.2f' % (100 * (0.05 - a0[2])), 'SoC h %, sigma=0 (matched)')
    macro('rkHStwo', '%.2f' % (100 * (0.05 - a2[2])), 'SoC h %, sigma=0.2 (matched)')
    macro('rkDJ', '%.2f' % (100 * dJ), 'rise in h*, pp, sigma 0 to 0.2, JoS')
    macro('rkDM', '%.2f' % (100 * dM), 'rise in h, pp, Merton')
    macro('rkDS', '%.2f' % (100 * dS), 'rise in h, pp, SoC matched')
    macro('rkRatioJ', '%.2f' % (dJ / dM), 'JoS rise / Merton rise')
    macro('rkRatioS', '%.2f' % (dS / dM), 'SoC rise / Merton rise')
    macro('rkMzero', '%.3f' % a0[0]['M'], 'M at sigma=0, y=5')
    macro('rkMtwo', '%.3f' % a2[0]['M'], 'M at sigma=0.2, y=5')
    macro('rkMSzero', '%.3f' % a0[3], 'M_S at sigma=0, y=5 (matched)')
    macro('rkThetaS', '%.2f' % a0[4], 'theta_S matched, y=5')
    macro('rkMgainzero', '%.3f' % b0[0]['M'], 'M at sigma=0, y=3, x=2')
    macro('rkMSgainzero', '%.3f' % b0[3], 'M_S at sigma=0, y=3, x=2 (matched)')
    macro('rkDJgain', '%.2f' % (100 * dJb), 'rise in h*, pp, sigma 0 to 0.2, y=3, x=2')
    macro('rkDSgain', '%.2f' % (100 * dSb), 'rise in h, pp, SoC, y=3, x=2')
    macro('rkDMgain', '%.2f' % (100 * (b2[1] - b0[1])), 'rise in h, pp, Merton, y=3, x=2')
    macro('rkRhoPtwo', '%.2f' % (100 * a2[0]['rp']), "rho' %, sigma=0.2, y=5, x=0")
    macro('rkDoverH', '%.2f' % (a0[0]['D'] / a0[0]['h']), 'D/h at sigma=0, y=5')
    macro('rkOneMinusRatio', '%.2f' % ((1 - a0[0]['M']) / (1 - a0[3])), '(1-M)/(1-M_S) at sigma=0, y=5 (matched)')
    macro('rkRemovedJoS', '%.0f' % (100 * (1 - a0[0]['M'])), 'share of benchmark response removed by JoS, per cent, sigma=0, y=5')
    macro('rkRemovedSoC', '%.0f' % (100 * (1 - a0[3])), 'share of benchmark response removed by SoC, per cent, sigma=0, y=5 (matched)')
    log('  sigma=0, y=5%%: (1-M)/(1-M_S) = %.4f; (1/M-1)/(1/M_S-1) = D/h = %.4f'
        % ((1 - a0[0]['M']) / (1 - a0[3]), a0[0]['D'] / a0[0]['h']))
    # M against sigma in the calibration (numerical finding, no monotonicity in sigma is proved)
    Mgrid = [solve(y0, x0, s)['M'] for s in np.linspace(0, 0.2, 201)]
    log('  M rises with sigma on a grid of 201 points in [0, 0.2] (y=5%%, x=0): %s' % bool(np.all(np.diff(Mgrid) > 0)))

    # gamma = 0.5: the sign reverses, the damping persists
    log('5. gamma = 0.5 (rho=3%, theta=0.2, y=5%, x=0)')
    for s in (0.0, 0.2):
        S = solve(y0, x0, s, gamma=0.5)
        log('  sigma=%.2f: Merton h=%.3f%%, JoS h*=%.3f%%, M=%.4f, dh/dsig2=%.4f (Merton %.4f)'
            % (s, 100 * h_merton(y0, x0, s, RHO, 0.5), 100 * S['h'], S['M'], S['dh'], -0.25))
    Sh0, Sh2 = solve(y0, x0, 0.0, gamma=0.5), solve(y0, x0, 0.2, gamma=0.5)
    assert Sh2['h'] < Sh0['h']
    macro('rkHalfDJ', '%.2f' % (100 * (Sh0['h'] - Sh2['h'])), 'fall in h*, pp, gamma=0.5')
    macro('rkHalfDM', '%.2f' % (100 * (h_merton(y0, x0, 0.0, RHO, 0.5) - h_merton(y0, x0, 0.2, RHO, 0.5))),
          'fall in h Merton, pp, gamma=0.5')
    log()

    # ---------------- 6. grid check of the proposition ----------------
    log('6. Grid check of Proposition R(ii)-(iii)')
    n = nsign = 0
    worst_fd = 0.0
    mono_cases = mono_fail = 0
    for gamma in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0):
        for rho in (0.01, 0.03, 0.05):
            for y in (0.02, 0.05):
                for x in (0.0, 0.02, 0.04):
                    for s in (0.0, 0.1, 0.2, 0.3):
                        # keep a margin from the boundary rho = (1-gamma)(y + xhat), where h_R -> y
                        if not condition(rho - 1e-4, gamma, y, x, s):
                            continue
                        Ms = []
                        for theta in (0.01, 0.1, 0.2, 1.0, 10.0):
                            S = solve(y, x, s, rho, gamma, theta)
                            n += 1
                            assert 0 < S['M'] < 1 and 0 < S['h'] < y and 0 < S['tau'] < 1
                            assert S['h'] > S['hR'] - 1e-15
                            e = 1e-7
                            hp = solve(y, x, np.sqrt(s * s + e), rho, gamma, theta)['h']
                            fd = (hp - S['h']) / e
                            worst_fd = max(worst_fd, abs(fd - S['dh']) / max(1e-3, abs(S['dh'])))
                            if gamma != 1:
                                nsign += np.sign(fd) == np.sign(gamma - 1)
                            else:
                                nsign += abs(fd) < 1e-6
                            # matched SoC damping is weaker
                            thS = theta_soc_matching(S['z'], y, x, s, rho, gamma)
                            assert thS > 0 and S['M'] < damping_soc(S['z'], thS, gamma)
                            Ms.append(S['M'])
                        if S['rp'] <= y and S['hR'] > 0:
                            mono_cases += 1
                            mono_fail += not all(np.diff(Ms) < 0)
    log('  %d solutions; sign of the finite-difference response equals sign(gamma-1) (zero at gamma=1) in %d;'
        ' worst relative gap analytic vs finite difference %.1e' % (n, nsign, worst_fd))
    log('  M in (0,1), h in (max(0,h_R), y), tau in (0,1), and M < M_SoC (matched) in every case')
    log('  M falls with theta (theta = 0.01, 0.1, 0.2, 1, 10) when h_R > 0: %d of %d parameter points fail'
        % (mono_fail, mono_cases))
    assert nsign == n and mono_fail == 0
    # a case with h_R < 0: M is not monotone in theta
    vals = [solve(0.02, 0.0, 0.0, 0.05, 2.0, th)['M'] for th in (1e-4, 0.01, 0.1, 1.0, 10.0, 1e3)]
    log('  h_R < 0 (y=2%%, rho=5%%, gamma=2, sigma=0): M over theta = 1e-4..1e3: %s' % ['%.4f' % v for v in vals])
    # the edge case h_R = 0 (rho' = y): M falls with theta, limit 1/(1+gamma) as theta -> 0
    th0 = (1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0)
    vals0 = [solve(0.05, 0.0, 0.0, 0.05, 2.0, th)['M'] for th in th0]
    assert abs(solve(0.05, 0.0, 0.0, 0.05, 2.0, 0.1)['hR']) < 1e-15
    log('  h_R = 0 (y=rho=5%%, x=0, gamma=2, sigma=0): M over theta = %s: %s; limit 1/(1+gamma) = %.4f'
        % (list(th0), ['%.4f' % v for v in vals0], 1 / 3))
    assert all(np.diff(vals0) < 0) and abs(vals0[0] - 1 / 3) < 2e-3
    log()

    # ---------------- 7. portfolio ----------------
    log('7. Portfolio: safe asset yield yf=2%; risky asset expected return 6% = income yr + expected gain xr; sr=0.2')
    yf, ER, sr = 0.02, 0.06, 0.20
    ptbl = []
    for yr in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
        xr = ER - yr
        Delta = yr - yf
        pM, tM = best_pi(yf, yr, xr, sr, RHO, GAMMA, 0.0)
        pJ, tJ = best_pi(yf, yr, xr, sr, RHO, GAMMA, THETA)
        merton = (Delta + xr) / (GAMMA * sr ** 2)
        y = yf + pJ * Delta
        S = solve(y, pJ * xr, pJ * sr)
        formula = (xr + Delta / (1 - S['tau'])) / (GAMMA * sr ** 2)
        # spirit of capitalism: the HJB term in pi is K(pi xr + pi Delta) - gamma K pi^2 sr^2/2, so pi = Merton
        log('  yr=%.0f%%, xr=%.0f%%: Merton pi=%.6f (best constant pi with theta=0: %.6f, turns %d);'
            ' JoS best pi=%.6f (turns %d), formula (x+Delta/(1-tau))/(gamma s^2)=%.6f, tau=%.4f, h*=%.3f%%, c/a=%.3f%%'
            % (100 * yr, 100 * xr, merton, pM, tM, pJ, tJ, formula, S['tau'], 100 * S['h'], 100 * S['z']))
        assert abs(pM - merton) < 1e-6 and abs(pJ - formula) < 1e-6 and tJ <= 1 and tM <= 1
        ptbl.append('%.0f & %.0f & %.2f & %.3f & %.3f & %.2f & %.2f\\\\'
                    % (100 * yr, 100 * xr, merton, pJ, S['tau'], 100 * S['h'], 100 * S['z']))
        if yr == 0.0:
            macro('rkPiLow', '%.2f' % pJ, 'JoS risky share, yr=0')
        if yr == 0.02:
            macro('rkPiMid', '%.2f' % pJ, 'JoS risky share, yr=yf=2')
        if yr == 0.04:
            macro('rkPiHigh', '%.2f' % pJ, 'JoS risky share, yr=4')
        if yr == 0.06:
            macro('rkPiTop', '%.2f' % pJ, 'JoS risky share, yr=6')
        # full HJB at K* = K(pi*): Phi(K*, pi) <= 0 for every pi on a grid, with the maximum at pi*
        Kstar = value_K(pJ, yf, yr, xr, sr, RHO, GAMMA, THETA)
        def Phi(p):
            yy = yf + p * Delta
            if yy <= 0:
                return -np.inf
            u = lambda z: (z ** (1 - GAMMA) + THETA * (yy - z) ** (1 - GAMMA)) / (1 - GAMMA)
            o = minimize_scalar(lambda z: -(u(z) - Kstar * z), bounds=(1e-12, yy * (1 - 1e-12)), method='bounded',
                                options={'xatol': 1e-15})
            return -o.fun + Kstar * (yy + p * xr - 0.5 * GAMMA * p ** 2 * sr ** 2) - RHO * Kstar / (1 - GAMMA)
        pg = np.linspace(-0.4, 2.0, 2401)
        ph = np.array([Phi(p) for p in pg])
        scale = abs(RHO * Kstar / (1 - GAMMA))
        log('      full HJB at K(pi*): max over pi in [-0.4, 2] of Phi(K*, pi)/|rho K*/(1-gamma)| = %.1e at pi = %.4f;'
            ' Phi(K*, pi*) = %.1e' % (ph.max() / scale, pg[int(np.argmax(ph))], Phi(pJ) / scale))
        assert ph.max() / scale < 1e-9 and abs(pg[int(np.argmax(ph))] - pJ) < 2e-3
    # finer scan of the best share against the income yield of the risky asset (numerical)
    scan = [(yr, best_pi(yf, yr, ER - yr, sr, RHO, GAMMA, THETA)[0]) for yr in np.arange(0.0, 0.0601, 0.005)]
    ipk = int(np.argmax([p for _, p in scan]))
    log('  best share against yr (step 0.5 points): %s' % ', '.join('%.1f%%: %.4f' % (100 * a, b) for a, b in scan))
    log('  the best share peaks at yr = %.1f%% (pi* = %.4f) on this grid, and falls after' % (100 * scan[ipk][0],
                                                                                              scan[ipk][1]))
    assert 0 < ipk < len(scan) - 1
    macro('rkPiPeak', '%.2f' % scan[ipk][1], 'peak JoS share on the yr grid')
    macro('rkPiPeakAt', '%.1f' % (100 * scan[ipk][0]), 'yr at the peak, %')
    with open(os.path.join(HERE, 'tbl_portfolio.tex'), 'w') as f:
        f.write('% generated by risk.py; do not edit\n' + '\n'.join(ptbl) + '\n')
    log()

    # ---------------- 8. figure ----------------
    figstyle.setup()
    sig = np.linspace(0.0, 0.2, 81)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axes[0]
    z0 = solve(y0, x0, 0.0)['z']
    thS = theta_soc_matching(z0, y0, x0, 0.0, RHO, GAMMA)
    hM = [100 * h_merton(y0, x0, s, RHO, GAMMA) for s in sig]
    hJ = [100 * solve(y0, x0, s)['h'] for s in sig]
    hJ1 = [100 * solve(y0, x0, s, theta=1.0)['h'] for s in sig]
    hS = [100 * (y0 - z_soc(y0, x0, s, RHO, GAMMA, thS)) for s in sig]
    ax.plot(sig, hM, color=figstyle.BENCH, ls='--', label=r'$\theta=0$')
    ax.plot(sig, hS, color=figstyle.ORANGE, ls='-.', label=r'stock, $\theta_S$ matched')
    ax.plot(sig, hJ, color=figstyle.BLUE, label=r'joy, $\theta=0.2$')
    ax.plot(sig, hJ1, color=figstyle.BLUE, ls=':', label=r'joy, $\theta=1$')
    ax.set_xlabel(r'volatility $\sigma$')
    ax.set_ylabel(r'active saving $s/a$, \%')
    ax.set_title('(a)')
    ax.set_xlim(0, 0.2)
    ax = axes[1]
    MJ = [solve(y0, x0, s)['M'] for s in sig]
    MJ1 = [solve(y0, x0, s, theta=1.0)['M'] for s in sig]
    MS = [damping_soc(z_soc(y0, x0, s, RHO, GAMMA, thS), thS, GAMMA) for s in sig]
    ax.axhline(1.0, color=figstyle.BENCH, ls='--', lw=1.0)
    ax.plot(sig, MS, color=figstyle.ORANGE, ls='-.')
    ax.plot(sig, MJ, color=figstyle.BLUE)
    ax.plot(sig, MJ1, color=figstyle.BLUE, ls=':')
    ax.set_xlabel(r'volatility $\sigma$')
    ax.set_ylabel(r'response relative to $\theta=0$')
    ax.set_title('(b)')
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    figstyle.save(fig, 'fig_risk', HERE)
    log('8. Figure fig_risk.pdf written (sigma from 0 to 0.2, 81 points).')
    log('   M at sigma=0.2: JoS theta=0.2 %.4f, theta=1 %.4f, SoC matched %.4f' % (MJ[-1], MJ1[-1], MS[-1]))
    log('elapsed %.1f s' % (time.time() - t0))

    with open(os.path.join(HERE, 'risk_output.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(HERE, 'risk_macros.tex'), 'w') as f:
        f.write('% generated by risk.py; do not edit\n' + '\n'.join(macros) + '\n')


if __name__ == '__main__':
    main()
