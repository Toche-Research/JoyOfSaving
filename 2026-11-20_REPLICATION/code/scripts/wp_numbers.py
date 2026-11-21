"""Numbers, table and figure added in the working-paper version of The Joy of Saving.

Every number below is computed here with its own root-finder for the balanced path; the package and the
earlier scripts are imported only to cross-check, never as the source of a value.

Model (Proposition gains). Wealth a at market value yields income at the rate y and appreciates at the rate x,
r = y + x, active saving s = y a - c, da/dt = s + x a, U(c,s) = [c^(1-g) + th s^(1-g)]/(1-g). On the balanced
path c = (y - h) a and s = h a, where h solves

    G(h; y, rho') = 1 - y/(g h + rho') - th ((y - h)/h)^g = 0,     rho' = rho + (g - 1) x,

on (max{0, (y - rho')/g}, y). At a given return r, the share of a unit of return moved from gain to yield that
is consumed is 1 + dh/dx, with dh/dx = (G_y - (g - 1) G_rho')/G_h (analytic; checked by finite differences).

Writes
  out/macros/wp_numbers.tex           macros quoted by the working paper
  out/tables/tbl_div_calib.tex        body of the table of dividend shares (splits by curvature)
  out/tables/tbl_div_seven.tex        body of the table of the 7-per-cent calibration
  out/figures/fig_dividend_share.pdf  the share 1 + dh/dx against theta, for yields above, at and below z_R
  out/wp_numbers.txt                  everything printed, with unrounded values
Run: cd code && PYTHONPATH=. python3 scripts/wp_numbers.py
"""
import contextlib
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import io
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# joy.style selects the pgf backend and must come before pyplot
from joy.style import setup, save, BLUE, ORANGE, AQUA, INK, INK2, LABELBOX   # noqa: E402
import matplotlib.pyplot as plt                                              # noqa: E402

OUT = os.path.join(HERE, '..', 'out')
MACDIR, TBLDIR, FIGDIR = (os.path.join(OUT, d) for d in ('macros', 'tables', 'figures'))
for d in (MACDIR, TBLDIR, FIGDIR):
    os.makedirs(d, exist_ok=True)

R, RHO, GAM, TH = 0.05, 0.03, 2.0, 0.2       # the paper's calibration
SAVE_TARGET = 0.07                           # saving out of income net of gains, Fagereng et al. (2025, p. 2)

macros, log = [], io.StringIO()


def say(*a):
    print(*a)
    print(*a, file=log)


def half_up(v, nd=2):
    """Round half up, after removing floating-point noise below 1e-10: a value that is exactly a half in
    exact arithmetic (e.g. the share (1 - 0.07)/2 = 0.465 at rho = r) is printed rounded up, not down."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal('%.10f' % v).quantize(q, rounding=ROUND_HALF_UP))


def sci(v):
    """One significant digit in LaTeX, rounded half up: 7.4e-05 -> 7\\times10^{-5}."""
    e = int(np.floor(np.log10(abs(v))))
    m = Decimal('%.10f' % (v / 10 ** e)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    m = ('%s' % m).rstrip('0').rstrip('.')
    return '%s\\times10^{%d}' % (m, e)


def fmtv(fmt, value):
    if fmt == '{:.2f}':
        return half_up(value, 2)
    if fmt == 'sci':
        return sci(value)
    return fmt.format(value)


def macro(name, value, fmt, note):
    macros.append('\\newcommand{\\%s}{%s}%% %s (%.6g)' % (name, fmtv(fmt, value), note, value))
    say('%-18s %-8s %s (%.8g)' % (name, fmtv(fmt, value), note, value))


# ------------------------------------------------------------------ the balanced path, own implementation
def h_star(y, x, rho, g, th):
    """Active saving per unit of wealth under the joy of saving (Proposition gains (ii))."""
    rp = rho + (g - 1) * x
    assert rp > (1 - g) * y and rp > 0
    G = lambda h: 1 - y / (g * h + rp) - th * ((y - h) / h) ** g
    lo = max(0.0, (y - rp) / g)
    a, b = lo + 1e-13 * y, y * (1 - 1e-13)
    return brentq(G, a, b, xtol=1e-17, rtol=1e-15, maxiter=1000)


def share(y, x, rho, g, th):
    """1 + dh*/dx at a given return: the part of a unit of return moved from gain to yield that is consumed."""
    h = h_star(y, x, rho, g, th)
    rp = rho + (g - 1) * x
    D, q = g * h + rp, (y - h) / h
    Gh = g * y / D ** 2 + th * g * q ** (g - 1) * y / h ** 2
    Gy = -1 / D - th * g * q ** (g - 1) / h
    Grp = y / D ** 2
    return 1 + (Gy - (g - 1) * Grp) / Gh


def share_fd(y, x, rho, g, th, e=1e-6):
    r = y + x
    f = lambda xx: h_star(r - xx, xx, rho, g, th)
    if x - e < 0:
        return 1 + (f(x + e) - f(x)) / e
    return 1 + (f(x + e) - f(x - e)) / (2 * e)


def theta_for_saving_rate(y, x, rho, g, rate):
    """theta at which active saving out of income, h*/y, equals `rate`. On the balanced path the root of G
    gives theta in closed form: theta = (1 - y/(g h + rho')) / ((y - h)/h)^g with h = rate * y."""
    h = rate * y
    rp = rho + (g - 1) * x
    lhs = 1 - y / (g * h + rp)
    if lhs <= 0:
        return np.nan            # the Ramsey consumer already saves more than `rate`: no theta > 0 reaches it
    th = lhs / ((y - h) / h) ** g
    assert abs(h_star(y, x, rho, g, th) - h) < 1e-12
    return th


# ------------------------------------------------------------------ 0. checks against the earlier code
import jos_dividends as jd                       # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    from joy import stock, linear                # noqa: E402
for x in (0.0, 0.01, 0.02, 0.03):
    for g in (0.5, 1.0, 2.0, 3.0):
        for th in (0.01, 0.2, 1.0):
            y = R - x
            s_own = share(y, x, RHO, g, th)
            assert abs(s_own - share_fd(y, x, RHO, g, th)) < 2e-5, (x, g, th)
            if g != 1.0:
                assert abs(h_star(y, x, RHO, g, th) - jd.h_root(y, x, RHO, g, th)) < 1e-12
assert abs(share(R, 0.0, RHO, GAM, TH) - (1 + stock.dh_dx_flow(R, 0.0, RHO, GAM, TH))) < 1e-8
say('checks: own root and share agree with jos_dividends.py, joy.stock and finite differences')

# ------------------------------------------------------------------ 1. the subsidy in numbers (theta = 0.2)
sol = linear.solve(R, RHO, GAM, TH)
tau = sol['tau']
say('\n[1] subsidy at theta = 0.2: tau = %.6f, r/(1-tau) = %.6f' % (tau, R / (1 - tau)))
macro('wpSubPts', 100 * R * tau / (1 - tau), '{:.1f}', 'subsidy in points, r tau/(1-tau), theta=0.2')
macro('wpSubMkt', tau / (1 - tau), '{:.2f}', 'subsidy as a fraction of the market return, tau/(1-tau)')

# ------------------------------------------------------------------ 2. Proposition 3: shares in the calibration
say('\n[2] share 1 + dh/dx, r = 5%, rho = 3%')
s0, s2 = share(R, 0.0, RHO, GAM, TH), share(R - 0.02, 0.02, RHO, GAM, TH)
say('  gamma=2 theta=0.2: all yield %.6f, two points of gains %.6f' % (s0, s2))
# decomposition of dh/dx = -h_y + (g-1) h_rho' at the reference point and at gamma = 0.5
for g in (2.0, 0.5):
    y, x, e = R, 0.0, 1e-7
    hy = (h_star(y + e, x, RHO, g, TH) - h_star(y - e, x, RHO, g, TH)) / (2 * e)
    # h_rho' : vary rho at x = 0, where rho' = rho
    hr = (h_star(y, x, RHO + e, g, TH) - h_star(y, x, RHO - e, g, TH)) / (2 * e)
    say('  gamma=%.1f, all yield, theta=0.2: dh/dy = %.4f, dh/drho\' = %.4f, dh/dx = %.4f'
        % (g, hy, hr, -hy + (g - 1) * hr))
    if g == 2.0:
        assert abs((-hy + (g - 1) * hr) - (share(R, 0.0, RHO, g, TH) - 1)) < 1e-5
        macro('wpHyTwo', hy, '{:.2f}', 'dh*/dy at gamma = 2, all yield, theta = 0.2')
        macro('wpHrTwo', -hr, '{:.2f}', '-dh*/drho\' at gamma = 2, all yield, theta = 0.2')
rev = share(R, 0.0, RHO, 0.5, TH)
macro('wpShareRev', rev, '{:.2f}', 'share at gamma = 0.5, all yield, theta = 0.2 (negative: sign reverses)')
hy05 = (h_star(R + 1e-7, 0.0, RHO, 0.5, TH) - h_star(R - 1e-7, 0.0, RHO, 0.5, TH)) / 2e-7
macro('wpHyHalf', hy05, '{:.2f}', 'dh*/dy at gamma = 0.5, all yield, theta = 0.2')

# ------------------------------------------------------------------ 3. table: shares by split and curvature
ROWS = [  # (label, y, x, theta)
    (r'all yield', 0.05, 0.00, 0.2),
    (r'yield 3\%, gain 2\%', 0.03, 0.02, 0.2),
    (r'yield 2\%, gain 3\%', 0.02, 0.03, 0.2),
    (r'yield 2\%, gain 3\%', 0.02, 0.03, 0.01),
]
GAMS = (1.0, 2.0, 3.0)
say('\n[3] table of shares (rows: split, theta; columns: gamma = 1, 2, 3)')
body, tbl = [], {}
for i, (lab, y, x, th) in enumerate(ROWS):
    vals = [share(y, x, RHO, g, th) for g in GAMS]
    tbl[i] = vals
    zr = [(RHO - (1 - g) * R) / g for g in GAMS]
    say('  %-22s theta=%-5g ' % (lab.replace('\\', ''), th) + '  '.join('%.4f' % v for v in vals)
        + '   z_R=' + ','.join('%.4f' % z for z in zr))
    body.append('%s & $%g$ & %s\\\\' % (lab, th, ' & '.join(half_up(v) for v in vals)))
with open(os.path.join(TBLDIR, 'tbl_div_calib.tex'), 'w') as fh:
    fh.write('% generated by code/scripts/wp_numbers.py; do not edit\n' + '\n'.join(body) + '\n')
allv = [v for vals in tbl.values() for v in vals]
macro('wpTblAllYieldGOne', tbl[0][0], '{:.2f}', 'all yield, theta 0.2, gamma 1')
macro('wpTblAllYieldGThree', tbl[0][2], '{:.2f}', 'all yield, theta 0.2, gamma 3')
macro('wpTblEqLo', min(tbl[2]), '{:.2f}', 'yield 2 gain 3, theta 0.2, min over gamma')
macro('wpTblEqHi', max(tbl[2]), '{:.2f}', 'yield 2 gain 3, theta 0.2, max over gamma')
macro('wpTblEqSmallLo', min(tbl[3]), '{:.2f}', 'yield 2 gain 3, theta 0.01, min over gamma')
macro('wpTblEqSmallHi', max(tbl[3]), '{:.2f}', 'yield 2 gain 3, theta 0.01, max over gamma')

# ------------------------------------------------------------------ 4. calibration to a 7 per cent saving rate
RHOS = (0.05, 0.06, 0.07, 0.08)
SPLITS = (('all yield', 0.05, 0.00), ('yield 2%, gain 3%', 0.02, 0.03))
say('\n[4] theta set so that active saving out of income, h*/y, is 7 per cent in the split itself; r = 5%')
seven = {}
for lab, y, x in SPLITS:
    for rho in RHOS:
        for g in GAMS:
            th = theta_for_saving_rate(y, x, rho, g, SAVE_TARGET)
            sh = share(y, x, rho, g, th) if np.isfinite(th) else np.nan
            zr = (rho - (1 - g) * R) / g
            seven[(lab, rho, g)] = (th, sh)
            say('  %-18s rho=%.2f gamma=%g  theta=%.3e  share=%.4f  z_R=%.4f (yield %s z_R)'
                % (lab, rho, g, th, sh, zr, '<' if y < zr else '>='))
# also: the same theta (calibrated with the whole return as yield) applied to the equity-like split
say('  cross-check: theta calibrated with all yield, applied to yield 2%, gain 3%:')
cross = []
for rho in RHOS:
    for g in GAMS:
        th = seven[('all yield', rho, g)][0]
        cross.append(share(0.02, 0.03, rho, g, th))
        say('    rho=%.2f gamma=%g theta=%.3e share=%.4f' % (rho, g, th, cross[-1]))
ths_all = [seven[('all yield', rho, g)][0] for rho in RHOS for g in GAMS]
ths_eq = [seven[('yield 2%, gain 3%', rho, g)][0] for rho in RHOS for g in GAMS]
sh_all = [seven[('all yield', rho, g)][1] for rho in RHOS for g in GAMS]
sh_eq = [seven[('yield 2%, gain 3%', rho, g)][1] for rho in RHOS for g in GAMS]
assert all(np.isfinite(ths_all)) and all(np.isfinite(ths_eq))
# smallest rho at which 7 per cent is reachable with all yield, gamma = 2: g_R < 0.07 r
macro('wpSevenRhoMin', 100 * (R - GAM * SAVE_TARGET * R), '{:.1f}',
      'rho below which a Ramsey consumer with gamma=2 and all yield already saves more than 7% of income')
macro('wpSevenThetaAllLo', min(ths_all), 'sci', 'theta range, all yield, low')
macro('wpSevenThetaAllHi', max(ths_all), '{:.2f}', 'theta range, all yield, high')
macro('wpSevenThetaEqLo', min(ths_eq), 'sci', 'theta range, equity split, low')
macro('wpSevenThetaEqHi', max(ths_eq), '{:.2f}', 'theta range, equity split, high')
macro('wpSevenShareAllLo', min(sh_all), '{:.2f}', 'share range, all yield, low')
macro('wpSevenShareAllHi', max(sh_all), '{:.2f}', 'share range, all yield, high')
macro('wpSevenShareEqLo', min(sh_eq), '{:.2f}', 'share range, equity split, low')
macro('wpSevenShareEqHi', max(sh_eq), '{:.2f}', 'share range, equity split, high')
macro('wpSevenCrossLo', min(cross), '{:.2f}', 'all-yield theta applied to equity split, share low')
macro('wpSevenCrossHi', max(cross), '{:.2f}', 'all-yield theta applied to equity split, share high')
# the theta = 0.2 benchmark: its saving rate out of income with all yield (Table 1: 44 per cent)
macro('wpSavRateBench', 100 * linear.growth(R, RHO, GAM, TH) / R, '{:.0f}',
      'saving out of income at theta = 0.2, all yield, r 5 rho 3 gamma 2')
# table body: rows rho, columns gamma; cells theta and the two shares
body = []
for rho in RHOS:
    cells = []
    for g in GAMS:
        th_a, sh_a = seven[('all yield', rho, g)]
        th_e, sh_e = seven[('yield 2%, gain 3%', rho, g)]
        cells.append('%s & %s' % (half_up(sh_a), half_up(sh_e)))
    body.append('$%g$ & %s\\\\' % (100 * rho, ' & '.join(cells)))
with open(os.path.join(TBLDIR, 'tbl_div_seven.tex'), 'w') as fh:
    fh.write('% generated by code/scripts/wp_numbers.py; do not edit\n' + '\n'.join(body) + '\n')

# ------------------------------------------------------------------ 5. the interest response (Corollary)
say('\n[5] interest response: sign of dg/dr - 1/gamma against Phi(g)')


def phi(g, r, rho, gm):
    return (gm ** 2 - 1) * g ** 2 - ((gm - 1) * r - gm * rho) * g - r * rho


def dgdr(r, rho, gm, th):
    """Growth (own root, x = 0) and its response to r from the formula (dgdr) of the paper, checked against a
    central difference of the own root."""
    g = h_star(r, 0.0, rho, gm, th)
    q = (r - g) / g
    Fg = gm * r / (gm * g + rho) ** 2 + th * gm * q ** (gm - 1) * r / g ** 2
    Fr = -1 / (gm * g + rho) - th * gm * q ** (gm - 1) / g
    d = -Fr / Fg
    e = 1e-7 * r
    fd = (h_star(r + e, 0.0, rho, gm, th) - h_star(r - e, 0.0, rho, gm, th)) / (2 * e)
    assert abs(d - fd) < 1e-4, (r, rho, gm, th, d, fd)
    return d, g


nviol, ncase = 0, 0
for gm in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
    for r in np.arange(0.02, 0.105, 0.01):
        for rho in (0.005, 0.01, 0.02, 0.03, 0.04, 0.06):
            if rho <= (1 - gm) * r + 1e-3:
                continue
            for th in (1e-3, 0.01, 0.1, 1.0, 10.0):
                d, g = dgdr(r, rho, gm, th)
                ph = phi(g, r, rho, gm)
                ncase += 1
                if abs(ph) > 1e-12 and np.sign(d - 1 / gm) != np.sign(ph):
                    nviol += 1
say('  sign of dg/dr - 1/gamma equals sign of Phi(g*) in %d of %d cases' % (ncase - nviol, ncase))
assert nviol == 0
gR = (R - RHO) / GAM
macro('wpCondLhs', (R - gR) * gR * 1e4, '{:.0f}', '(r - g_R) g_R times 1e4, calibration')
macro('wpCondRhs', R * RHO * 1e4, '{:.0f}', 'r rho times 1e4, calibration')
gdag = brentq(lambda g: phi(g, R, RHO, GAM), 1e-9, R)
thdag = linear.tau_lhs(gdag, R, RHO, GAM) / ((R - gdag) / gdag) ** GAM
say('  calibration: g_dag = %.6f, theta_dag = %.6f' % (gdag, thdag))
assert abs(round(thdag, 2) - 0.15) < 1e-9          # \linDgDrCrossTheta of make_tables.py
macro('wpThetaDag', thdag, '{:.2f}', 'theta at which the response crosses 1/gamma, calibration')
# counterexample: r = 10%, rho = 1%, gamma = 2
r2, rho2 = 0.10, 0.01
gR2 = (r2 - rho2) / GAM
macro('wpCxLhs', (r2 - gR2) * gR2 * 1e4, '{:.2f}', '(r - g_R) g_R x 1e4 at r=10%, rho=1%, gamma=2')
macro('wpCxRhs', r2 * rho2 * 1e4, '{:.0f}', 'r rho x 1e4 at r=10%, rho=1%')
macro('wpCxDgDrSmall', dgdr(r2, rho2, GAM, 0.01)[0], '{:.3f}', 'dg/dr at theta = 0.01, counterexample')
macro('wpCxDgDrB', dgdr(r2, rho2, GAM, 0.2)[0], '{:.3f}', 'dg/dr at theta = 0.2, counterexample')
# how often the dip is absent on a grid with gamma > 1 and r > rho
# in exact rational arithmetic: the grid has cases with (r - g_R) g_R = r rho exactly (gamma = 1.5, r = 4 rho), in
# which Corollary (iii) applies and the dip is absent; floating point misclassifies some of them
absent, tot, ties = 0, 0, 0
for gm in (Fraction(3, 2), Fraction(2), Fraction(3), Fraction(5)):
    for r in (Fraction(k, 100) for k in range(2, 11)):
        for rho in (Fraction(1, 200), Fraction(1, 100), Fraction(2, 100), Fraction(3, 100), Fraction(4, 100)):
            if r <= rho:
                continue
            g_r = (r - rho) / gm
            tot += 1
            absent += (r - g_r) * g_r >= r * rho
            ties += (r - g_r) * g_r == r * rho
say('  grid: %d cases with (r - g_R) g_R = r rho exactly (counted as absent, Corollary (iii))' % ties)
macro('wpDipAbsent', absent, '{:d}', 'grid cases with gamma > 1, r > rho in which the dip is absent')
macro('wpDipCases', tot, '{:d}', 'grid cases with gamma > 1 and r > rho')
say('  grid: gamma in {1.5,2,3,5}, r in {2,...,10}%%, rho in {0.5,1,2,3,4}%%, r > rho')

# ------------------------------------------------------------------ 5b. BNW (2007): first differences times four
# Table 3, p. 247: coefficients on the change in dividends for total expenditure, 0.057 and 0.093; BNW (p. 246)
# expect them to be about a quarter of the levels coefficients, so four times them is the comparable figure
macro('wpBnwFourLo', 4 * 0.057, '{:.2f}', 'four times the BNW first-difference coefficient 0.057')
macro('wpBnwFourHi', 4 * 0.093, '{:.2f}', 'four times the BNW first-difference coefficient 0.093')

# ------------------------------------------------------------------ 6. the equivalent gain of Table 2 at gamma = 2
from joy import neoclassical     # noqa: E402
import make_tables as mt         # noqa: E402
e0 = neoclassical.Economy(theta=0.0)
K0 = mt.K0_FRAC * e0.k_mgr
mu = neoclassical.Economy(theta=0.5).consumption_equivalent(K0, T=mt.CEQ_T, n=mt.CEQ_N)
ramsey_tr, ramsey_ss = e0.transition(K0, T=mt.CEQ_T, n=mt.CEQ_N), e0.steady_state()
e5 = neoclassical.Economy(theta=0.5)
v_r = e5.path_value(ramsey_tr, ramsey_ss['c'], ramsey_ss['s'])
sup = e5.path_value(ramsey_tr, ramsey_ss['c'], ramsey_ss['s'], scale_c=1e14)
v_j = e5.path_value(e5.transition(K0, T=mt.CEQ_T, n=mt.CEQ_N), e5.steady_state()['c'], e5.steady_state()['s'])
F = (v_j - v_r) / (sup - v_r)
say('\n[6] theta = 0.5: equivalent gain mu = %.6f, F = %.6f, F/(1-F) = %.6f' % (mu, F, F / (1 - F)))
assert abs(mu - F / (1 - F)) < 1e-6

# ------------------------------------------------------------------ 7. figure: the share against theta
setup()
thetas = np.logspace(-4, 1, 400)
fig, ax = plt.subplots(figsize=(5.6, 3.1))
zR = (RHO - (1 - GAM) * R) / GAM
curves = ((0.05, BLUE, '-', r'yield $5\%>z_R$'), (zR, AQUA, (0, (1, 1.5)), r'yield $4\%=z_R$'),
          (0.03, ORANGE, (0, (5, 2)), r'yield $3\%<z_R$'))
for y, col, ls, lab in curves:
    x = R - y
    v = np.array([share(y, x, RHO, GAM, th) for th in thetas])
    ax.plot(thetas, v, color=col, ls=ls, lw=2.0 if col == AQUA else 1.6, label=lab)
    say('  figure: yield %.3f, share at theta=1e-4: %.4f, at 0.2: %.4f, at 10: %.4f'
        % (y, v[0], share(y, x, RHO, GAM, 0.2), v[-1]))
for lvl in (1.0, GAM / (1 + GAM), 0.0):    # the limits of Proposition rule (iii)
    ax.axhline(lvl, color=INK2, lw=0.6, ls=':')
ax.axvline(TH, color=INK2, lw=0.6, ls=':')
ax.text(TH * 1.08, 0.04, r'$\theta=0.2$', fontsize=8, color=INK, bbox=LABELBOX)
ax.set_xscale('log')
ax.set_ylim(-0.03, 1.05)
ax.set_xlabel(r'strength of the joy of saving, $\theta$')
ax.set_ylabel(r'share consumed, $1+\partial h^*/\partial x$')
ax.legend(loc='upper right', fontsize=8, frameon=True, framealpha=1.0, facecolor='white', edgecolor='none')
fig.tight_layout()
save(fig, 'fig_dividend_share', FIGDIR)

# ------------------------------------------------------------------ write
with open(os.path.join(MACDIR, 'wp_numbers.tex'), 'w') as fh:
    fh.write('% generated by code/scripts/wp_numbers.py; do not edit\n' + '\n'.join(macros) + '\n')
with open(os.path.join(OUT, 'wp_numbers.txt'), 'w') as fh:
    fh.write(log.getvalue())
print('wrote out/macros/wp_numbers.tex, out/tables/tbl_div_calib.tex, out/tables/tbl_div_seven.tex,'
      ' out/figures/fig_dividend_share.pdf, out/wp_numbers.txt')
