"""Macros for the paragraphs on dividends and on predictable income.

Dividends: at a given total return r = y + x, the share of a unit of return moved from capital gain to
yield that the joy-of-saving consumer spends is 1 + dh*/dx (Proposition gains), at x = 0 and at x = 2 points.
Predictable income: the share of a small temporary addition to income spent at a fixed costate,
tau q/(1 + tau q) with q = c/s, its elasticity (times Y/c), and its maximum over all theta > 0 at gamma = 2,
taken over the growth rate and checked against a theta grid; also checks that at a fixed costate the share
of a payment that is spent falls with the size of the payment.

The values are computed by the functions of scripts/jos_dividends.py and scripts/jos_predictable.py, which
also print the checks (closed-form root against a direct Bellman maximization, analytic derivative against
a finite difference, fixed-costate formula against a full dynamic solution). This script runs both, cross-
checks the dividend shares against the package function used by flow_vs_stock.py, and writes the macros.

Writes:
  out/dividends.txt          the printed output of jos_dividends.py and jos_predictable.py, then the macros
  out/macros/dividends.tex   the macros the paper quotes
Run: cd code && PYTHONPATH=. python3 scripts/dividend_macros.py"""
import contextlib
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jos_dividends as jd          # noqa: E402
import jos_predictable as jp        # noqa: E402
from joy import stock               # noqa: E402

OUTDIR = os.path.join(HERE, '..', 'out')
MACDIR = os.path.join(OUTDIR, 'macros')
os.makedirs(MACDIR, exist_ok=True)

R, RHO, GAMMA = 0.05, 0.03, 2.0
THETA = 0.2          # the reference weight of the paragraph on capital gains
X = 0.02             # capital gains, points of the return R
assert (jd.r, jd.rho, jd.G) == (R, RHO, GAMMA) and (jp.r, jp.rho, jp.G) == (R, RHO, GAMMA)

report = io.StringIO()
with contextlib.redirect_stdout(report):
    jd.main()
    jp.main()

macros, lines = [], []


def macro(name, value, fmt, note):
    macros.append('\\newcommand{\\%s}{%s}%% %s' % (name, fmt.format(value), note))
    lines.append('%-14s %s   (%.6f; %s)' % (name, fmt.format(value), value, note))


# dividends at a given return
div0 = 1 + jd.dhdx_analytic(R, 0.0, RHO, GAMMA, THETA)
div2 = 1 + jd.dhdx_analytic(R - X, X, RHO, GAMMA, THETA)
# cross-check against the package function behind \evMargDispZero and \evMargDispGain
assert abs(div0 - (1 + stock.dh_dx_flow(R, 0.0, RHO, GAMMA, THETA))) < 1e-8
assert abs(div2 - (1 + stock.dh_dx_flow(R - X, X, RHO, GAMMA, THETA))) < 1e-8
assert abs(div2 - (1 + jd.dhdx_fd(R, X, RHO, GAMMA, THETA))) < 1e-5
macro('evDivZero', div0, '{:.2f}', '1 + dh*/dx at x = 0, theta = 0.2')
macro('evDivGain', div2, '{:.2f}', '1 + dh*/dx at x = 0.02, theta = 0.2')

# predictable income, fixed costate
m, tau, q = jd.transitory(R, 0.0, RHO, GAMMA, THETA)
h = jd.h_root(R, 0.0, RHO, GAMMA, THETA)
macro('evPredMPC', m, '{:.2f}', 'tau q/(1 + tau q), theta = 0.2')
macro('evPredElast', m * R / (R - h), '{:.2f}', 'tau q/(1 + tau q) times Y/c, theta = 0.2')
ths = np.logspace(-4, 4, 4001)                    # the grid of jos_dividends.py
mmax_grid = max(jd.transitory(R, 0.0, RHO, GAMMA, th)[0] for th in ths)
# The same maximum taken over the growth rate: theta -> g* is one to one from (0, inf) onto
# (max{0, g_R}, R), and on the balanced path tau = 1 - R/(gamma g + rho) and q = (R - g)/g, so the
# share is a function of g alone and is maximized over that interval with no grid in theta.
from scipy.optimize import minimize_scalar        # noqa: E402
g_lo = max(0.0, (R - RHO) / GAMMA)


def share_of_g(g):
    tau_g = 1 - R / (GAMMA * g + RHO)
    q_g = (R - g) / g
    return tau_g * q_g / (1 + tau_g * q_g)


opt = minimize_scalar(lambda g: -share_of_g(g), bounds=(g_lo + 1e-12, R - 1e-12), method='bounded',
                      options={'xatol': 1e-14})
mmax = share_of_g(opt.x)
theta_at_max = (1 - R / (GAMMA * opt.x + RHO)) / ((R - opt.x) / opt.x) ** GAMMA
assert mmax >= mmax_grid - 1e-9 and mmax - mmax_grid < 1e-4, (mmax, mmax_grid)
macro('evPredMPCmax', mmax, '{:.2f}', 'max over the growth rate (all theta > 0) of tau q/(1 + tau q), gamma = 2'
      ' (theta at the max %.4f; theta-grid max %.6f)' % (theta_at_max, mmax_grid))
# At a fixed costate the share of a payment P that is spent falls as P rises (paper, paragraph
# "Predictable income"): check it on a grid of payments at theta = 0.2 and at the maximizing theta.
for th_chk in (THETA, theta_at_max):
    g_chk = jp.g_root(th_chk)
    c0, s0 = R - g_chk, g_chk
    lam_chk = c0 ** -GAMMA - th_chk * s0 ** -GAMMA
    shares = [(jp.c_of(lam_chk, c0 + s0 + P, th_chk) - c0) / P for P in np.linspace(1e-4, 2.0, 400) * c0]
    assert all(b < a for a, b in zip(shares, shares[1:])), 'share not decreasing in the payment'
    assert shares[0] < mmax + 1e-6
# the full dynamic solution agrees with the formula to the second decimal for a small change
jump, _, _ = jp.run(THETA, 0.01 * R)
assert abs(jump - m) < 0.005, (jump, m)

with open(os.path.join(MACDIR, 'dividends.tex'), 'w') as fh:
    fh.write('% generated by code/scripts/dividend_macros.py; do not edit\n')
    fh.write('\n'.join(macros) + '\n')
with open(os.path.join(OUTDIR, 'dividends.txt'), 'w') as fh:
    fh.write(report.getvalue())
    fh.write('\n== macros ==\n' + '\n'.join(lines) + '\n')
print('\n'.join(lines))
