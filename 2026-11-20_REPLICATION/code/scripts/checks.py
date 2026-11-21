"""Numerical accuracy checks behind the notes to Tables 1 and 2, written to code/out/checks.txt.

Five checks, in the order they are written to the file:
  1. the closed form for the linear technology against a direct maximisation of the Bellman equation,
     at every weight theta in Table 1;
  2. the closed-form determinant of the Jacobian against finite differences at step sizes 1e-4 to 1e-7,
     which is the agreement to eight decimal places reported in the notes to Table 2;
  3. the impact gross saving rate and the half-life against the offset of the backward integration from
     the steady state and against the tolerance of the integrator, which is the invariance to four decimal
     places reported in the notes to Table 2;
  4. the residual of the steady-state condition Psi(k*) = 0, the position of k* relative to the modified
     golden rule, and the sign of the costate;
  5. the costate (Euler) equation along the saddle path used for Figure 3 and the last column of Table 2.

The file records the checks; the paper quotes them in words rather than through the numbers.tex macros, so
nothing in the paper changes if the file is regenerated.
Run: cd code && PYTHONPATH=. python3 scripts/checks.py"""
import os
import numpy as np
from scipy.integrate import solve_ivp
from joy import linear, neoclassical

OUT = os.path.join(os.path.dirname(__file__), '..', 'out', 'checks.txt')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
lines = []


def log(s):
    """Print a line and keep it for out/checks.txt."""
    print(s); lines.append(s)


# 1. Linear closed form against the direct Bellman maximization, all table rows
worst = 0.0
for th in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0):
    g = linear.growth(0.05, 0.03, 2.0, th)
    z, _ = linear.hjb_direct(0.05, 0.03, 2.0, th)
    worst = max(worst, abs(0.05 - g - z))
log(f'linear: max |z(formula) - z(direct Bellman maximization)| over the table rows = {worst:.2e}')

# 2. Jacobian step size and closed-form determinant
for th in (0.02, 0.1, 0.5):
    e = neoclassical.Economy(theta=th)
    dets = [np.linalg.det(e.jacobian(h=h)) for h in (1e-4, 1e-5, 1e-6, 1e-7)]
    log(f'neoclassical theta={th}: det J by finite differences, h=1e-4..1e-7: ' + ', '.join(f'{d:.8f}' for d in dets)
        + f'; closed form {e.det_closed_form():.8f}')

# 3. Transition: impact saving rate and half-life against the branch offset and ODE tolerances
e0 = neoclassical.Economy(theta=0.0)
for th in (0.05, 0.2):
    e = neoclassical.Economy(theta=th)
    ss = e.steady_state(); ks, cs = ss['k'], ss['c']
    v = e.stable_vector(); v = v / np.linalg.norm(v)
    res = []
    for off, rtol in ((1e-7, 1e-11), (1e-6, 1e-11), (1e-8, 1e-11), (1e-7, 1e-8)):
        y0 = np.array([ks, cs]) - off * ks * v
        stop = lambda t, y: y[0] - 0.5 * e0.k_mgr
        stop.terminal = True
        sol = solve_ivp(lambda t, y: -np.array(e.rhs(t, y)), [0, 1e5], y0, events=stop, rtol=rtol, atol=rtol * 1e-2, max_step=2.0)
        c0 = sol.y[1][-1]; k0 = sol.y[0][-1]
        res.append(100 * (e.f(k0) - c0) / e.f(k0))
    log(f'neoclassical theta={th}: impact saving rate (%) from k0 = 0.5 k_MGR with offset/rtol (1e-7,1e-11), (1e-6,1e-11), '
        f'(1e-8,1e-11), (1e-7,1e-8): ' + ', '.join(f'{x:.4f}' for x in res))

# 4. Steady-state root: residual and bracket
for th in (0.02, 0.1, 0.5):
    e = neoclassical.Economy(theta=th); ss = e.steady_state()
    log(f'neoclassical theta={th}: |Psi(k*)| = {abs(e.Psi(ss["k"])):.2e}, k*/k_MGR = {ss["k"]/e.k_mgr:.6f}, '
        f'lambda* = {ss["lam"]:.6f} > 0')

# 5. Euler equation along the saddle path (costate residual), theta = 0.2
e = neoclassical.Economy(theta=0.2)
tr = e.transition(0.5 * e0.k_mgr, T=200.0, n=4001)
lam = tr['c'] ** (-e.gamma) - e.theta * tr['s'] ** (-e.gamma)
dt = tr['t'][1] - tr['t'][0]
lamdot = np.gradient(lam, dt)
formula = (e.rho + e.delta) * lam - e.fp(tr['k']) * tr['c'] ** (-e.gamma)
i = slice(2, 3000)
log(f'neoclassical theta=0.2: max |d lambda/dt (differences) - formula| / max |formula| along the path = '
    f'{np.max(np.abs(lamdot[i]-formula[i]))/np.max(np.abs(formula[i])):.2e}; min lambda = {lam.min():.4f}')

with open(OUT, 'w') as f:
    f.write('\n'.join(lines) + '\n')
