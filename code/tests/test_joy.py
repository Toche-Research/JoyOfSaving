"""Tests of the two models, against closed forms, against the first-order conditions and against finite
differences. There is no pytest dependency: the file runs every function whose name begins with test_ and
prints a line for each, so a silent run is a passing run.

Run: cd code && PYTHONPATH=. python3 tests/test_joy.py
"""
import numpy as np
from joy import linear, neoclassical

R, RHO, GAMMA = 0.05, 0.03, 2.0


def test_closed_form_against_bellman():
    """The growth rate from the one-line formula equals the z maximizing the Bellman equation directly."""
    for theta in (0.05, 0.2, 1.0):
        for gamma in (2.0, 0.5, 4.0):
            g = linear.growth(R, RHO, gamma, theta)
            z_direct, A = linear.hjb_direct(R, RHO, gamma, theta)
            assert abs((R - g) - z_direct) < 1e-7, (theta, gamma, g, z_direct)
            # envelope: u'(z) = (1 - gamma) A
            s = R - z_direct
            assert abs((z_direct ** (-gamma) - theta * s ** (-gamma)) - (1 - gamma) * A) < 1e-6 * abs(A)
    print('closed form = direct Bellman maximization: ok')


def test_tau_is_Us_over_Uc_and_lambda_positive():
    """On the balanced growth path tau equals U_s/U_c, lies strictly between zero and one, and the growth
    rate lies strictly between the Ramsey rate and the interest rate."""
    for theta in (0.01, 0.1, 0.5, 5.0):
        d = linear.solve(R, RHO, GAMMA, theta)
        c, s = d['z'], d['g']
        assert abs(theta * s ** (-GAMMA) / c ** (-GAMMA) - d['tau']) < 1e-12
        assert 0 < d['tau'] < 1
        assert d['g'] > linear.ramsey_growth(R, RHO, GAMMA) and d['g'] < R
    print('tau = U_s/U_c in (0,1), g in (g_R, r): ok')


def test_euler_on_bgp():
    """d ln lambda/dt = rho - r/(1 - tau) with lambda ~ e^{-gamma g t} on the balanced growth path."""
    d = linear.solve(R, RHO, GAMMA, 0.3)
    assert abs(-GAMMA * d['g'] - (RHO - R / (1 - d['tau']))) < 1e-12
    print('Euler equation on the BGP: ok')


def test_impatient_consumer_saves():
    """A consumer more impatient than the market (r = 3%, rho = 5%) still accumulates, which a Ramsey
    consumer with the same parameters would not: the Ramsey growth rate is negative."""
    g = linear.growth(0.03, 0.05, GAMMA, 0.2)
    assert g > 0 and linear.ramsey_growth(0.03, 0.05, GAMMA) < 0
    print('impatient consumer accumulates: ok')


def test_comparative_statics_by_finite_differences():
    """The three comparative statics of the growth rate agree with finite differences, and growth rises
    with both the interest rate and the strength of the joy of saving at this calibration."""
    th, h = 0.2, 1e-6
    d = linear.solve(R, RHO, GAMMA, th)
    fd_r = (linear.growth(R + h, RHO, GAMMA, th) - linear.growth(R - h, RHO, GAMMA, th)) / (2 * h)
    fd_th = (linear.growth(R, RHO, GAMMA, th + h) - linear.growth(R, RHO, GAMMA, th - h)) / (2 * h)
    fd_rho = (linear.growth(R, RHO + h, GAMMA, th) - linear.growth(R, RHO - h, GAMMA, th)) / (2 * h)
    assert abs(fd_r - d['dg_dr']) < 1e-6 and abs(fd_th - d['dg_dtheta']) < 1e-6
    assert abs(fd_rho - linear.dg_drho(d['g'], R, RHO, GAMMA, th)) < 1e-6
    assert d['dg_dtheta'] > 0 and d['dg_dr'] > 0
    print('dg/dr, dg/dtheta, dg/drho against finite differences: ok')


def test_log_case():
    """The logarithmic case gamma = 1 satisfies the growth equation in its gamma = 1 form."""
    g = linear.growth(R, RHO, 1.0, 0.2)
    assert abs((g + RHO - R) / (g + RHO) - 0.2 * (R - g) / g) < 1e-12
    print('log case: ok')


def test_steady_state_and_golden_rule():
    """The steady state satisfies its first-order condition, is the only root on the bracketing interval,
    lies beyond the modified golden rule, and passes the golden rule exactly when the closed-form threshold
    says it does; the Ramsey case returns the modified golden rule itself."""
    for theta in (0.02, 0.1, 0.5):
        e = neoclassical.Economy(theta=theta)
        ss = e.steady_state()
        assert abs(e.fp(ss['k']) - (e.rho + e.delta) * (1 - ss['tau'])) < 1e-12
        assert ss['k'] > e.k_mgr and ss['lam'] > 0
        assert (ss['k'] > e.k_gr) == e.beyond_golden_rule()
        # uniqueness: Psi strictly decreasing on (k_mgr, k_max)
        ks = np.linspace(e.k_mgr, e.k_max * 0.999, 2000)
        assert np.all(np.diff(e.Psi(ks)) < 0)
    e0 = neoclassical.Economy(theta=0.0)
    assert abs(e0.steady_state()['k'] - e0.k_mgr) < 1e-14
    print('steady state: f\'(k*) = (rho+delta)(1-tau*), unique, golden-rule threshold: ok')


def test_determinant_closed_form():
    """The closed-form determinant used for the half-life agrees with a finite-difference Jacobian, the
    trace equals rho, and the steady state is a saddle at every weight tested."""
    for theta in (0.0, 0.05, 0.2, 1.0):
        e = neoclassical.Economy(theta=theta)
        J = e.jacobian()
        assert abs(np.linalg.det(J) - e.det_closed_form()) < 1e-7 * abs(e.det_closed_form()), (theta, np.linalg.det(J), e.det_closed_form())
        assert abs(np.trace(J) - e.rho) < 1e-7
        assert e.det_closed_form() < 0
        ev = np.sort(np.linalg.eigvals(J).real)
        assert abs(ev[0] - e.eigen()[0]) < 1e-7
    print('det J closed form = finite-difference Jacobian, trace = rho, saddle: ok')


def test_saddle_path_residuals():
    """The computed transition converges to the steady state, satisfies the costate equation along the way,
    and keeps the costate positive, so the path is the optimum and not merely a solution of the system."""
    e = neoclassical.Economy(theta=0.2)
    tr = e.transition(0.5 * e.steady_state()['k'], T=600.0)
    ss = e.steady_state()
    assert abs(tr['k'][-1] / ss['k'] - 1) < 1e-6 and abs(tr['c'][-1] / ss['c'] - 1) < 1e-6
    # the costate equation holds along the path: compare lambda-dot by differences with its formula
    lam = tr['c'] ** (-e.gamma) - e.theta * tr['s'] ** (-e.gamma)
    dt = tr['t'][1] - tr['t'][0]
    lamdot = np.gradient(lam, dt)
    formula = (e.rho + e.delta) * lam - e.fp(tr['k']) * tr['c'] ** (-e.gamma)
    i = slice(5, 400)
    assert np.max(np.abs(lamdot[i] - formula[i])) < 1e-3 * np.max(np.abs(formula[i]))
    assert np.all(lam > 0)
    print('transition converges, costate equation residual small, lambda > 0 along the path: ok')


def test_cdot_zero_locus():
    """The dc/dt = 0 locus is a closed curve: one root for k < k_MGR, two or none for k > k_MGR; the steady state
    lies on its lower branch, and the Ramsey case returns NaN (vertical line)."""
    e = neoclassical.Economy(theta=0.2); ss = e.steady_state()
    lo, hi = e.cdot_zero_locus(np.array([0.5 * e.k_mgr, ss['k'], 3.0 * ss['k']]))
    assert lo[0] == hi[0] and 0 < lo[0] < e.f(0.5 * e.k_mgr)
    assert abs(lo[1] - ss['c']) < 1e-8 * ss['c'] and hi[1] > lo[1]
    assert np.isnan(lo[2]) and np.isnan(hi[2])
    for c in (lo[1], hi[1]):
        assert abs(e.cdot(ss['k'], c)) < 1e-9
    assert np.all(np.isnan(neoclassical.Economy(theta=0.0).cdot_zero_locus(np.array([ss['k']]))[0]))
    print('dc/dt = 0 locus: closed curve, steady state on the lower branch: ok')


def test_ramsey_benchmark():
    """With theta = 0 the code reproduces the textbook Ramsey model: consumption growth along the
    transition equals (f'(k) - delta - rho)/gamma."""
    e = neoclassical.Economy(theta=0.0)
    tr = e.transition(0.5 * e.k_mgr, T=600.0)
    # Ramsey Euler equation c-dot/c = (f' - delta - rho)/gamma
    assert np.max(np.abs(tr['gc'] - (e.fp(tr['k']) - e.delta - e.rho) / e.gamma)) < 1e-12
    print('Ramsey benchmark: ok')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
    print('all tests passed')
