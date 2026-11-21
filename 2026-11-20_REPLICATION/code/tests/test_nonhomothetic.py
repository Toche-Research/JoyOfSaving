"""Tests of the non-homothetic steady-state analysis, against the paper's own solver, finite differences, a dense
grid, Kurz's printed roots and the trace and determinant conditions of the canonical system.  Same convention as
test_joy.py: no pytest, every test_ function runs and prints a line.

Run: cd code && PYTHONPATH=. python3 tests/test_nonhomothetic.py
"""
import numpy as np
from joy.nonhomothetic import (Technology, Separable, min_gc_for_multiplicity, min_cbar_for_multiplicity,
                               min_rho_for_multiplicity, flow_jacobian)
from joy.neoclassical import Economy

T = Technology()


def test_slope_against_finite_differences():
    """The analytic d ln theta*/d ln k agrees with a central difference, flow and stock, with and without cbar."""
    for model in ('flow', 'stock'):
        for gc, gs, cb in ((2.0, 2.0, 0.0), (5.0, 0.3, 0.0), (2.0, 2.0, 1.2), (30.0, 0.0, 0.0)):
            S = Separable(T, gc=gc, gs=gs, cbar=cb, model=model)
            lo, hi = S.k_range()
            for k in np.exp(np.linspace(np.log(lo), np.log(hi), 9)[1:-1]):
                h = 1e-6
                fd = (np.log(S.theta_star(k * np.exp(h))) - np.log(S.theta_star(k * np.exp(-h)))) / (2 * h)
                assert abs(fd - S.slope(k)) < 1e-5 * max(1.0, abs(fd)), (model, gc, gs, cb, k, fd, S.slope(k))
    print('slope of theta* = finite difference: ok')


def test_gamma_min_against_grid_and_invariance():
    """Gamma_min from the first-order condition in z equals the minimum on a dense grid in k; it is unchanged when
    A changes and when rho and delta are scaled together."""
    for r, d, a in ((0.03, 0.05, 1 / 3), (0.06, 0.03, 0.45), (0.3706, 0.01, 0.45)):
        t = Technology(r, d, a)
        G, kG, _ = t.Gamma_min()
        k = np.exp(np.linspace(np.log(t.k_mgr), np.log(t.k_gr), 400_001)[1:-1])
        assert abs(np.min(t.Gamma(k)) - G) < 1e-6 * G
        assert abs(t.Gamma(kG) - G) < 1e-9 * G
        assert abs(Technology(r, d, a, A=7.0).Gamma_min()[0] - G) < 1e-9 * G
        assert abs(Technology(3 * r, 3 * d, a).Gamma_min()[0] - G) < 1e-9 * G
    print('Gamma_min: grid, scale and (rho, delta) invariance: ok')


def test_paper_case_unique_and_equal_to_paper_solver():
    """With the paper's utility, theta* is strictly increasing and its root is the paper's steady state; the
    Jacobian of the separable code equals the paper's, and its determinant equals the closed form eq:det."""
    S = Separable(T, gc=2.0, gs=2.0)
    assert S.max_slope_deficit() < 0
    for th in (0.02, 0.2, 2.0):
        e = Economy(theta=th)
        ss = e.steady_state()
        ks = S.steady_states(th)
        assert len(ks) == 1 and abs(ks[0] - ss['k']) < 1e-9 * ss['k']
        J = flow_jacobian(S, th, ks[0])
        assert np.allclose(J, e.jacobian(), rtol=1e-5, atol=1e-9)
        assert abs(np.linalg.det(J) - e.det_closed_form()) < 1e-6 * abs(e.det_closed_form())
    print('paper case: unique, same k*, same Jacobian, det = eq:det: ok')


def test_threshold_curvature_counts_roots():
    """Just above the curvature threshold there are three steady states for some theta, just below there is one
    for every theta; the roots do not move when the grid is refined eightfold."""
    g0, _ = min_gc_for_multiplicity(T, 0.0)
    above, below = Separable(T, gc=1.01 * g0, gs=0.0), Separable(T, gc=0.99 * g0, gs=0.0)
    fo = above.folds()
    assert len(fo) == 2 and not below.folds() and below.max_slope_deficit() < 0
    th = np.sqrt(fo[0][1] * fo[1][1])
    k1, k8 = above.steady_states(th, n=50_001), above.steady_states(th, n=400_001)
    assert len(k1) == 3 and np.allclose(k1, k8, rtol=1e-11)
    for th in np.exp(np.linspace(np.log(1e-3), np.log(1e3), 25)):
        assert len(below.steady_states(th)) == 1
    print('curvature threshold: 3 roots above, 1 below, grid-stable: ok')


def test_subsistence_threshold():
    """Subsistence just above cbar_min gives three steady states for some theta; just below, none fold."""
    cb, _, _ = min_cbar_for_multiplicity(T, 2.0, 2.0)
    assert len(Separable(T, gc=2.0, gs=2.0, cbar=1.001 * cb).folds()) == 2
    assert not Separable(T, gc=2.0, gs=2.0, cbar=0.999 * cb).folds()
    print('subsistence threshold: ok')


def test_kurz_example_and_flow_equivalence():
    """Kurz's Example 1 roots to within his 0.25, and the flow model with the mapped weight has the same roots."""
    KT = Technology(rho=0.3706, delta=0.01, alpha=0.45, A=0.3)
    kz = Separable(KT, gc=0.7, gs=0.2, model='stock').steady_states(0.2 / 0.3)
    assert len(kz) == 3 and all(abs(a - b) <= 0.25 for a, b in zip(kz, [6.25, 10.75, 19.75]))
    kf = Separable(KT, gc=0.7, gs=0.2, model='flow').steady_states(
        (0.2 / 0.3) / ((KT.rho + KT.delta) * KT.delta ** (-0.2)))
    assert np.allclose(kz, kf, rtol=1e-10)
    print('Kurz Example 1 reproduced; flow model has the same steady states: ok')


def test_min_rho_is_a_threshold():
    """min_rho_for_multiplicity separates one from several steady states."""
    r = min_rho_for_multiplicity(0.05, 1 / 3, 2.0, 0.0)
    assert Separable(Technology(1.02 * r, 0.05, 1 / 3), gc=2.0, gs=0.0).folds()
    assert not Separable(Technology(0.98 * r, 0.05, 1 / 3), gc=2.0, gs=0.0).folds()
    print('rho threshold: ok')


def test_middle_steady_state_is_an_unstable_node():
    """At a calibration with three steady states the outer two are saddles and the middle one has two real positive
    roots with det J < rho^2/4 (Wirl and Feichtinger 2005, Proposition 8: no unstable focus in a concave problem)."""
    t = Technology(rho=0.07, delta=0.03, alpha=0.4)
    S = Separable(t, gc=3.0, gs=0.0)
    fo = S.folds()
    th = np.sqrt(fo[0][1] * fo[1][1])
    ks = S.steady_states(th)
    dets = [np.linalg.det(flow_jacobian(S, th, k)) for k in ks]
    assert dets[0] < 0 and dets[2] < 0 and 0 < dets[1] < t.rho ** 2 / 4
    print('outer saddles, middle unstable node: ok')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
