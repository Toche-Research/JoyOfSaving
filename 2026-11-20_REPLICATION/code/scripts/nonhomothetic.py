"""Can non-homothetic (additively separable) utility give several steady states in the neoclassical model?

Feasibility study for a section on multiple steady states.  Utility u(c) + theta v(s) with
u'(c) = (c - cbar)^(-gc) and v'(s) = s^(-gs); the paper's eq:iso is cbar = 0, gc = gs = gamma.
Along the steady states theta*(k) = (1 - x) u'(c)/v'(s), x = f'/(rho + delta), and

    d ln theta*/d ln k = (1 - alpha) x/(1 - x) + sigma_s - sigma_c eps_c,

so several steady states exist for some theta iff sigma_c(c(k)) eps_c(k) > (1-alpha) x/(1-x) + sigma_s somewhere
in (k_MGR, k_GR), and only if sigma_c exceeds Gamma_min, a number fixed by rho/delta and alpha.

Writes:
  out/nonhomothetic.txt                  the report
  out/macros/nonhomothetic_numbers.tex   the numbers, as macros
Run: cd code && PYTHONPATH=. python3 scripts/nonhomothetic.py"""
import os
import numpy as np
from joy.nonhomothetic import (Technology, Separable, min_gc_for_multiplicity, min_cbar_for_multiplicity,
                               min_rho_for_multiplicity, flow_jacobian)
from joy.neoclassical import Economy

HERE = os.path.dirname(__file__)
OUTDIR = os.path.join(HERE, '..', 'out')
MACDIR = os.path.join(OUTDIR, 'macros')
os.makedirs(MACDIR, exist_ok=True)

RHO, DELTA, ALPHA, GAMMA = 0.03, 0.05, 1 / 3, 2.0       # the paper's calibration
macros, lines = {}, []


def mac(name, value, fmt):
    macros[name] = format(value, fmt)
    return macros[name]


T = Technology(RHO, DELTA, ALPHA)
lines += ["NON-HOMOTHETIC SEPARABLE UTILITY u(c) + theta v(s): CAN THERE BE SEVERAL STEADY STATES?", "",
          f"Paper's calibration: rho={RHO}, delta={DELTA}, alpha={ALPHA:.4f}, gamma={GAMMA}, A=1",
          f"  k_MGR={T.k_mgr:.4f}  k_GR={T.k_gr:.4f}  k_max={T.k_max:.4f}", ""]

# 0. consistency: the paper's isoelastic steady state is the root of theta*(k) = theta
for th in (0.05, 0.2, 1.0):
    k_paper = Economy(rho=RHO, delta=DELTA, gamma=GAMMA, theta=th, alpha=ALPHA).steady_state()['k']
    ks = Separable(T, gc=GAMMA, gs=GAMMA).steady_states(th)
    assert len(ks) == 1 and abs(ks[0] - k_paper) < 1e-9 * k_paper, (th, ks, k_paper)
iso = Separable(T, gc=GAMMA, gs=GAMMA)
assert iso.max_slope_deficit() < 0 and not iso.folds()
lines += ["0. Check: with the paper's utility theta* is strictly increasing on (k_MGR, k_max) and its root",
          "   equals the steady state of the paper's own solver at theta = 0.05, 0.2, 1 (to 1e-9).", ""]

# 1. the curvature bound at the calibration
G, kG, tauG = T.Gamma_min()
lines += ["1. Necessary condition (any concave u, v): sigma_c(c(k)) > Gamma(k) for some k in (k_MGR, k_GR).",
          f"   min Gamma = {G:.3f}, at k = {kG / T.k_mgr:.3f} k_MGR, where tau = {tauG:.3f}",
          f"   => the EIS 1/sigma_c must fall below {1 / G:.4f} at some steady-state consumption level",
          f"   largest elasticity of steady-state consumption in capital on (k_MGR,k_GR): eps_c(k_MGR) = "
          f"{T.eps_c(T.k_mgr * (1 + 1e-12)):.4f}", ""]
mac('nhGammaMin', G, '.1f')
mac('nhEISMax', 1 / G, '.3f')
mac('nhEpsCMax', T.eps_c(T.k_mgr * (1 + 1e-12)), '.2f')

# 1a. power utilities, unequal curvatures
g0, _ = min_gc_for_multiplicity(T, 0.0)
g2, _ = min_gc_for_multiplicity(T, GAMMA)
assert abs(g0 - G) < 1e-6 * G
lines += ["1a. Power utilities, u' = c^-gc, v' = s^-gs (scale free: neither A nor theta matters):",
          f"   several steady states for some theta iff gc > {g0:.3f} when gs = 0 (linear joy of saving)",
          f"                                          iff gc > {g2:.3f} when gs = {GAMMA}", ""]
mac('nhGcMinLinear', g0, '.1f')
mac('nhGcMinGsTwo', g2, '.1f')

# verify the threshold by counting roots just above and below it (gs = 0)
for gc, want_multi in ((g0 * 1.01, True), (g0 * 0.99, False)):
    S = Separable(T, gc=gc, gs=0.0)
    fo = S.folds()
    assert (len(fo) == 2) == want_multi, (gc, fo)
    if want_multi:
        th = np.sqrt(fo[0][1] * fo[1][1])
        assert len(S.steady_states(th)) == 3

# 1b. subsistence consumption (Stone-Geary) with the paper's common curvature
cmgr = T.c(T.k_mgr)
for gc in (GAMMA, 1.0):
    cb, kb, share = min_cbar_for_multiplicity(T, gc, gc)
    lines += [f"1b. Subsistence: u' = (c - cbar)^-{gc:g}, v' = s^-{gc:g}:",
              f"   several steady states for some theta iff cbar > {cb:.5f} = {100 * cb / cmgr:.2f} per cent of Ramsey "
              f"steady-state consumption",
              f"   ({100 * share:.2f} per cent of consumption where the condition first holds, k = {kb / T.k_mgr:.4f} k_MGR;"
              f" local EIS there {(1 - share) / gc:.4f})", ""]
    if gc == GAMMA:
        mac('nhCbarShare', 100 * cb / cmgr, '.1f')
        mac('nhCbarShareLocal', 100 * share, '.1f')
        mac('nhCbarEIS', (1 - share) / gc, '.3f')
        S = Separable(T, gc=gc, gs=gc, cbar=cb * 1.002)
        fo = S.folds()
        assert len(fo) == 2
        assert len(S.steady_states(np.sqrt(fo[0][1] * fo[1][1]))) == 3
        assert not Separable(T, gc=gc, gs=gc, cbar=cb * 0.998).folds()
    else:
        # log case: the bound is Ramsey consumption itself; below it theta* is increasing, above it U-shaped
        assert abs(cb / cmgr - 1) < 1e-6
        assert not Separable(T, gc=gc, gs=gc, cbar=cmgr * 0.999).folds()
        assert len(Separable(T, gc=gc, gs=gc, cbar=cmgr * 1.001).folds()) == 1
        lines.insert(-1, "   (log case: the bound is Ramsey consumption itself; above it theta* is U-shaped, giving two"
                         " steady states or none)")

# 1c. how large must rho/delta be?  Gamma depends on rho/delta and alpha only.
lines += ["1c. Smallest rho/delta at which power utilities give several steady states (gs = curvature of v):",
          "     alpha   gc   gs=0     gs=0.5"]
tab = {}
for a in (1 / 3, 0.4, 0.45):
    for gc in (2.0, 3.0, 4.0):
        r0 = min_rho_for_multiplicity(DELTA, a, gc, 0.0) / DELTA
        r5 = min_rho_for_multiplicity(DELTA, a, gc, 0.5) / DELTA
        tab[(a, gc)] = (r0, r5)
        lines.append(f"     {a:.3f}  {gc:.0f}    {r0:6.2f}   {r5:6.2f}")
# scale invariance in (rho, delta): same ratio at another delta
r_alt = min_rho_for_multiplicity(0.03, 1 / 3, 2.0, 0.0) / 0.03
assert abs(r_alt - tab[(1 / 3, 2.0)][0]) < 1e-6, (r_alt, tab[(1 / 3, 2.0)])
lines += [f"   (the paper's calibration has rho/delta = {RHO / DELTA:.2f}; the ratio, not the levels, matters:",
          f"    at delta = 0.03 the first entry is {r_alt:.2f} again)", ""]
mac('nhRhoDeltaPaper', RHO / DELTA, '.1f')
mac('nhRhoDeltaGcTwo', tab[(1 / 3, 2.0)][0], '.1f')
mac('nhRhoDeltaGcFour', tab[(1 / 3, 4.0)][0], '.1f')
mac('nhRhoDeltaGcThreeAlphaHigh', tab[(0.45, 3.0)][0], '.2f')

# box of calibrations
best = (np.inf, None)
for r in np.linspace(0.01, 0.06, 11):
    for d in np.linspace(0.03, 0.10, 15):
        for a in np.linspace(0.25, 0.45, 9):
            g = Technology(r, d, a).Gamma_min()[0]
            if g < best[0]:
                best = (g, (r, d, a))
lines += [f"1d. Over rho in [1,6]%, delta in [3,10]%, alpha in [0.25,0.45]: min Gamma = {best[0]:.2f},"
          f" at rho={best[1][0]:.3f}, delta={best[1][1]:.3f}, alpha={best[1][2]:.2f} (the corner).", ""]
mac('nhGammaBoxMin', best[0], '.1f')

# 2. an illustration at a favourable corner, to show what multiplicity looks like when it occurs
TE = Technology(rho=0.07, delta=0.03, alpha=0.4)
SE = Separable(TE, gc=3.0, gs=0.0)
fo = SE.folds()
assert len(fo) == 2
th_lo, th_hi = sorted([fo[0][1], fo[1][1]])
th = np.sqrt(th_lo * th_hi)
ks = SE.steady_states(th)
assert len(ks) == 3
lines += ["2. Favourable corner: rho=0.07, delta=0.03, alpha=0.4, u' = c^-3 (EIS 1/3), v(s) = s (linear joy).",
          f"   three steady states iff {th_lo:.6f} < theta < {th_hi:.6f}: a window {100 * (th_hi / th_lo - 1):.2f} per cent wide",
          f"   at theta = {th:.6f} (geometric middle of the window):"]
mac('nhCornerWindow', 100 * (th_hi / th_lo - 1), '.1f')
for i, k in enumerate(ks):
    J = flow_jacobian(SE, th, k)
    ev = np.linalg.eigvals(J)
    assert abs(np.trace(J) - TE.rho) < 1e-6 and np.all(np.isreal(ev))
    det = np.linalg.det(J)
    kind = 'saddle' if det < 0 else ('unstable node' if det < TE.rho ** 2 / 4 else 'unstable focus')
    assert kind == ('unstable node' if i == 1 else 'saddle')
    hl = np.log(2) / abs(min(ev.real)) if det < 0 else np.nan
    lines.append(f"     k = {k / TE.k_gr:.3f} k_GR, saving rate {100 * TE.delta * k / TE.f(k):.1f} per cent, "
                 f"tau = {1 - TE.x(k):.3f}, eigenvalues {np.sort(ev.real)}, {kind}"
                 + (f", half-life {hl:.0f} years" if det < 0 else ""))
    tag = 'ABC'[i]
    mac(f'nhCornerSav{tag}', 100 * TE.delta * k / TE.f(k), '.0f')
    if det < 0:
        mac(f'nhCornerHalf{tag}', hl, '.0f')
lines.append("")

# 3. Kurz's stock model: same criterion, same steady states under theta_K = (rho+delta) delta^-gs theta
KT = Technology(rho=0.3806 - 0.01, delta=0.01, alpha=0.45, A=0.3)      # Kurz Example 1, p.356
KS = Separable(KT, gc=0.7, gs=0.2, model='stock')
kz = KS.steady_states(0.2 / 0.3)
assert len(kz) == 3 and all(abs(a - b) <= 0.25 for a, b in zip(kz, [6.25, 10.75, 19.75]))
KF = Separable(KT, gc=0.7, gs=0.2, model='flow')
kf = KF.steady_states((0.2 / 0.3) / ((KT.rho + KT.delta) * KT.delta ** (-0.2)))
assert np.allclose(kz, kf, rtol=1e-10)
KG = KT.Gamma_min()[0]
rho_min_kurz = min_rho_for_multiplicity(KT.delta, KT.alpha, 0.7, 0.2)
rho_min_paper = min_rho_for_multiplicity(DELTA, ALPHA, 0.7, 0.2)
lines += ["3. Kurz (1968) Example 1 (stock model; rho + delta = 0.3806, delta = 0.01, alpha = 0.45):",
          f"   steady states {[round(x, 2) for x in kz]} (Kurz p.356: 6.25, 10.75, 19.75)",
          f"   the flow model with v(s) = s^0.8/0.8 and theta = theta_K delta^0.2/(rho+delta) has the SAME steady states",
          f"   min Gamma at Kurz's parameters = {KG:.3f} (rho/delta = {KT.rho / KT.delta:.1f}); his gc = 0.7 clears it",
          f"   with Kurz's curvatures (0.7, 0.2) and his delta, alpha: several steady states iff rho > {rho_min_kurz:.4f}"
          f" (he uses {KT.rho:.4f})",
          f"   with Kurz's curvatures and the paper's delta, alpha: rho_min = {rho_min_paper} (none below rho = 10)", ""]
mac('nhKurzGammaMin', KG, '.2f')
mac('nhKurzRhoDelta', KT.rho / KT.delta, '.0f')
mac('nhKurzRhoMin', rho_min_kurz, '.3f')

with open(os.path.join(OUTDIR, 'nonhomothetic.txt'), 'w') as fh:
    fh.write("\n".join(lines) + "\n")
with open(os.path.join(MACDIR, 'nonhomothetic_numbers.tex'), 'w') as fh:
    fh.write("% generated by code/scripts/nonhomothetic.py; do not edit\n")
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("\n".join(lines))
