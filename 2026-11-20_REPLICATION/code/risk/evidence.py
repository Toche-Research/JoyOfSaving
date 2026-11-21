"""Model counterparts of the evidence on the demand for dividend income (portfolio corollary of risk_section.tex).

Uses the machinery of risk.py (same folder): for a constant portfolio the wealth process is that of
Proposition prop:risk with
    y(pi) = yf + pi . Delta,   x(pi) = pi . x,   sigma(pi)^2 = pi' Sigma pi,
so the value of each constant portfolio is in closed form, K(pi) a^(1-gamma)/(1-gamma) + const, and the best
constant portfolio maximizes K/(1-gamma). Corollary cor:portfolio is stated for one risky asset; the first-order
condition with several risky assets, pi* = (gamma Sigma)^(-1) (x + Delta/(1-tau)), is checked here numerically only.

Outputs (folder of this script): evidence_output.txt (log), evidence_macros.tex (numbers quoted in
evidence_paragraphs.tex). Run: python3 evidence.py. Every number in evidence_paragraphs.tex and EVIDENCE.md that is
a model number comes from this script.
"""
import os

import numpy as np
from scipy.optimize import minimize, minimize_scalar

import risk  # imports figstyle (pgf backend); no figure is drawn here
from risk import solve, condition, best_pi, value_K

HERE = os.path.dirname(os.path.abspath(__file__))
G, RHO, TH = risk.GAMMA, risk.RHO, risk.THETA
lines, macros = [], []


def log(s=''):
    print(s)
    lines.append(s)


def macro(name, value, comment):
    assert name.isalpha(), name
    macros.append('\\newcommand{\\%s}{%s}%% %s' % (name, value, comment))


def merton(yf, yr, xr, sr, gamma=G):
    return (yr - yf + xr) / (gamma * sr ** 2)


def tau_at(pi, yf, yr, xr, sr, rho=RHO, gamma=G, theta=TH):
    return solve(yf + pi * (yr - yf), pi * xr, abs(pi) * sr, rho, gamma, theta)['tau']


def one_asset(yf, yr, xr, sr, rho=RHO, gamma=G, theta=TH):
    pM = merton(yf, yr, xr, sr, gamma)
    pJ, turns = best_pi(yf, yr, xr, sr, rho, gamma, theta)
    t = tau_at(pJ, yf, yr, xr, sr, rho, gamma, theta)
    formula = (xr + (yr - yf) / (1 - t)) / (gamma * sr ** 2)
    assert abs(pJ - formula) < 1e-6 and turns <= 1, (pJ, formula, turns)
    return pM, pJ, t


# ------------------------------------------------------------------------------------------------------------------
# Several risky assets: value of a constant portfolio vector
# ------------------------------------------------------------------------------------------------------------------
def K_vec(pi, yf, yv, xv, Sig, rho=RHO, gamma=G, theta=TH):
    pi = np.asarray(pi, float)
    y = yf + pi @ (yv - yf)
    x = pi @ xv
    s = float(np.sqrt(max(pi @ Sig @ pi, 0.0)))
    if y <= 0 or not condition(rho, gamma, y, x, s):
        return np.nan
    S = solve(y, x, s, rho, gamma, theta)
    if theta == 0:
        return S['z'] ** (-gamma)
    return S['z'] ** (-gamma) - theta * S['h'] ** (-gamma), S


def best_vec(yf, yv, xv, Sig, rho=RHO, gamma=G, theta=TH, bounds=None):
    sgn = 1.0 / (1 - gamma)

    def obj(p):
        r = K_vec(p, yf, yv, xv, Sig, rho, gamma, theta)
        if isinstance(r, float) and np.isnan(r):
            return 1e6
        K = r[0] if isinstance(r, tuple) else r
        return -sgn * K

    # start from the Merton portfolio (clipped to the bounds when there are any)
    p0 = np.linalg.solve(gamma * Sig, (yv - yf) + xv)
    if bounds is not None:
        p0 = np.clip(p0, [b[0] for b in bounds], [b[1] for b in bounds])
    best = None
    for start in (p0, p0 * 0.5, np.full(len(yv), 0.3)):
        r = minimize(obj, start, method='L-BFGS-B' if bounds else 'Nelder-Mead', bounds=bounds,
                     options={'xatol': 1e-12, 'fatol': 1e-16, 'maxiter': 20000} if not bounds else {'ftol': 1e-15})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def main():
    log('Model counterparts of the evidence on the demand for dividend income.')
    log('gamma=%g, rho=%g, theta=%g unless stated. Safe asset yields yf; risky asset: income yield yr, expected gain'
        ' xr, volatility sr.' % (G, RHO, TH))
    log()

    # ---- 0. reproduce the table of risk_section.tex ----
    log('0. Reproduction of tbl_portfolio (yf=2%, expected return 6%, sr=0.2)')
    base = {}
    for yr in (0.0, 0.02, 0.04, 0.06):
        pM, pJ, t = one_asset(0.02, yr, 0.06 - yr, 0.2)
        base[yr] = (pM, pJ, t)
        log('   yr=%.0f%%: Merton %.4f, joy of saving %.4f, tau %.4f' % (100 * yr, pM, pJ, t))
    log()

    # ---- 1. safe yield (Hartzmark-Solomon 2019, Table VIII: dividend demand higher when interest rates are low) ----
    log('1. The safe yield. Risky asset fixed: yr=4%, xr=2%, sr=0.2; yf varies.')
    rows = []
    for yf in (0.01, 0.02, 0.03, 0.04):
        pM, pJ, t = one_asset(yf, 0.04, 0.02, 0.2)
        rows.append((yf, pM, pJ, t))
        log('   yf=%.0f%%: Merton %.4f, joy of saving %.4f, excess %.4f, ratio %.4f, tau %.4f'
            % (100 * yf, pM, pJ, pJ - pM, pJ / pM, t))
    ex = [r[2] - r[1] for r in rows]
    rat = [r[2] / r[1] for r in rows]
    log('   excess share falls as yf rises: %s; ratio falls: %s' % (bool(np.all(np.diff(ex) < 0)),
                                                                   bool(np.all(np.diff(rat) < 0))))
    assert np.all(np.diff(ex) < 0) and abs(ex[-1]) < 1e-6
    # same comparison with the expected excess return held fixed: lower yf with a lower risky gain
    log('   expected excess return held at 4 points: yr = yf + 2%, xr = 2%')
    rows2 = []
    for yf in (0.01, 0.02, 0.03):
        pM, pJ, t = one_asset(yf, yf + 0.02, 0.02, 0.2)
        rows2.append((yf, pM, pJ, t))
        log('   yf=%.0f%%, yr=%.0f%%: Merton %.4f, joy of saving %.4f, tau %.4f' % (100 * yf, 100 * (yf + 0.02), pM, pJ, t))
    macro('evPiYfOne', '%.2f' % rows[0][2], 'JoS share, yf=1, yr=4, xr=2')
    macro('evPiYfTwo', '%.2f' % rows[1][2], 'JoS share, yf=2, yr=4, xr=2')
    macro('evPiYfThree', '%.2f' % rows[2][2], 'JoS share, yf=3, yr=4, xr=2')
    macro('evPiMYfOne', '%.3f' % rows[0][1], 'Merton share, yf=1')
    macro('evPiMYfThree', '%.3f' % rows[2][1], 'Merton share, yf=3')
    macro('evRatioYfOne', '%.2f' % rat[0], 'JoS/Merton, yf=1')
    macro('evRatioYfThree', '%.2f' % rat[2], 'JoS/Merton, yf=3')
    log()

    # ---- 2. volatility (Hartzmark-Solomon 2013, Table 5: dividend-month returns higher with VIX and in recessions) ----
    log('2. Volatility. yf=2%, yr=4%, xr=2%; sr varies.')
    vrows = []
    for sr in (0.15, 0.20, 0.25, 0.30):
        pM, pJ, t = one_asset(0.02, 0.04, 0.02, sr)
        vrows.append((sr, pM, pJ, t))
        log('   sr=%.2f: Merton %.4f, joy of saving %.4f, ratio %.4f, tau %.4f' % (sr, pM, pJ, pJ / pM, t))
    log('   ratio JoS/Merton over sr: %s' % ['%.4f' % (r[2] / r[1]) for r in vrows])
    log()

    # ---- 3. impatience and the weight theta (a proxy for the life cycle; Shefrin-Statman Table 1, Pettit) ----
    log('3. Impatience and theta. yf=2%, yr=4%, xr=2%, sr=0.2.')
    rr = []
    for rho in (0.01, 0.02, 0.03, 0.05, 0.08):
        pM, pJ, t = one_asset(0.02, 0.04, 0.02, 0.2, rho=rho)
        S = solve(0.02 + pJ * 0.02, pJ * 0.02, pJ * 0.2, rho, G, TH)
        rr.append((rho, pJ, t, S['z']))
        log('   rho=%.0f%%: joy of saving %.4f (Merton %.4f), tau %.4f, c/a %.3f%%, h %.3f%%'
            % (100 * rho, pJ, pM, t, 100 * S['z'], 100 * S['h']))
    tt = []
    for th in (0.02, 0.05, 0.2, 1.0, 5.0):
        pM, pJ, t = one_asset(0.02, 0.04, 0.02, 0.2, theta=th)
        S = solve(0.02 + pJ * 0.02, pJ * 0.02, pJ * 0.2, RHO, G, th)
        tt.append((th, pJ, t, S['z']))
        log('   theta=%g: joy of saving %.4f (Merton %.4f), tau %.4f, c/a %.3f%%, h %.3f%%'
            % (th, pJ, pM, t, 100 * S['z'], 100 * S['h']))
    log('   share rises with rho: %s; share with theta: %s'
        % (bool(np.all(np.diff([r[1] for r in rr]) > 0)), ['%.3f' % r[1] for r in tt]))
    macro('evPiRhoLow', '%.2f' % rr[0][1], 'JoS share, rho=1%')
    macro('evPiRhoHigh', '%.2f' % rr[-1][1], 'JoS share, rho=8%')
    macro('evCaRhoLow', '%.2f' % (100 * rr[0][3]), 'c/a %, rho=1%')
    macro('evCaRhoHigh', '%.2f' % (100 * rr[-1][3]), 'c/a %, rho=8%')
    log()

    # ---- 4. two risky assets with the same expected return and volatility, different yields ----
    log('4. Two risky assets, equal expected return 6% and volatility 0.2, yields 0% (A) and 4% (B); yf=2%.')
    yf = 0.02
    yv = np.array([0.0, 0.04])
    xv = 0.06 - yv
    out4 = {}
    for corr in (0.0, 0.5, 0.8):
        Sig = 0.04 * np.array([[1.0, corr], [corr, 1.0]])
        pM = np.linalg.solve(G * Sig, (yv - yf) + xv)
        pJ = best_vec(yf, yv, xv, Sig)
        K, S = K_vec(pJ, yf, yv, xv, Sig)
        t = S['tau']
        formula = np.linalg.solve(G * Sig, xv + (yv - yf) / (1 - t))
        gap = float(np.max(np.abs(pJ - formula)))
        # no short sales, no borrowing on the risky side: shares in [0, 1]
        pC = best_vec(yf, yv, xv, Sig, bounds=[(0.0, 1.0), (0.0, 1.0)])
        wM, wJ, wC = pM[1] / pM.sum(), pJ[1] / pJ.sum(), pC[1] / pC.sum()
        out4[corr] = (pM, pJ, pC, wJ, wC, t, gap)
        log('   corr=%.1f: Merton (%.4f, %.4f), weight of B in equity %.3f; joy of saving (%.4f, %.4f), weight of B %.3f,'
            ' tau %.4f, |pi - formula| %.1e; with shares in [0,1]: (%.4f, %.4f), weight of B %.3f'
            % (corr, pM[0], pM[1], wM, pJ[0], pJ[1], wJ, t, gap, pC[0], pC[1], wC))
        assert gap < 2e-4, gap
    a, b = out4[0.5], out4[0.0]
    macro('evTwoAzero', '%.2f' % b[1][0], 'JoS share in A (yield 0), corr 0')
    macro('evTwoBzero', '%.2f' % b[1][1], 'JoS share in B (yield 4), corr 0')
    macro('evTwoMzero', '%.2f' % (np.linalg.solve(G * 0.04 * np.eye(2), (yv - yf) + xv)[0]), 'Merton share in each, corr 0')
    macro('evTwoWzero', '%.2f' % b[3], 'weight of B in equity, corr 0')
    macro('evTwoMhalf', '%.2f' % a[0][0], 'Merton share in each, corr 0.5')
    macro('evTwoAhalf', '%.2f' % a[1][0], 'JoS share in A, corr 0.5')
    macro('evTwoBhalf', '%.2f' % a[1][1], 'JoS share in B, corr 0.5')
    macro('evTwoCAhalf', '%.2f' % a[2][0], 'JoS share in A, corr 0.5, no short sales')
    macro('evTwoCBhalf', '%.2f' % a[2][1], 'JoS share in B, corr 0.5, no short sales')
    log()

    # ---- 5. the fall of the share at high yields: size ----
    log('5. The share at high yields (yf=2%, expected return 6%, sr=0.2), step 0.5 points:')
    scan = [(yr, best_pi(0.02, yr, 0.06 - yr, 0.2, RHO, G, TH)[0]) for yr in np.arange(0.03, 0.0601, 0.005)]
    log('   ' + ', '.join('%.1f%%: %.4f' % (100 * y, p) for y, p in scan))
    pk = max(p for _, p in scan)
    log('   fall from the peak to yr=6%%: %.4f, that is %.1f%% of the peak share' % (pk - scan[-1][1],
                                                                                 100 * (pk - scan[-1][1]) / pk))
    macro('evFall', '%.3f' % (pk - scan[-1][1]), 'fall of the share from the peak to yr=6')
    log()

    # ---- 6. Graham and Kumar (2006): portfolio dividend yield against impatience (a proxy for old age) ----
    log('6. Portfolio dividend yield (PDY) of the equity holding, two stocks with equal expected return 6% and')
    log('   volatility 0.2, yields 0% (A) and 4% (B), yf=2%; Merton PDY = 2% (equal weights). Shares in [0, 1].')
    out6 = {}
    for corr in (0.0, 0.5):
        Sig = 0.04 * np.array([[1.0, corr], [corr, 1.0]])
        rows6 = []
        for rho in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
            pJ = best_vec(yf, yv, xv, Sig, rho=rho, bounds=[(0.0, 1.0), (0.0, 1.0)])
            K, S = K_vec(pJ, yf, yv, xv, Sig, rho=rho)
            pdy = 100 * (pJ @ yv) / pJ.sum()
            if np.all(pJ > 1e-6) and np.all(pJ < 1 - 1e-6):  # interior: compare with the two-asset formula
                form = np.linalg.solve(G * Sig, xv + (yv - yf) / (1 - S['tau']))
                assert np.max(np.abs(form - pJ)) < 2e-4, (corr, rho, pJ, form)
            rows6.append((rho, pdy, 100 * S['z'], pJ[1] / pJ.sum()))
            log('   corr=%.1f rho=%.0f%%: shares (%.4f, %.4f), weight of B %.3f, PDY %.3f%%, c/a %.3f%%'
                % (corr, 100 * rho, pJ[0], pJ[1], pJ[1] / pJ.sum(), pdy, 100 * S['z']))
        r6 = np.array(rows6)
        slope = np.polyfit(r6[:, 2], r6[:, 1], 1)[0]
        log('   (interior solutions match the two-asset formula to 2e-4)')
        log('   corr=%.1f: PDY rises with rho: %s; least-squares slope dPDY/d(c/a) over the grid = %.3f;'
            ' c/a difference giving +0.308 point of PDY = %.2f points'
            % (corr, bool(np.all(np.diff(r6[:, 1]) > 0)), slope, 0.308 / slope))
        out6[corr] = (rows6, slope)
    rows6, slope0 = out6[0.0]
    macro('evPdyRhoLow', '%.2f' % rows6[0][1], 'PDY %, corr 0, rho=1%')
    macro('evPdyRhoHigh', '%.2f' % rows6[-1][1], 'PDY %, corr 0, rho=8%')
    macro('evCaPdyLow', '%.2f' % rows6[0][2], 'c/a %, corr 0, rho=1%')
    macro('evCaPdyHigh', '%.2f' % rows6[-1][2], 'c/a %, corr 0, rho=8%')
    macro('evPdyMerton', '2.00', 'Merton PDY %, equal weights in yields 0 and 4')
    macro('evPdySlope', '%.2f' % slope0, 'dPDY/d(c/a), corr 0, least squares over rho grid')
    macro('evGKca', '%.2f' % (0.308 / slope0), 'c/a difference (points) giving +0.308 PDY, corr 0')
    log()

    # ---- 7. Share of a dividend consumed against theta (Brauer et al. 2022: savers consume more of a dividend) ----
    log('7. Share consumed of a unit of return moved from gain to yield, 1 + dh*/dx at r = 5%, sigma = 0')
    shares = {}
    for x0 in (0.0, 0.02):
        row = []
        for th in (1e-4, 0.01, 0.05, 0.2, 1.0, 5.0):
            e = 1e-6
            hp = solve(0.05 - (x0 + e), x0 + e, 0.0, theta=th)['h']
            hm = solve(0.05 - max(x0 - e, 0.0), max(x0 - e, 0.0), 0.0, theta=th)['h']
            d = (hp - hm) / ((x0 + e) - max(x0 - e, 0.0))
            S = solve(0.05 - x0, x0, 0.0, theta=th)
            row.append((th, 1 + d, S['h'] / (0.05 - x0)))
        shares[x0] = row
        log('   x=%.0f%%: ' % (100 * x0) + ', '.join('theta=%g: share %.3f (saving rate out of income %.2f)'
                                                     % (a, b, c) for a, b, c in row))
    s02 = [b for a, b, c in shares[0.0] if a == 0.2][0]
    s22 = [b for a, b, c in shares[0.02] if a == 0.2][0]
    for x0 in (0.0, 0.02):
        row = []
        for rho in (0.01, 0.02, 0.03, 0.04, 0.05):
            e = 1e-6
            hp = solve(0.05 - (x0 + e), x0 + e, 0.0, rho=rho)['h']
            hm = solve(0.05 - max(x0 - e, 0.0), max(x0 - e, 0.0), 0.0, rho=rho)['h']
            d = (hp - hm) / ((x0 + e) - max(x0 - e, 0.0))
            row.append((rho, 1 + d, solve(0.05 - x0, x0, 0.0, rho=rho)['h'] / (0.05 - x0)))
        shares[('rho', x0)] = row
        log('   theta=0.2, x=%.0f%%: ' % (100 * x0) + ', '.join('rho=%.0f%%: share %.3f (saving rate %.2f)'
                                                              % (100 * a, b, c) for a, b, c in row))
    log('   check against the article: theta=0.2 gives %.2f (x=0) and %.2f (x=2%%); article 0.35 and 0.54' % (s02, s22))
    assert abs(s02 - 0.35) < 0.006 and abs(s22 - 0.54) < 0.006
    log('   share falls with theta when x=2%%: %s; with x=0 the share is %s in theta'
        % (bool(np.all(np.diff([b for a, b, c in shares[0.02]]) < 0)),
           'monotone' if (np.all(np.diff([b for a, b, c in shares[0.0]]) > 0) or
                          np.all(np.diff([b for a, b, c in shares[0.0]]) < 0)) else 'not monotone'))
    sh_lo = [b for a, b, c in shares[0.02] if a == 0.05][0]
    sh_hi = [b for a, b, c in shares[0.02] if a == 1.0][0]
    macro('evShareThetaLo', '%.2f' % sh_lo, 'share consumed, x=2%, theta=0.05')
    macro('evShareThetaHi', '%.2f' % sh_hi, 'share consumed, x=2%, theta=1')
    macro('evSaveThetaLo', '%.2f' % [c for a, b, c in shares[0.02] if a == 0.05][0], 'saving rate out of income, x=2%, theta=0.05')
    macro('evSaveThetaHi', '%.2f' % [c for a, b, c in shares[0.02] if a == 1.0][0], 'saving rate out of income, x=2%, theta=1')
    log()

    with open(os.path.join(HERE, 'evidence_output.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(HERE, 'evidence_macros.tex'), 'w') as f:
        f.write('% generated by evidence.py; do not edit\n' + '\n'.join(macros) + '\n')


if __name__ == '__main__':
    main()
