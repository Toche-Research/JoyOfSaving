"""Where the multiplicity in Kurz (1968) comes from: the curvatures, not the stock.

Section 4 states that uniqueness of the steady state follows from homotheticity of felicity rather
than from putting the FLOW of saving into it, and that Kurz's multiplicity examples are not
homothetic.  This script establishes both claims and writes the numbers the paper quotes.

Kurz (1968), "Optimal Economic Growth and Wealth Effects", IER 9(3).  Stationary loci, p.355,
equations (20a)-(20b):

    (20a)   phi_1(k,c) = f(k) - c - n k           [the kdot = 0 locus]
    (20b)   phi_2(k,c) = U_k/U_c + f'(k) - delta  [the cdot = 0 locus]

His two multiplicity examples, p.356.  In BOTH, the curvature of felicity in the stock differs from
its curvature in consumption, so U_k/U_c is not a function of c/k alone and felicity is not
homothetic.  Giving the two arguments a common curvature leaves exactly one steady state, by the
argument of Remark 1 with c/k in place of c/s.

Writes:
  out/kurz_homothetic.txt    the report: Kurz's roots reproduced, then the homothetic counterfactual
  out/macros/kurz.tex        the curvatures quoted in Section 4
Run: cd code && PYTHONPATH=. python3 scripts/kurz_homothetic.py"""
import os
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(__file__)
OUTDIR = os.path.join(HERE, '..', 'out')
MACDIR = os.path.join(OUTDIR, 'macros')
os.makedirs(MACDIR, exist_ok=True)

# Kurz's two examples as printed on p.356.  gc and gk are the curvatures of felicity in consumption
# and in the stock: U_c = B c^-gc and U_k = A k^-gk.  Homothetic would require gc == gk.
EX = {
    'Example 1': dict(f=lambda k: 0.3 * k ** 0.45, fp=lambda k: 0.135 * k ** -0.55,
                      n=0.01, delta=0.3806, A=0.2, B=0.3, gc=0.7, gk=0.2,
                      hi=600.0, kurz=[6.25, 10.75, 19.75]),
    'Example 2': dict(f=lambda k: 0.5 * k ** 0.3 + 0.08 * k, fp=lambda k: 0.15 * k ** -0.7 + 0.08,
                      n=0.03, delta=0.725, A=0.7, B=0.6, gc=0.4, gk=0.3,
                      hi=1e7, kurz=[50.75, 111.75]),   # no kbar: f'(inf)=0.08 > n=0.03,
                                                       # so consumption is unbounded and the
                                                       # search window must not be truncated
}
TOL = 0.25          # the accuracy to which Kurz states his roots
GAMMAS = [0.2, 0.3, 0.4, 0.7, 1.0, 2.0, 5.0]


def steady_states(f, fp, n, delta, mrs, hi, m=400_000):
    """Zeros of phi_2 along the phi_1 = 0 locus, on the feasible range 0 < k < kbar."""
    k = np.linspace(0.05, hi, m)
    c = f(k) - n * k
    ok = c > 1e-9                                   # c <= 0 is outside the feasible range
    k = k[ok]
    v = mrs(k, f(k) - n * k) + fp(k) - delta
    fin = np.isfinite(v)
    k, v = k[fin], v[fin]
    phi = lambda x: mrs(x, f(x) - n * x) + fp(x) - delta
    s = np.sign(v)
    return [brentq(phi, k[i], k[i + 1], xtol=1e-11) for i in np.where(np.diff(s) != 0)[0]]


lines = ["KURZ (1968) p.356, HIS OWN EXAMPLES: felicity over the STOCK, unequal curvatures", ""]
for nm, e in EX.items():
    r = steady_states(e['f'], e['fp'], e['n'], e['delta'],
                      lambda k, c, e=e: (e['A'] * k ** -e['gk']) / (e['B'] * c ** -e['gc']), e['hi'])
    assert len(r) == len(e['kurz']), f"{nm}: {len(r)} roots, Kurz reports {len(e['kurz'])}"
    for got, want in zip(r, e['kurz']):
        assert abs(got - want) <= TOL, f"{nm}: {got:.2f} against Kurz's {want} (tolerance {TOL})"
    lines += [f"  {nm}   computed {[round(x, 2) for x in r]}",
              f"              Kurz p.356 {e['kurz']}   (reproduced to within {TOL})",
              f"              curvature {e['gc']} in c against {e['gk']} in k: NOT homothetic", ""]

lines += ["=" * 78, "", "THE SAME EXAMPLES WITH THE TWO CURVATURES MADE EQUAL (homothetic)", ""]
for nm, e in EX.items():
    lines.append(f"  {nm}:  U_k/U_c = ({e['A']}/{e['B']})(c/k)^gamma")
    for g in GAMMAS:
        r = steady_states(e['f'], e['fp'], e['n'], e['delta'],
                          lambda k, c, g=g, e=e: (e['A'] / e['B']) * (c / k) ** g, e['hi'])
        assert len(r) == 1, f"{nm}, gamma={g}: {len(r)} steady states under homotheticity"
        lines.append(f"     gamma={g:<4} -> {len(r)} steady state(s): {[round(x, 2) for x in r]}")
    lines.append("")

lines += ["=" * 78, "",
          "Every homothetic case gives at most one, by the argument of Remark 1 with c/k in place of",
          "c/s: U_k/U_c = m(c/k) with m increasing; c/k = f(k)/k - n falls strictly with k because f",
          "is concave with f(0) = 0; f' falls; so phi_2 is strictly decreasing and has at most one",
          "zero.", "",
          "Uniqueness is therefore a consequence of homotheticity and not of the flow formulation.",
          "It holds in the stock model too.  Kurz's multiplicity comes from the unequal curvatures."]

with open(os.path.join(OUTDIR, 'kurz_homothetic.txt'), 'w') as fh:
    fh.write("\n".join(lines) + "\n")

with open(os.path.join(MACDIR, 'kurz.tex'), 'w') as fh:
    fh.write("% generated by code/scripts/kurz_homothetic.py; do not edit\n")
    for key, nm in (('A', 'Example 1'), ('B', 'Example 2')):
        fh.write(f"\\newcommand{{\\kurzGc{key}}}{{{EX[nm]['gc']}}}\n")
        fh.write(f"\\newcommand{{\\kurzGk{key}}}{{{EX[nm]['gk']}}}\n")

print("\n".join(lines))
