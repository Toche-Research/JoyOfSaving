"""The paper's three figures, written to code/out/figures/ as PDF (used by the paper) and PNG (for drafts).

  fig_linear.pdf      Figure 1: linear technology. Growth of consumption and wealth, and tau = U_s/U_c,
                       against the weight theta on the joy of saving, for three values of gamma.
  fig_phase.pdf       Figure 2: neoclassical technology. Phase diagram in (k, c): the stationary loci, the
                       flow, and the saddle paths with and without the joy of saving.
  fig_transition.pdf  Figure 3: neoclassical technology. Transition from half the modified-golden-rule
                       capital stock: capital, consumption and the gross saving rate against time.

The figures carry no number of their own into the text: every number quoted in the prose or in the tables
comes from scripts/make_tables.py.
Run: cd code && PYTHONPATH=. python3 scripts/figures.py
An output directory may be given as the single argument, which is how draft variants are kept apart."""
import os, sys
import numpy as np
from scipy.integrate import solve_ivp
from joy import linear, neoclassical
# joy.style selects the pgf backend, which matplotlib fixes at the moment pyplot is first imported.
# It therefore has to be imported before pyplot, and it raises if it is not. Do not reorder these
# two lines into alphabetical or any other order: the script will stop on import.
from joy.style import setup, save, BLUE, ORANGE, AQUA, INK, INK2, GRID, BENCH, DARKORANGE, STREAM, LABELBOX
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'out', 'figures')
R, RHO = 0.05, 0.03                 # linear technology
THETA_PHASE = 0.2                   # joy of saving in the phase diagram and the transition
THETAS_TRANS = (0.05, 0.2)


# ------------------------------------------------------------ Figure 1: growth and the strength of the motive (linear)
def fig_linear():
    """Figure 1: growth and tau against theta on a log scale, for gamma = 1, 2 and 4, with each Ramsey
    growth rate as a dotted horizontal line and the ceiling g = r marked."""
    thetas = np.logspace(-3, 1, 300)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    # the gamma = 1 curve is fine-dotted, so its dots are drawn a little thicker than the other two
    # in order to read at the printed size
    for gam, col, ls, lw in ((1.0, AQUA, (0, (1, 1.5)), 2.1), (2.0, BLUE, '-', 1.6), (4.0, ORANGE, (0, (5, 2)), 1.6)):
        g = np.array([linear.growth(R, RHO, gam, th) for th in thetas])
        tau = linear.tau_lhs(g, R, RHO, gam)
        axs[0].plot(thetas, 100 * g, color=col, ls=ls, lw=lw, label=rf'$\gamma={gam:g}$')
        axs[1].plot(thetas, tau, color=col, ls=ls, lw=lw)
        axs[0].axhline(100 * linear.ramsey_growth(R, RHO, gam), color=col, lw=0.6, ls=':')
    axs[0].axhline(100 * R, color='black', lw=0.8, ls=':')
    axs[0].text(thetas[0], 100 * R, r' $g=r$', va='bottom', fontsize=7, color='black', bbox=LABELBOX)
    axs[0].set_xscale('log'); axs[1].set_xscale('log')
    axs[0].set_xlabel(r'strength of the joy of saving, $\theta$'); axs[1].set_xlabel(r'strength of the joy of saving, $\theta$')
    axs[0].set_ylabel(r'growth rate $g^*$, %'); axs[1].set_ylabel(r'$\tau=U_s/U_c$')
    # opaque white legend box (grid lines behind the text), black entries
    axs[0].legend(loc='lower right', fontsize=7, frameon=True, framealpha=1.0, facecolor='white', edgecolor='none',
                  labelcolor='black')
    axs[1].set_ylim(0, 1)
    fig.tight_layout()
    save(fig, 'fig_linear', OUT)


# ------------------------------------------------------------ Figure 2: phase diagram (neoclassical)
FS2 = 8                 # annotation font size in the phase diagram (one step above the other figures)
LONGDASH = (0, (6, 3))  # left from an earlier draft, when the benchmark capital stocks were drawn across the
                        # panel as long-dashed vertical lines; they are now marked on the axis instead, so no
                        # curve in the figure uses this pattern
LOCUSBOX = dict(facecolor='white', edgecolor='none', pad=2.0, alpha=1.0)   # roomier box behind the locus labels


def _motion_arrows(ax, k, c, kstar, kmax, col, lw, zorder=5):
    """Stealth tips on a saddle path, two each side, pointing at the steady state.

    Left of the steady state capital is being accumulated, so motion is to the
    right; to its right, motion is to the left.  Each tip is drawn as a short
    annotation along the path, so it sits on the curve and inherits its slope.
    """
    ks = [kstar * (1 - f) for f in (0.40, 0.18)] + \
         [kstar + g * (kmax - kstar) for g in (0.13, 0.33)]
    step = 0.018 * kmax
    for ka in ks:
        if not (k[0] < ka < k[-1]):
            continue
        d = step if ka < kstar else -step
        kb = ka + d
        if not (k[0] < kb < k[-1]):
            continue
        ca, cb = np.interp(ka, k, c), np.interp(kb, k, c)
        ax.annotate('', xy=(kb, cb), xytext=(ka, ca),
                    arrowprops=dict(arrowstyle='-|>,head_length=0.85,head_width=0.22',
                                    color=col, lw=lw, shrinkA=0, shrinkB=0,
                                    mutation_scale=8), zorder=zorder)


def _divergent_paths(ax, e, k_arm, c_arm, kmax, ymax, nstart=4, lift=0.05, col=INK2, lw=0.7):
    """Trajectories started above the stable arm, drawn across the upper part of the panel.

    The streamplot shows the direction of motion everywhere; these show what becomes of an economy
    that does not start on the arm.  A point above the arm has consumption too high for the capital
    in place: gross saving falls short of depreciation, the stock runs down, and consumption falls
    with it, so the path moves down and to the left and never returns.  Nothing about it is a
    knife-edge, which is the point of drawing several: the whole region above the arm behaves this
    way, and only the arm itself reaches the steady state.

    The family is started on a vertical near the right-hand edge and spread in consumption from
    just above the arm to the top of the frame, so that the paths fan out across the upper region
    rather than retracing one another.  Consumption cannot exceed output, so the highest start is
    set by f(k) at that capital and not by the frame.

    Each path is carried until consumption reaches output, which is where gross saving runs out
    and the path leaves the economically meaningful part of the plane.  Stopping it earlier, when
    saving is down to a tenth of output, leaves the four ends sitting on the Ramsey arm, because at
    that capital the Ramsey arm happens to save about a tenth of output as well; the coincidence
    invites the reading that these paths approach the arm, and they do not, the arm belonging to a
    different economy.  Carried to the ceiling they cross it and go on, which settles the matter.
    The arrow is set at the midpoint of each arc in arc length.
    """
    def stop_top(t, y):
        return y[1] - ymax
    stop_top.terminal, stop_top.direction = True, 1

    def stop_ceiling(t, y):
        return 0.97 * e.f(y[0]) - y[1]
    stop_ceiling.terminal, stop_ceiling.direction = True, -1

    k0 = 0.95 * kmax
    c_lo = np.interp(k0, k_arm, c_arm) + lift * ymax
    c_hi = 0.80 * e.f(k0)
    for c0 in np.linspace(c_lo, c_hi, nstart):
        if not (c_lo <= c0 < ymax) or e.f(k0) - c0 <= 1e-3:
            continue
        r = solve_ivp(e.rhs, (0, 400), [k0, c0], events=[stop_top, stop_ceiling],
                      rtol=1e-9, atol=1e-11, max_step=0.05, dense_output=True)
        k, c = r.y[0], r.y[1]
        keep = (k > 0) & (k <= kmax) & (c <= ymax)
        if keep.sum() < 8:
            continue
        k, c = k[keep], c[keep]
        ax.plot(k, c, color=col, lw=lw, ls='-', zorder=2)
        # one tip at the midpoint in arc length, pointing the way time runs
        d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(k) / kmax, np.diff(c) / ymax))])
        i = int(np.searchsorted(d, 0.5 * d[-1]))
        i = min(max(i, 1), len(k) - 2)
        ax.annotate('', xy=(k[i + 1], c[i + 1]), xytext=(k[i - 1], c[i - 1]),
                    arrowprops=dict(arrowstyle='-|>,head_length=0.8,head_width=0.22',
                                    color=col, lw=lw, shrinkA=0, shrinkB=0,
                                    mutation_scale=8), zorder=2)


def fig_phase():
    """Figure 2: phase diagram at theta = THETA_PHASE against the Ramsey case. The stationary loci are thin
    and solid, the saddle paths thick; both saddle paths are carried down to the same low capital stock so
    that the panel compares them over a common range. A family of grey trajectories started above the
    joy-of-saving arm shows the motion off the arm."""
    e = neoclassical.Economy(theta=THETA_PHASE); e0 = neoclassical.Economy(theta=0.0)
    ss, ss0 = e.steady_state(), e0.steady_state()
    kmax = 1.6 * ss['k']
    fig, ax = plt.subplots(figsize=(6.9, 4.56))
    kstart = 1e-5
    kk = np.concatenate([np.geomspace(kstart, 0.02 * ss['k'], 160),
                         np.linspace(0.02 * ss['k'], kmax, 400)])
    # k-dot = 0 locus (common): c = f(k) - delta k.  Solid, thin: the loci are thinner than
    # the trajectories, which is what separates the two kinds of curve in this panel.
    ax.plot(kk, e.f(kk) - e.delta * kk, color=INK, lw=1.0, ls='-', zorder=3)
    # c-dot = 0 locus with the joy of saving: a closed curve, drawn from its lower and upper branches
    kl = np.concatenate([np.geomspace(kstart, 0.02 * ss['k'], 160),
                         np.linspace(0.02 * ss['k'], kmax, 600)])
    c_lo, c_hi = e.cdot_zero_locus(kl)
    ok = ~np.isnan(c_lo)
    k_loop = np.concatenate([kl[ok], kl[ok][::-1]]); c_loop = np.concatenate([c_lo[ok], c_hi[ok][::-1]])
    ax.plot(k_loop, c_loop, color=INK, lw=1.0, ls='-', zorder=3)
    # saddle paths (computed before the limits: the Ramsey path sets the height of the panel)
    k_lo = 0.1                              # both arms carried down to the same low capital,
    k0, c0 = e0.saddle_path(k_lo / ss0['k'], kmax / ss0['k'])   # by backward integration from k*
    k1, c1 = e.saddle_path(k_lo / ss['k'], kmax / ss['k'])
    c0_end, c1_end = np.interp(kmax, k0, c0), np.interp(kmax, k1, c1)
    # The panel is cut just above the top tick at 3.0, by the same fraction of a tick interval
    # as the x-axis is cut above 35.  The Ramsey arm then leaves through the top edge, so its
    # margin label is clamped to the top rather than set level with the curve's end.
    ymax = 3.0 + ((kmax - 35.0) / 5.0) * 0.5
    c_kdot_end = e.f(kmax) - e.delta * kmax
    # Ramsey c-dot = 0 locus (vertical, thin and solid like the other loci), labelled above the panel,
    # centred on the line
    ax.axvline(ss0['k'], color=INK, lw=1.0, ls='-', zorder=2)
    ax.annotate('$\\dot c=0$\n(Ramsey)', (ss0['k'], ymax), xytext=(0, 3), textcoords='offset points', fontsize=FS2,
                color=INK, ha='center', va='bottom', linespacing=1.1, annotation_clip=False, clip_on=False)
    # the loop's label sits just above its upper branch, in the clear band between the loop
    # and the Ramsey saddle path, and is named so that it pairs with the Ramsey locus above
    k_loop_lab = 0.72 * kmax
    c_branch = np.interp(k_loop_lab, kl[ok], c_hi[ok])
    ax.text(k_loop_lab, c_branch + 0.048 * ymax, '$\\dot c=0$\n(Joy of Saving)', fontsize=FS2, color=INK,
            ha='center', va='center', linespacing=1.1, bbox=LOCUSBOX, zorder=7)
    # flow (joy of saving)
    Kq, Cq = np.meshgrid(np.linspace(0.05 * ss['k'], kmax, 24), np.linspace(0.05, 1.0 * e.f(kmax), 24))
    U_ = np.full_like(Kq, np.nan); V_ = np.full_like(Kq, np.nan)
    for i in range(Kq.shape[0]):
        for j in range(Kq.shape[1]):
            if e.f(Kq[i, j]) - Cq[i, j] > 1e-3:
                d = e.rhs(0, [Kq[i, j], Cq[i, j]]); U_[i, j], V_[i, j] = d
    # STREAM, not ARROW. The two were one constant until the colour of a leader was separated from
    # the colour of the flow; ARROW then became black with the rest of the text, and this call, left
    # on the old name, drew the whole flow field in black over the loci it is meant to sit behind.
    ax.streamplot(Kq, Cq, U_, V_, color=STREAM, density=0.9, linewidth=0.5, arrowsize=0.6, zorder=1)
    _divergent_paths(ax, e, k1, c1, kmax, ymax)
    ax.plot(k0, c0, color='black', lw=1.9, ls='-', zorder=4)
    ax.plot(k1, c1, color=BLUE, lw=2.2, zorder=4)
    _motion_arrows(ax, k0, c0, ss0['k'], kmax, 'black', 1.9)
    _motion_arrows(ax, k1, c1, ss['k'], kmax, BLUE, 2.2)
    ax.set_xlim(0, kmax); ax.set_ylim(0, ymax)
    # saddle-path labels in the right margin, each level with the end of its curve at the right edge (no legend)
    for c_end, lab, col in ((min(c0_end, ymax), r'Ramsey ($\theta=0$)', 'black'),
                            (c1_end, rf'Joy of Saving ($\theta={THETA_PHASE:g}$)', BLUE),
                            (c_kdot_end, r'$\dot k=0$', INK)):
        ax.annotate(lab, (kmax, c_end), xytext=(5, 0), textcoords='offset points', fontsize=FS2, color=col,
                    ha='left', va='center', annotation_clip=False, clip_on=False)
    # steady states: black on the Ramsey path, blue on the joy-of-saving path
    ax.plot(ss0['k'], ss0['c'], 'o', color='black', ms=4, zorder=6)
    ax.plot(ss['k'], ss['c'], 'o', color=BLUE, ms=4, zorder=6)
    # the two benchmark levels of capital, marked on the axis rather than drawn across the panel:
    # a tick below the axis and a label in a second row under the numerals
    blend = ax.get_xaxis_transform()            # x in data units, y in axes fraction
    for kb, lab in ((e.k_gr, 'GR'), (ss0['k'], 'MGR')):
        ax.plot([kb, kb], [0.0, -0.035], transform=blend, color=INK, lw=1.2,
                clip_on=False, zorder=8)
        ax.text(kb, -0.095, lab, transform=blend, ha='center', va='top', fontsize=FS2 - 1,
                color=INK, clip_on=False, zorder=8)
    # the key sits under the axis on the same row as those two labels, out to their right
    ax.text(24.0, -0.095, 'GR: Golden Rule\nMGR: Modified Golden Rule', transform=blend,
            ha='left', va='top', fontsize=FS2 - 1, color=INK, linespacing=1.35,
            clip_on=False, zorder=8)
    ax.set_xlabel('capital, $k$', labelpad=20); ax.set_ylabel('consumption, $c$')
    fig.subplots_adjust(right=0.78, top=0.92, bottom=0.19)   # room for the margin labels and the second tick row
    save(fig, 'fig_phase', OUT)


# ------------------------------------------------------------ Figure 3: transitions from low capital
def fig_transition():
    """Figure 3: transitions from half the modified-golden-rule capital stock, for the Ramsey case and for
    the weights in THETAS_TRANS. Capital and consumption are shown relative to their Ramsey steady-state
    values, so that the three cases are comparable, and the golden-rule stock is marked."""
    e0 = neoclassical.Economy(theta=0.0)
    k0 = 0.5 * e0.k_mgr
    T = 150
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    # The Ramsey case is labelled by its weight, like the other two, so that the three
    # "theta =" labels line up on the page; its name goes on a second line below the curve.
    cases = [(0.0, 'black', '-', r'$\theta=0$')] + \
            [(th, col, '-', rf'$\theta={th:g}$') for th, col in zip(THETAS_TRANS, (AQUA, BLUE))]
    ends = []
    for th, col, ls, lab in cases:
        e = neoclassical.Economy(theta=th)
        tr = e.transition(k0, T=T, n=1501)
        axs[0].plot(tr['t'], tr['k'] / e0.k_mgr, color=col, ls=ls)
        axs[1].plot(tr['t'], tr['c'] / e0.steady_state()['c'], color=col, ls=ls)
        axs[2].plot(tr['t'], 100 * tr['saving_rate'], color=col, ls=ls)
        ends.append((tr['k'][-1] / e0.k_mgr, col, lab))
    # labels above the lines at their right-hand end (no legend); headroom so the top label stays inside the axes
    lo, hi = axs[0].get_ylim(); axs[0].set_ylim(lo, hi + 0.08 * (hi - lo))
    for y, col, lab in ends:
        axs[0].annotate(lab, (T, y), xytext=(0, 3), textcoords='offset points', fontsize=7, color=col, ha='right',
                        va='bottom', bbox=LABELBOX)
        if lab == r'$\theta=0$':
            axs[0].annotate('(Ramsey)', (T, y), xytext=(0, -3), textcoords='offset points', fontsize=7,
                            color=col, ha='right', va='top', bbox=LABELBOX)
    ygr = e0.k_gr / e0.k_mgr
    axs[0].axhline(ygr, color=DARKORANGE, lw=0.8, ls='-')
    axs[0].annotate('Golden Rule', (T, ygr), xytext=(0, 3), textcoords='offset points', fontsize=7, color=DARKORANGE,
                    ha='right', va='bottom', bbox=LABELBOX)
    axs[0].set_ylabel(r'capital, $k/k_{\mathrm{MGR}}$'); axs[1].set_ylabel(r'consumption, $c/c_{\mathrm{MGR}}$'); axs[2].set_ylabel('gross saving rate, %')
    for ax in axs: ax.set_xlabel('years')
    fig.tight_layout()
    save(fig, 'fig_transition', OUT)


if __name__ == '__main__':
    setup()
    os.makedirs(OUT, exist_ok=True)
    fig_linear(); fig_phase(); fig_transition()
    print('figures written to', os.path.abspath(OUT))
