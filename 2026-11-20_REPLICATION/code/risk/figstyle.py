"""Common matplotlib style for the figures of the DynamicPlans papers.

This file is a COPY of Styles/figstyle.py, installed into each paper so that its code directory
is self-contained and can be published on its own. Edit the master and run Styles/sync_styles.sh
to install it again, or edit this copy and accept that the paper has deliberately diverged;
sync_styles.sh reports which papers have.

Colour. All text is black and grey is for furniture only, so that nothing a reader must read is
set in grey. INK carries text and data, INK2 the axis frame, reference lines and streamlines, and
GRID the grid. BLUE, ORANGE and AQUA are the three categorical slots, and a colour means the same
thing in every figure of a paper.

A paper that needs a colour outside this set defines it in its own copy, beside this palette, with
a comment saying what it is for.
"""
import os
import sys

import matplotlib

# The backend must be chosen before pyplot is imported, so this module has to be imported before
# it. Importing it afterwards would leave the figure on whatever backend pyplot already chose and
# the paper would get a figure that looks subtly wrong rather than an error. That is exactly how
# thirty-five scripts in this project came to set their own backend and bypass this file for
# months, so the condition is checked rather than trusted.
if 'matplotlib.pyplot' in sys.modules:
    raise RuntimeError(
        'import this style module before matplotlib.pyplot: the pgf backend must be selected '
        'first, and pyplot has already been imported')
matplotlib.use('pgf')
import matplotlib.pyplot as plt

# The figures are typeset by LaTeX itself, through matplotlib's pgf backend, rather than by
# matplotlib's own mathtext. Mathtext is a reimplementation of TeX's maths layout and only
# approximates it: superscripts come out at the wrong optical size and the space around relations
# and around Greek letters is not TeX's. Worse, `font.family='serif'` does not mean the document's
# serif, it means matplotlib's fallback, DejaVu Serif, so every figure in this project used to set
# its prose in one typeface and its mathematics in another, and neither matched the paper. Handing
# the labels to LaTeX removes all three problems at once and makes the figures inherit whatever
# the document sets, so a later change of font needs no change here.
#
# pdflatex rather than lualatex: the papers load no font package, so both engines give Computer
# Modern and the result is identical, while pdflatex needs no luaotfload and therefore runs on any
# machine a replicator might use. Change this one line if the papers ever move to a Unicode font.
TEXSYSTEM = 'pdflatex'

# The size scale. One number sets everything, and the ratios between the sizes are fixed here rather
# than in each script, so the hierarchy is guaranteed: a panel title is the largest text in a figure,
# an axis label is next, tick labels and legends sit below that, and text placed inside the axes is
# the smallest, because it shares the space with the curves.
#
# A script must not write a bare number for a size. matplotlib applies `axes.titlesize` only when
# `set_title` is called WITHOUT `fontsize=`, and it has no rcParam at all for `annotate`, whose text
# falls back to `font.size`. A literal in either place is invisible to this file, and changing BASE
# then appears to do nothing: that is exactly how the panel titles came to be pinned at 8 while the
# axis labels moved to 10, which is the wrong way round.
BASE = 10          # an axis label
TITLE = BASE + 1   # a panel title: (a), (b), ... deliberately the largest text in the figure
TICK = BASE - 1    # tick labels, and the legend
ANNOT = BASE - 2   # annotations and other text placed inside the axes

# All text is pure black. Grey is for furniture only: the axis frame (INK2) and the grid (GRID).
INK, INK2, GRID = '#000000', '#52514e', '#d9d8d4'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'

# Promoted here because three papers had declared them identically in their own copies, with the same
# comment, which is the test for whether something belongs in the master.
DARKORANGE = '#b35900'   # a line and its label where ORANGE would not be legible on white
BENCH = INK2             # a benchmark line, drawn dashed: the same grey as the frame
# Two kinds of arrow, which are not the same object and must not be drawn alike.
#
# LEADER joins a label to the thing it names. It carries no head: a label and a curve cannot be
# mistaken for one another, so a head adds ink without adding information. Felicity drew its leaders
# this way all along.
LEADER = dict(arrowstyle='-', color=INK, lw=0.8, shrinkA=1, shrinkB=2)

# MOTION is an arrowhead ON a path, saying which way the path runs. There the head IS the
# information, so it keeps one. Two things matter in how it is drawn. It takes `lw=0`, so that only
# the head appears and the path keeps its own weight and colour: an arrow drawn as a separate
# segment over a line appears as a second line of a different weight, which is what made the
# direction arrows in Health's phase diagram sit badly on the axis they belonged to. And
# `mutation_scale=1` makes head_length and head_width plain points, matched to the Stealth tips the
# TikZ figures use, so the two kinds of figure draw the same arrowhead. The colour is the caller's:
# an arrowhead on a curve takes that curve's colour.
MOTION = dict(arrowstyle='-|>, head_length=3.6, head_width=1.5', lw=0, mutation_scale=1)

ARROW = INK              # the colour of a leader, kept as a name because scripts refer to it

# Ordered families are not categories, so they do not take a slot above. A series whose index is a
# magnitude rather than an identity is coloured along this ramp, light to dark, so that the reader
# reads the order off the figure instead of a legend.
SEQ = 'viridis'

# A white box behind an annotation, so that grid lines and arrows do not cross the text.
LABELBOX = dict(facecolor='white', edgecolor='none', pad=1.2, alpha=1.0)


def setup(**overrides):
    """Apply the defaults. Called once before any figure is drawn.

    In matplotlib `xtick.color` and `ytick.color` colour the tick *labels* as well as the tick
    marks, so they are set to INK rather than to a grey: a grey there is the usual cause of axis
    numbers that look washed out. `axes.titlecolor` is set for the same reason.

    Keyword arguments are passed through to rcParams, for a paper that needs one setting changed
    without diverging from the rest of this file.
    """
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': BASE, 'axes.labelsize': BASE, 'axes.titlesize': TITLE,
        'legend.fontsize': TICK, 'xtick.labelsize': TICK, 'ytick.labelsize': TICK,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
        'axes.titlecolor': INK, 'text.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        # How a grid looks, not whether there is one: `axes.grid` is deliberately absent.
        # Whether a figure carries a grid is the figure's decision, made with ax.grid(),
        # as twelve scripts in this project already do; the master only fixes its colour
        # and weight so that every grid that does appear looks the same. The same rule
        # separates legend.frameon and legend.fontsize, which are here, from the call to
        # ax.legend(), which is not: the master says how a thing looks when it appears,
        # never whether it appears. Axis limits and labels are figure-specific for the
        # same reason. The top and right spines are the exception that proves the rule:
        # they are off in every figure in the project and never vary, so they are a house
        # style rather than a per-figure choice.
        'grid.color': GRID, 'grid.linewidth': 0.5,
        'lines.linewidth': 1.6, 'legend.frameon': False, 'savefig.bbox': 'tight',
        'figure.dpi': 150,
        # LaTeX owns the fonts: rcfonts off, and the preamble mirrors what the papers load.
        'pgf.texsystem': TEXSYSTEM, 'pgf.rcfonts': False,
        'pgf.preamble': '\n'.join([r'\usepackage{amsmath}', r'\usepackage{amssymb}'])})
    if overrides:
        plt.rcParams.update(overrides)
    # A silent fall back to another backend would mean matplotlib, not LaTeX, setting the labels.
    assert matplotlib.get_backend().lower() == 'pgf', (
        'the pgf backend did not take: matplotlib is using %r' % matplotlib.get_backend())


def save(fig, name, outdir, png=False):
    """Write the figure to outdir as <name>.pdf and close it.

    `bbox_inches="tight"` is passed at the call site as well as through rcParams, because without
    it a long axis label or a two-line title is silently clipped at the figure boundary.

    The paper uses the PDF. A PNG is written only when asked for, because PNG files are drafts and
    are not published: they have carried generator provenance metadata in their headers, which PDFs
    have not.
    """
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + '.pdf'), bbox_inches='tight')
    if png:
        fig.savefig(os.path.join(outdir, name + '.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# Additions specific to this paper. Everything above this line is the master,
# Styles/figstyle.py, and sync_styles.sh compares only that part.
# ---------------------------------------------------------------------------
STREAM = '0.72'             # the streamlines of the phase diagram, lighter than the loci over them.
                            # Named ARROW until it was noticed that it colours no arrow: the leader
                            # arrows are ARROW, which the master sets to black.
