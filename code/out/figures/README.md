<!-- Describes what belongs in this directory. -->

# code/out/figures

Where the figure scripts write their output. **This directory is generated.** Everything in it is
reproducible from the code, and nothing in it should be edited by hand.

Produced by `figures.py`, run from `code/` with `PYTHONPATH=.` set.

Copied from here into `tex/figures/` by `code/sync_to_paper.sh`, which is the only thing that should
ever write into `tex/figures/`. The paper then includes them through `\graphicspath{{figures/}}`.

The PDFs are committed deliberately, so that the paper compiles without anyone having to run the
code first. That also means they can fall out of date: whenever a figure script or the style module
changes, rerun the script and the sync before rebuilding the paper.
