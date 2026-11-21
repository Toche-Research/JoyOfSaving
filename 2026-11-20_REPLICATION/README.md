# The Joy of Saving

Replication package for the paper "The Joy of Saving", by Patrick Toche.

## What is here

| | |
|---|---|
| `code/` | the code, the data it relies on (`code/data/`) and everything it produces (`code/out/`) |
| `working_paper/` | the working paper version of the paper, as a PDF |
| `README.md` | this file |

## The paper and the code

The paper studies an optimal growth model in which utility is defined over consumption and over the flow of
saving, `U(c, s)` with `s = f(k) - c`, and not over consumption alone. In the calibrations utility is isoelastic,
with a weight `theta` on saving, so that `theta = 0` is the Ramsey model and every result can be compared with
that benchmark. Two technologies are solved. With linear technology the model has a balanced growth path and the
growth rate is the root of a single equation. With neoclassical technology, `f(k) = A k^alpha` with depreciation,
the code locates the steady state, verifies that it is a saddle, computes the half-life of the adjustment from the
determinant of the linearized system, and traces the saddle path by backward integration from the steady state.
Further scripts compute the response of consumption to dividends and to capital gains, the limit in which the
weight on saving vanishes (the rule of mental accounting), the comparison with temptation preferences, and the
working paper's section on financial risk.

## Data

The paper analyses no dataset of its own. The only data it uses are estimates published by other authors, which it
quotes and compares with the predictions of the model. They are listed, each with its source and the table, figure
or page where it is found, in `code/data/published_estimates.csv`. The scripts do not read that file: every number,
table and figure the paper computes comes from the parameters set in the scripts.

## Requirements

Python 3.10 or later with numpy, scipy and matplotlib, which are the only third-party imports. `code/requirements.txt`
pins the versions with which the shipped outputs were produced (Python 3.13, numpy 2.5.3, scipy 1.18.1,
matplotlib 3.11.2):

```
pip install -r code/requirements.txt
```

No LaTeX, no compiled extension and no special hardware is needed. The whole package runs in about one minute on
one core.

## How to run it

The scripts import the `joy` package from `code/`, which is not installed, so `code` must be the working directory
and `PYTHONPATH` must point at it.

```
cd code
export PYTHONPATH=.
python3 tests/test_joy.py
python3 tests/test_nonhomothetic.py
for s in make_tables figures checks flow_vs_stock kurz_homothetic nonhomothetic dividend_macros ceq_bound \
         ma_limit ma_temptation ma_macros wp_numbers jos_dividends jos_predictable risk; do
    python3 scripts/$s.py
done
python3 risk/risk.py
python3 risk/evidence.py
```

The tests need no pytest: each file runs its own checks and stops with an `AssertionError` on a failure;
`test_joy.py` ends with `all tests passed`.

## What a run reproduces

Everything the scripts write goes to `code/out/` (and, for the two scripts in `code/risk/`, beside the scripts).
The outputs shipped here are those of the author's run. On 20 November 2026 the package was run again from a clean
copy with the versions above: every `.tex` and `.txt` file under `code/out/` came out byte for byte identical to the
shipped file. The figure PDFs came out with the same size and differ only in the creation date and document
identifier that a PDF file embeds, and `code/risk/risk_output.txt` differs only in its last line, which reports the
running time.

## The scripts

All in `code/scripts/`; each writes under `code/out/`.

| Script | What it computes | Writes |
|---|---|---|
| `make_tables.py` | The balanced growth path at each weight `theta` (linear technology); the steady state, the half-life and the impact saving rate at each weight (neoclassical technology); the thresholds at which capital passes the golden rule and long-run consumption returns to its Ramsey level; a grid of calibrations for the speed of adjustment | `tables/tbl_linear.tex`, `tables/tbl_neoclassical.tex`, their `_article` versions, `tables/tbl_speed_gamma.tex`, `macros/numbers.tex`, `numbers.txt`, `speed_grid.txt` |
| `figures.py` | The figures: growth and the ratio `tau = U_s/U_c` against `theta`; the phase diagram; the transition | `figures/fig_linear.pdf`, `fig_linear_article.pdf`, `fig_phase.pdf`, `fig_transition.pdf` |
| `checks.py` | Accuracy checks: the closed form against a direct maximization of the Bellman equation; the determinant against finite differences; the steady-state condition; the costate equation along the saddle path | `checks.txt` |
| `flow_vs_stock.py` | Saving as a flow against wealth as a stock in the utility function: active saving with and without capital gains; the response of growth to the interest rate | `macros/evidence.tex`, `flow_vs_stock.txt` |
| `dividend_macros.py` | The share of a dividend consumed at a given return; the response to a predictable change in income | `macros/dividends.tex`, `dividends.txt` |
| `kurz_homothetic.py` | Why the examples of Kurz (1968) have several steady states: unequal curvatures, not the stock | `macros/kurz.tex`, `kurz_homothetic.txt` |
| `nonhomothetic.py` | Steady states when utility is not homothetic | `macros/nonhomothetic_numbers.tex`, `nonhomothetic.txt` |
| `ceq_bound.py` | The bound on the consumption-equivalent gain quoted in the note to the table for neoclassical technology | `macros/ceq_bound.tex` |
| `ma_limit.py` | The limit of the plan as `theta` tends to zero: convergence to the rule of mental accounting, with rates | `ma/ma_limit.txt` |
| `ma_temptation.py` | Temptation preferences with a mental account (Gul and Pesendorfer, 2004) against the joy of saving; uses `ma_hjb.py` and `ma_common.py` | `ma/ma_temptation.txt` |
| `ma_macros.py` | The numbers quoted on mental accounting | `ma/ma_macros.tex`, `ma/ma_macros.txt` |
| `wp_numbers.py` | Numbers, tables and a figure of the working paper: the share of a dividend consumed over a grid of calibrations | `macros/wp_numbers.tex`, `tables/tbl_div_calib.tex`, `tables/tbl_div_seven.tex`, `figures/fig_dividend_share.pdf`, `wp_numbers.txt` |
| `jos_dividends.py`, `jos_predictable.py` | Independent checks of the dividend and predictable-income formulas; they print to the screen | nothing |
| `risk.py` | An exploratory computation with a risky return, not used in the paper | `risk.txt` |

`code/risk/risk.py` and `code/risk/evidence.py` produce the table, figure and numbers of the working paper's section
on financial risk and write them beside the scripts, with logs in `risk_output.txt` and `evidence_output.txt`.

## How the numbers reach the text

Numbers are not typed into the paper. Each script formats the quantities the text quotes and writes them as LaTeX
macros, one per line, for example

```
\newcommand{\neoThetaGR}{0.135}
```

and the paper refers to each number by name. The files in `code/out/macros/` and `code/out/tables/` are therefore the
numbers and the table bodies printed in the paper, and the PDFs in `code/out/figures/` are its figures.

## Layout of `code/`

```
joy/          the model: linear and neoclassical technology, the stock model, the figure style
scripts/      the scripts listed above
risk/         the working paper's section on financial risk
tests/        the test suite
data/         the published estimates the paper quotes
out/          everything the scripts write: tables/, macros/, figures/, ma/ and text reports
```

## Licences

The code is released under the MIT Licence (`code/LICENSE`) and the generated output under CC BY 4.0;
`code/LICENSES.md` gives the details. The published estimates in `code/data/` belong to the publications cited.
