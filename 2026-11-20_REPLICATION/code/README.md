<!-- Describes what belongs in this directory. -->

# code

The replication package: the code that produces every number, table and figure the paper quotes, and
the outputs it writes.

| | |
|---|---|
| the package directory | the model itself, imported by the scripts; writes nothing |
| `scripts/` | the producers, run from here with `PYTHONPATH=.` set |
| `tests/` | checks against closed forms and limiting cases |
| `out/` | everything the scripts write; generated, and reproducible |
| `data/` | the published estimates the paper quotes, with their sources (`published_estimates.csv`); the paper analyses no dataset of its own, and the scripts do not read this file |

Run the scripts from this directory, not from `scripts/`. Nothing writes outside `out/`.
