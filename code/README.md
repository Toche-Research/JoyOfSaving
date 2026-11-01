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
| `sync_to_paper.sh` | copies `out/` into `tex/`; the only thing that should write there |

Run the scripts from this directory, not from `scripts/`. Nothing writes outside the repository.
After a run, `sync_to_paper.sh` puts the results where the paper will find them.
