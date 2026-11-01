<!-- Describes what belongs in this directory. -->

# code/scripts

The scripts that produce everything the paper quotes: the figures, the table bodies and the macro
files holding the computed numbers.

They are run from `code/`, not from here, with `PYTHONPATH=.` set, so that the package directory
beside them imports. Each writes into `code/out/`; none writes outside the repository. After running
them, `code/sync_to_paper.sh` copies the results into `tex/`.

The model itself lives in the package directory, not here: these scripts set parameters, call it,
and format the output.
