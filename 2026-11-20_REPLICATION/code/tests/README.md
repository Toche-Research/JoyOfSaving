<!-- Describes what belongs in this directory. -->

# code/tests

The test suite. It checks the model code against closed forms, limiting cases and independently
computed benchmarks, so that a change to the model that breaks a published number fails here first.

Run from `code/` with `PYTHONPATH=.` set. The tests need no data and no network.
