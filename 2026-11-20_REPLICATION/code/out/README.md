<!-- Describes what belongs in this directory. -->

# code/out

Everything the code produces. **This whole tree is generated** and is reproducible by rerunning the
scripts; nothing in it should be edited by hand.

| subdirectory | what it holds |
|---|---|
| `figures/` | the figures of the paper, as PDF |
| `tables/` | the bodies of the paper's tables, as LaTeX rows |
| `macros/` | the computed numbers the paper quotes, as LaTeX macros |

The text files directly under `out/` are the reports the scripts print. The paper's LaTeX source, which is not
part of the repository, takes `tables/`, `figures/` and `macros/` as inputs.

The outputs are committed deliberately, so that they can be compared with a fresh run.
