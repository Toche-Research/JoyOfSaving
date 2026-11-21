<!-- Describes what belongs in this directory. -->

# data

The paper analyses no dataset of its own. The only data it uses are estimates published by other authors, which
it quotes and sets against the predictions of the model. They are all listed in one file:

| | |
|---|---|
| `published_estimates.csv` | every published estimate quoted in the article or the working paper: source, where it is found in the source (table, figure or page), what it measures, the value as quoted, the unit, the sample, and whether the article or only the working paper quotes it |

The values were transcribed by hand from the cited publications and are reproduced as the paper quotes them
(ranges are entered as a low and a high row; values the paper gives as approximate are marked so in the unit
column). The scripts do not read this file: the numbers the paper computes come from the model's own parameters
and are written to `../out/`. The file is here so that every number in the paper that is not computed by the
code can be traced to its source in one place.
