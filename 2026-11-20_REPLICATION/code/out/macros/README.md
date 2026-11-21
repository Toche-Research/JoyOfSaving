<!-- Describes what belongs in this directory. -->

# code/out/macros

Generated LaTeX macro files. **This directory is generated.**

Every number quoted in the paper that comes out of a computation is defined here as a macro, so that
the paper never carries a hand-typed figure that can drift from the code. The paper `\input`s these
files and then uses the macros in the text.

To change a number, change the computation and rerun; never edit the macro file.
