#!/bin/sh
# Copy the generated tables, macro files and figures from this repository into the paper's own
# directories, which are not published. Run from code/ after regenerating:
#
#     sh sync_to_paper.sh
#
# The scripts write into out/, so that everything a replicator produces stays inside the
# repository. This step is only for building the PDF of the paper.
#
# Two rules, both learned from this paper.
#
# The list below names every file, one by one, instead of copying out/tables/*.tex and
# out/figures/*.pdf. A wildcard copies whatever happens to be in the directory, and out/tables/
# held three files left over from a renaming, numbers.tex, table_linear.tex and
# table_neoclassical.tex, whose numbers were two revisions behind the ones the paper quotes. A
# wildcard carried them into tex/ at every sync. A file no generator writes is therefore a file
# this script does not copy, and a generator that starts writing a new file is added here by hand.
#
# A file is copied only when the source is newer than the destination. The sync used to copy
# unconditionally, so a run of it after an out/ that had been cleared, or restored from an older
# state, replaced a freshly shipped figure in tex/figures/ with an older one and said nothing. An
# older source is now reported and skipped, and the message says which file and what to do.
set -eu
cd "$(dirname "$0")"

copied=0
skipped=0
missing=0

install_file() {         # $1 source path under out/, $2 destination directory
    src=$1
    dst=$2/$(basename "$1")
    if [ ! -f "$src" ]; then
        echo "  missing  $src (run the script that writes it)"
        missing=$((missing + 1))
        return
    fi
    mkdir -p "$2"
    if [ -f "$dst" ] && [ "$dst" -nt "$src" ]; then
        echo "  SKIPPED  $dst is newer than $src; the copy would go backwards."
        echo "           Regenerate, or remove the destination if the older file is the wanted one."
        skipped=$((skipped + 1))
        return
    fi
    cp "$src" "$dst"
    copied=$((copied + 1))
}

# Table bodies: scripts/make_tables.py
install_file out/tables/tbl_linear.tex        ../tex/tables
install_file out/tables/tbl_neoclassical.tex  ../tex/tables
install_file out/tables/tbl_speed_gamma.tex   ../tex/tables

# Macro files: scripts/make_tables.py writes numbers.tex, scripts/kurz_homothetic.py writes kurz.tex
install_file out/macros/numbers.tex           ../tex/macros
install_file out/macros/kurz.tex              ../tex/macros

# Figures: scripts/figures.py
install_file out/figures/fig_linear.pdf       ../tex/figures
install_file out/figures/fig_phase.pdf        ../tex/figures
install_file out/figures/fig_transition.pdf   ../tex/figures

echo "copied $copied, skipped $skipped, missing $missing"
[ "$skipped" -eq 0 ] || echo "a skipped file means tex/ holds something newer than out/; nothing was overwritten"
