# Paper (IEEE conference format)

- `main.tex`: the paper, `\documentclass[conference]{IEEEtran}`.
- `IEEEtran.cls`, `IEEEtran.bst`: IEEEtran **V1.8b**, the official IEEE LaTeX class. Downloaded
  2026-10-24 from CTAN (`https://mirrors.ctan.org/macros/latex/contrib/IEEEtran.zip`, sha256
  `e0cd4f5afbd42c8076092280e72b3e0a5111efe501d35de9f715cfb8da313cb4`).
  - ieee.org serves its conference template zip only to browsers: scripts get an empty HTTP 202.
    That zip ships the same IEEEtran class. If the conference asks for its own template, download
    it in a browser from https://www.ieee.org/conferences/publishing/templates.html. Then swap in
    its `.cls` and compare its `conference_101719.tex` preamble with ours.
- `tables/*.tex`: **generated**. Run `python scripts/make_paper_tables.py --seed 0`. Never edit
  numbers by hand.
- `figures/*.pdf`: **generated** by `scripts/make_figures.py` and copied from `results/figures/`.
- `refs.bib`: entries with `note = {VERIFY ...}` must be checked against the originals.

## Compile
No LaTeX is installed on the dev machine. Either:
- **Overleaf** (free): upload the whole `paper/` folder and compile `main.tex` with pdfLaTeX.
- **Local**: install MiKTeX or TeX Live, then run
  `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Before submission (TODO)
- Authors and affiliations (`main.tex`, `\author{...}`).
- Repository URL (Code and Data Availability section).
- Every `VERIFY` entry in `refs.bib`.
- Page limit of the target venue. BIBM regular papers are typically 8 pages; check the call for
  papers.
