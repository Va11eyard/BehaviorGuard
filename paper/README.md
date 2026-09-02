# Paper

`behaviorguard.tex` is the LaTeX source for the arXiv submission (paper title
unchanged; the software package is now TurnShift):

> **BehaviorGuard: Context-Aware Anomaly Detection in Conversational AI via Behavioral User Profiling**
> Dinmukhammed Mynzhassar — Independent Research, February 2026
> arXiv: cs.CR (primary), cs.CL, cs.LG

---

## Compiling

The paper uses a **manual bibliography** (`\thebibliography`), so only `pdflatex` is
needed — no BibTeX or Biber run required.

```bash
cd paper
pdflatex behaviorguard.tex
pdflatex behaviorguard.tex   # second pass resolves cross-references
```

Two passes are sufficient. The output is `behaviorguard.pdf`.

---

## Required LaTeX Packages

All packages are available in any standard TeX distribution
([TeX Live](https://www.tug.org/texlive/) 2022+ or [MiKTeX](https://miktex.org/) 22+):

| Package | Purpose |
|---|---|
| `geometry` | Page margins |
| `times` | Times New Roman font |
| `microtype` | Microtypography |
| `amsmath`, `amssymb` | Math symbols and equations |
| `graphicx` | Figure inclusion |
| `booktabs` | Publication-quality tables |
| `multirow` | Multi-row table cells |
| `xcolor` | Colour definitions |
| `hyperref` | Clickable cross-references and URLs |
| `cleveref` | Smart cross-references (`\Cref`) |
| `algorithm`, `algpseudocode` | Algorithm 1 pseudocode |
| `natbib` | Author-year citation style |
| `caption`, `subcaption` | Figure/table captions |
| `enumitem` | Compact lists |
| `url` | URL formatting |

On Debian/Ubuntu with TeX Live:

```bash
sudo apt install texlive-latex-extra texlive-fonts-recommended
```

---

## Relation to the Codebase

The paper describes exactly the system implemented in `src/turnshift/`.
To reproduce the reported numbers, follow the steps in
[`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) at the repository root.
