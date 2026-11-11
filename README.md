# VJCS multi-cancer classification research draft

## Contents

- `main.tex` — complete **working research draft**, including study rationale, literature, architecture, equations, protocol, template results/ablation tables, discussion, limitations, declarations, and references.
- `references.bib` — BibTeX references for migration into the official template.
- `references_manual.tex` — bibliography automatically included in the working PDF build, so only pdfLaTeX is required.
- `paper_preview.pdf` — compiled reading preview (if present).
- `AUDIT_AND_SUBMISSION_PLAN.md` — notebook-specific verification and manuscript finishing tasks.

**Important:** This draft is **not submission-ready**. The supplied notebook saved an installation error and no completed evaluation metrics. No performance numbers were invented. Red `[TO VERIFY: ...]` and `PENDING` items must be replaced with measured evidence, and the paper must be recompiled using the official World Scientific VJCS template before submission.

## Quick start in VS Code on Windows 11

1. Install VS Code and its `LaTeX Workshop` extension (by James Yu).
2. Install **MiKTeX** (Windows-friendly) or **TeX Live** (more complete). For MiKTeX, allow missing packages to install when prompted.
3. Unzip this project to a local folder and open the folder in VS Code.
4. Open `main.tex`; invoke **LaTeX Workshop: Build LaTeX project** from the command palette. Configure the extension to use `latexmk` or run from the VS Code terminal:

   ```powershell
   latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
   ```

   This command runs pdfLaTeX; the working draft includes `references_manual.tex` so BibTeX is not required. Keep all project files together. The `.bib` file is for future journal-template migration.
5. View `main.pdf` inside VS Code using LaTeX Workshop's PDF preview.
6. After edits, run `latexmk -C` to remove intermediate build files if desired.

### Without a full local installation

Upload `main.tex` and `references.bib` as the same Overleaf project. Set the compiler to **pdfLaTeX**. Overleaf's free-plan compilation and project limits can change. If your file exceeds an online compilation limit, local VS Code + TeX Live/MiKTeX is not constrained by Overleaf's online compile-time quota (but uses your own disk/RAM/CPU).

## Journal-specific formatting

This archive uses the standard LaTeX `article` class for a portable, readable draft; it **does not claim** to be the official VJCS manuscript style. Get the current World Scientific VJCS LaTeX package and author guidelines from the journal site prior to final submission, then migrate section bodies, citations, tables, and figures into that template. Confirm the current abstract word limit, keyword limit, citation style, ORCID rules, cover letter, and AI-use disclosure requirements. These details can change.

## Citation and drafting ethics

Use the actual authors' names, affiliations, confirmed results, and disclosures. Verify the bibliography and all cited claims against the original sources. AI-assisted text must be critically reviewed by human authors and disclosed as required by the current journal policy. Do not submit placeholders or claim clinical diagnosis from dataset-class classification.
