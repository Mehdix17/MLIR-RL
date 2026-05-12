# Thesis Assistant

**Description:** Assistant for writing the Master's thesis on "Using Deep Reinforcement Learning for Automatic Code Optimization in the MLIR Compiler."

## Project Overview
This is a modular LaTeX document for a Master's thesis.
- Main entry point is [main.tex](main.tex)
- Content files are under `chapters/`
- Bibliography is at [references.bib](references.bib)

## Writing Conventions
- **Language & Tone:** Academic English, precise technical vocabulary (MLIR, LLVM, RL).
- **Bibliography:** Use `biblatex` (`ieee` style). Always cite using `\cite{}`.
- **Acronyms:** Use the `glossaries` package. Never write acronyms as plain text initially. Use `\gls{mlir}`, `\gls{rl}`, `\gls{ppo}`, `\gls{llvm}`, etc. 
- **Equations:** Use standard `amsmath` environments (`equation`, `align`).
- **File Structure Constraint:** Ensure `chapters/` filename mappings align with `\include{}` directives in `main.tex`.

## Commands
- To compile the document, suggest using `pdflatex` followed by `biber` and `pdflatex` again, or `latexmk -pdf main.tex`.