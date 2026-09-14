# tools/ — External Tools & Binaries

Third-party tools and utilities for MLIR processing and analysis.

## Directory Structure

```
tools/
├── ast_dumper/           # C++ AST dumper for MLIR
├── pre_vec/              # Pre-vectorization analysis
└── vectorizer/           # Vectorization tools
```

## ast_dumper/

C++ tool that dumps the Abstract Syntax Tree (AST) of MLIR operations. Used by `data_utils/extract/extract_blocks.py` to identify consumer→producer relationships for multi-op block extraction.

**Usage** (via environment variable):
```bash
export AST_DUMPER_BIN_PATH=/path/to/tools/ast_dumper/binary
python -m data_utils.extract.extract_blocks --input model.mlir --output-dir output/
```

**Requirements**:
- Built from LLVM/MLIR source
- Requires matching LLVM version as the project

## pre_vec/

Pre-vectorization analysis tools for understanding loop nest structure before vectorization passes.

## vectorizer/

Vectorization utilities and analysis tools.

## Building Tools

These tools may need to be built from source:

```bash
# Example for ast_dumper
cd tools/ast_dumper
# Follow build instructions specific to the tool
```

## Notes

- Tools are **not tracked by git** (check .gitignore)
- Built binaries may be platform-specific
- Ensure LLVM version compatibility with the main project
