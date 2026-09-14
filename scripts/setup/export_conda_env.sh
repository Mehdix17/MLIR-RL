#!/usr/bin/env bash
# Export the current or named conda environment to environment.yml and a pip requirements file
# Usage:
#   ./scripts/export_conda_env.sh           # uses $CONDA_DEFAULT_ENV if set
#   ./scripts/export_conda_env.sh <envname> # export named environment
set -euo pipefail

ENV_NAME="${1:-}"

if [[ -z "$ENV_NAME" ]]; then
  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    ENV_NAME="$CONDA_DEFAULT_ENV"
  else
    echo "No environment name supplied and CONDA_DEFAULT_ENV is not set. Activate environment or pass name as first arg."
    exit 2
  fi
fi

OUT_DIR="env_export"
mkdir -p "$OUT_DIR"
YML_OUT="$OUT_DIR/environment_${ENV_NAME}.yml"
PIP_OUT="$OUT_DIR/pip_requirements_${ENV_NAME}.txt"
EXPLICIT_OUT="$OUT_DIR/explicit_${ENV_NAME}.txt"

# Export conda environment (no builds to be more portable) and with channels
echo "Exporting conda environment '$ENV_NAME' to $YML_OUT"
conda env export --name "$ENV_NAME" --no-builds > "$YML_OUT"
# Remove the 'prefix:' line to make the YAML more portable
if grep -q "^prefix:" "$YML_OUT"; then
  sed -i '/^prefix:/d' "$YML_OUT"
fi

# Also list explicit specs (exact binaries) if you want exact reproducibility (not portable across platforms)
conda list --name "$ENV_NAME" --explicit > "$EXPLICIT_OUT" || true

# Export pip packages from the environment
# Activate env for deterministic pip freeze
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  python -m pip freeze > "$PIP_OUT"
else
  echo "Warning: CONDA_PREFIX is not set; pip freeze may not point to the desired environment" >&2
  python -m pip freeze > "$PIP_OUT"
fi

echo "Export complete. Files generated:"
ls -lh "$YML_OUT" "$PIP_OUT" "$EXPLICIT_OUT"

echo "Tip: Review $YML_OUT and remove platform-specific or path dependent packages (e.g., local wheel installs)."

echo "Done. If you want to make a portable environment, give these files to your colleague and ask them to run 'scripts/create_conda_env_from_export.sh'."
