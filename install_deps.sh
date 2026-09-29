#!/usr/bin/env bash
# Create (or update) the needlework conda environment and everything it needs.
#
#   export NEEDLEWORK_ROOT=/path/with/space/for/data/and/runs
#   ./install_deps.sh --dinov3-weights /path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
#
# Steps: conda env with Python only -> uv pip sync of the exported lock (plus this
# package, editable) -> pinned DINOv3 source and verified weights under
# $NEEDLEWORK_ROOT/cache/dinov3 -> pre-commit hooks -> import check.
# Safe to re-run; each step skips work that is already done and correct.
#
# Pins are read from their one definition: Python from .python-version, uv from the
# dev group in pyproject.toml, DINOv3 commit and weights from needlework/constants.py.
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Run this script, do not source it: ./install_deps.sh" >&2
    return 1
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ENV_NAME="needlework"
PYTHON_VERSION="$(< "${ROOT_DIR}/.python-version")"
UV_VERSION="$(sed -n 's/^ *"uv==\([0-9.]*\)".*/\1/p' "${ROOT_DIR}/pyproject.toml")"
DINOV3_REPO_URL="https://github.com/facebookresearch/dinov3.git"
DINOV3_WEIGHTS_SRC=""

usage() {
    cat <<EOF
Usage: ./install_deps.sh [--env-name NAME] [--dinov3-weights PATH]

  --env-name NAME        Conda environment to create or sync (default: ${ENV_NAME}).
  --dinov3-weights PATH  The DINOv3 ViT-B/16 weights (file name and sha256 in
                         src/needlework/constants.py), downloaded after accepting the
                         DINOv3 license (https://ai.meta.com/dinov3/). Copied to
                         \$NEEDLEWORK_ROOT/cache/dinov3/ after its sha256 is checked.
                         Needed once; later runs reuse the verified copy.

Requires NEEDLEWORK_ROOT to be set (see set_env.sh).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name) ENV_NAME="${2:?--env-name requires a value}"; shift 2 ;;
        --dinov3-weights) DINOV3_WEIGHTS_SRC="${2:?--dinov3-weights requires a value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

step() { echo; echo "==> $*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -n "${NEEDLEWORK_ROOT:-}" ]] || fail "NEEDLEWORK_ROOT is not set. Export it first (see set_env.sh)."
[[ -n "${PYTHON_VERSION}" ]] || fail "could not read the Python version from .python-version"
[[ -n "${UV_VERSION}" ]] || fail "could not read the uv pin from pyproject.toml"
[[ "${NEEDLEWORK_ROOT}" = /* ]] || fail "NEEDLEWORK_ROOT must be an absolute path: ${NEEDLEWORK_ROOT}"
DINOV3_DIR="${NEEDLEWORK_ROOT}/cache/dinov3"

if command -v mamba >/dev/null 2>&1; then CONDA=mamba
elif command -v conda >/dev/null 2>&1; then CONDA=conda
else fail "conda or mamba is required (https://github.com/conda-forge/miniforge)."
fi

step "Conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})"
if "${CONDA}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Exists; reusing it."
else
    "${CONDA}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi
ENV_PREFIX="$("${CONDA}" run -n "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"
ENV_PY="${ENV_PREFIX}/bin/python"

step "Locked Python dependencies (uv ${UV_VERSION})"
"${ENV_PY}" -m pip install --quiet "uv==${UV_VERSION}"
# Conda only provides Python; every package comes from uv.lock. `uv pip sync` makes the
# env match the export exactly (removing anything else) and installs this package in
# editable mode, so no PYTHONPATH is needed.
(cd "${ROOT_DIR}" && "${ENV_PY}" -m uv export --locked --quiet \
    --preview-features extra-build-dependencies \
    | "${ENV_PY}" -m uv pip sync --python "${ENV_PY}" \
        --preview-features extra-build-dependencies -)

# Only now is needlework importable: read the DINOv3 pins from its constants.
read -r DINOV3_COMMIT DINOV3_WEIGHTS_NAME DINOV3_WEIGHTS_SHA256 < <("${ENV_PY}" -c \
    'from needlework import constants as c; print(c.DINOV3_SOURCE_COMMIT, c.DINOV3_WEIGHTS_FILE, c.DINOV3_WEIGHTS_SHA256)')

step "DINOv3 source @ ${DINOV3_COMMIT:0:7}"
if [[ ! -d "${DINOV3_DIR}/src/.git" ]]; then
    mkdir -p "${DINOV3_DIR}"
    git clone --quiet --filter=blob:none "${DINOV3_REPO_URL}" "${DINOV3_DIR}/src"
fi
git -C "${DINOV3_DIR}/src" checkout --quiet "${DINOV3_COMMIT}" \
    || fail "Could not check out ${DINOV3_COMMIT} in ${DINOV3_DIR}/src (local changes present)."
[[ "$(git -C "${DINOV3_DIR}/src" rev-parse HEAD)" == "${DINOV3_COMMIT}" ]] \
    || fail "${DINOV3_DIR}/src is not at ${DINOV3_COMMIT}."

step "DINOv3 ViT-B/16 weights"
DINOV3_WEIGHTS="${DINOV3_DIR}/${DINOV3_WEIGHTS_NAME}"
if [[ -n "${DINOV3_WEIGHTS_SRC}" ]]; then
    [[ -f "${DINOV3_WEIGHTS_SRC}" ]] || fail "No such file: ${DINOV3_WEIGHTS_SRC}"
    cp "${DINOV3_WEIGHTS_SRC}" "${DINOV3_WEIGHTS}.tmp"
    mv "${DINOV3_WEIGHTS}.tmp" "${DINOV3_WEIGHTS}"
fi
if [[ -f "${DINOV3_WEIGHTS}" ]]; then
    echo "${DINOV3_WEIGHTS_SHA256}  ${DINOV3_WEIGHTS}" | sha256sum --check --quiet \
        || fail "sha256 mismatch for ${DINOV3_WEIGHTS}; expected ${DINOV3_WEIGHTS_SHA256}."
    echo "Verified ${DINOV3_WEIGHTS}"
else
    echo "Not installed. Image-encoder steps will fail until you re-run with"
    echo "  --dinov3-weights /path/to/${DINOV3_WEIGHTS_NAME}"
fi

step "pre-commit hooks"
if [[ -e "${ROOT_DIR}/.git" ]]; then
    (cd "${ROOT_DIR}" && "${ENV_PREFIX}/bin/pre-commit" install)
else
    echo "Skipped: ${ROOT_DIR} is not the top of a git checkout."
fi

step "Import check"
"${ENV_PY}" - "${DINOV3_DIR}/src" <<'PY'
import sys

import mujoco
import robomimic.envs.env_robosuite  # noqa: F401
import robosuite
import torch

import needlework

from needlework.constants import DINOV3_MODEL

torch.hub.load(sys.argv[1], DINOV3_MODEL, source="local", pretrained=False)
print(f"needlework {needlework.__version__}, torch {torch.__version__}, "
      f"robosuite {robosuite.__version__}, mujoco {mujoco.__version__}")
print(f"CUDA available: {torch.cuda.is_available()} (training requires a CUDA GPU)")
PY

echo
echo "Done. Start a session with:"
echo "  conda activate ${ENV_NAME} && source ${ROOT_DIR}/set_env.sh"
