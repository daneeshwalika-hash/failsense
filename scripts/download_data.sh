#!/usr/bin/env bash
# Download the three FailSense datasets into data/raw/.
# Usage: scripts/download_data.sh [--force]
# Idempotent: a dataset is skipped if its marker file already exists.
set -euo pipefail

# Always work relative to the repo root, wherever the script is called from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw"

AI4I_URL="https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
SECOM_URL="https://archive.ics.uci.edu/static/public/179/secom.zip"
CMAPSS_URL="https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"

FORCE=false
[[ "${1:-}" == "--force" ]] && FORCE=true

log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*"; }

require() {
    command -v "$1" >/dev/null 2>&1 || { log "ERROR: '$1' is not installed (sudo apt install $1)"; exit 1; }
}

# download_dataset <name> <url> <marker file that must exist afterwards>
download_dataset() {
    local name="$1" url="$2" marker="$3"
    local target="${RAW_DIR}/${name}"

    if [[ -n "$(find "$target" -name "$marker" -print -quit 2>/dev/null)" && "$FORCE" == false ]]; then
        log "${name}: already present, skipping (use --force to re-download)"
        return 0
    fi

    rm -rf "$target"
    mkdir -p "$target"
    log "${name}: downloading ..."
    curl --fail --location --retry 3 --silent --show-error \
        --output "${target}/${name}.zip" "$url"

    log "${name}: extracting ..."
    unzip -q -o "${target}/${name}.zip" -d "$target"
    rm "${target}/${name}.zip"

    # Some archives contain another zip inside (e.g. NASA C-MAPSS). Extract those too.
    while IFS= read -r -d '' inner; do
        unzip -q -o "$inner" -d "$(dirname "$inner")"
        rm "$inner"
    done < <(find "$target" -name '*.zip' -print0)

    # Flatten: move files out of sub-folders (e.g. "6. Turbofan Engine ... Data Set/")
    # so paths are short and contain no spaces. -n = never overwrite an existing file.
    find "$target" -mindepth 2 -type f -exec mv -n {} "$target"/ \;
    find "$target" -mindepth 1 -type d -empty -delete

    # Data files are read-only data, not programs: rw-r--r-- for all of them.
    find "$target" -type f -exec chmod 644 {} +

    if [[ -z "$(find "$target" -name "$marker" -print -quit)" ]]; then
        log "ERROR: ${name}: expected file '${marker}' not found after extraction"
        exit 1
    fi
    log "${name}: OK -> $(find "$target" -name "$marker" -print -quit)"
}

main() {
    require curl
    require unzip
    mkdir -p "$RAW_DIR"

    download_dataset "ai4i2020" "$AI4I_URL" "ai4i2020.csv"
    download_dataset "secom" "$SECOM_URL" "secom.data"
    download_dataset "cmapss" "$CMAPSS_URL" "train_FD001.txt"

    log "All datasets ready in ${RAW_DIR}"
    du -sh "$RAW_DIR"/*
}

main "$@"
