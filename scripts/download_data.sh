#!/usr/bin/env bash
# Prepare the FailSense datasets in data/raw/.
# Usage: scripts/download_data.sh [--force]
#
#   cmapss : downloaded automatically (NASA C-MAPSS, ~44 MB)
#   scania : manual download (SCANIA Component X, ~1.5 GB), then this script
#            extracts, flattens and verifies the files
#
# Idempotent: a dataset is skipped if its marker file already exists.
# Bosch CNC Machining (Should item) is added on 09.11.
set -euo pipefail

# Always work relative to the repo root, wherever the script is called from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw"

CMAPSS_URL="https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"

SCANIA_PAGE="https://researchdata.se/en/catalogue/dataset/2024-34"
SCANIA_FILES=(
    train_operational_readouts.csv
    train_specifications.csv
    train_tte.csv
    validation_operational_readouts.csv
    validation_specifications.csv
    validation_labels.csv
    test_operational_readouts.csv
    test_specifications.csv
    test_labels.csv
)
# Fallback only: SHA-256 of the largest file as published on the SND catalogue page (version 3).
# Normally all files are checked against the sha256.txt that ships with the archive.
SCANIA_TRAIN_SHA256="e01cb0bd87dfab4c9dbad215d51d81282fc0d413be96d6f819ea872fb7a3c715"

FORCE=false
[[ "${1:-}" == "--force" ]] && FORCE=true

log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*"; }

require() {
    command -v "$1" >/dev/null 2>&1 || { log "ERROR: '$1' is not installed (sudo apt install $1)"; exit 1; }
}

# extract_and_flatten <target dir>
# Unzip every archive in <target dir> (also archives inside archives), move all
# files to the top level (no sub-folders, no spaces in paths) and make them read-only data.
extract_and_flatten() {
    local target="$1" archive

    while [[ -n "$(find "$target" -name '*.zip' -print -quit)" ]]; do
        archive="$(find "$target" -name '*.zip' -print -quit)"
        unzip -q -o "$archive" -d "$(dirname "$archive")"
        rm "$archive"
    done

    # -n = never overwrite an existing file.
    find "$target" -mindepth 2 -type f -exec mv -n {} "$target"/ \;
    find "$target" -mindepth 1 -type d -empty -delete

    # Data files are read-only data, not programs: rw-r--r-- for all of them.
    find "$target" -type f -exec chmod 644 {} +
}

# download_dataset <name> <url> <marker file that must exist afterwards>
download_dataset() {
    local name="$1" url="$2" marker="$3"
    local target="${RAW_DIR}/${name}"

    if [[ -f "${target}/${marker}" && "$FORCE" == false ]]; then
        log "${name}: already present, skipping (use --force to re-download)"
        return 0
    fi

    rm -rf "$target"
    mkdir -p "$target"
    log "${name}: downloading ..."
    curl --fail --location --retry 3 --silent --show-error \
        --output "${target}/${name}.zip" "$url"

    log "${name}: extracting ..."
    extract_and_flatten "$target"

    if [[ ! -f "${target}/${marker}" ]]; then
        log "ERROR: ${name}: expected file '${marker}' not found after extraction"
        exit 1
    fi
    log "${name}: OK -> ${target}/${marker}"
}

# verify_scania_checksums <target dir>
# The archive ships sha256.txt with the original paths inside the zip
# (e.g. "2024-34-3/data/train_tte.csv"). The files were flattened, so sed keeps
# only the file name before sha256sum checks every file.
verify_scania_checksums() {
    local target="$1"

    if [[ -f "${target}/sha256.txt" ]]; then
        log "scania: verifying SHA-256 of all files listed in sha256.txt ..."
        if (cd "$target" && sed -E 's#^([0-9a-f]+) .*/#\1  #' sha256.txt | sha256sum --check --quiet -); then
            log "scania: all checksums OK"
        else
            log "ERROR: scania: checksum mismatch - delete ${target} and download the zip again"
            exit 1
        fi
    else
        # Fallback when the files were downloaded one by one (no sha256.txt).
        log "scania: no sha256.txt found, checking train_operational_readouts.csv only ..."
        if [[ "$(sha256sum "${target}/train_operational_readouts.csv" | cut -d ' ' -f 1)" == "$SCANIA_TRAIN_SHA256" ]]; then
            log "scania: checksum OK (version 3)"
        else
            log "WARNING: scania: checksum differs - probably another dataset version; note the version in the README"
        fi
    fi
}

# prepare_scania: the SND portal has no stable direct download link, so the
# archive is downloaded by hand once. This function extracts and verifies it.
prepare_scania() {
    local target="${RAW_DIR}/scania" file missing=()

    mkdir -p "$target"

    if [[ -n "$(find "$target" -name '*.zip' -print -quit)" ]]; then
        log "scania: extracting archive(s) ..."
        extract_and_flatten "$target"
    fi

    for file in "${SCANIA_FILES[@]}"; do
        [[ -f "${target}/${file}" ]] || missing+=("$file")
    done

    if (( ${#missing[@]} > 0 )); then
        log "scania: ${#missing[@]} of ${#SCANIA_FILES[@]} files missing: ${missing[*]}"
        log "scania: manual download needed (~1.5 GB):"
        log "  1. Open ${SCANIA_PAGE}"
        log "  2. Download all files (zip) and copy the zip into ${target}/"
        log "  3. Run 'make data' again"
        exit 1
    fi

    verify_scania_checksums "$target"
    log "scania: OK -> ${target} (${#SCANIA_FILES[@]} files)"
}

main() {
    require curl
    require unzip
    require sha256sum
    mkdir -p "$RAW_DIR"

    download_dataset "cmapss" "$CMAPSS_URL" "train_FD001.txt"
    prepare_scania

    log "All datasets ready in ${RAW_DIR}"
    du -sh "$RAW_DIR"/*/
}

main "$@"
