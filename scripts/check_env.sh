#!/usr/bin/env bash
# Check that all tools needed for FailSense are installed in WSL2.
# Usage: scripts/check_env.sh
set -euo pipefail

ok=0
fail=0

check() {
    # check <label> <command...>
    local label="$1"; shift
    local out
    if out="$("$@" 2>&1)"; then
        printf '  [OK]   %-22s %s\n' "$label" "$(head -n1 <<<"$out")"
        ok=$((ok + 1))
    else
        printf '  [FAIL] %-22s %s\n' "$label" "$(head -n1 <<<"$out")"
        fail=$((fail + 1))
    fi
}

echo "FailSense environment check"
check "git" git --version
check "git user.name" git config --global user.name
check "git user.email" git config --global user.email
check "uv" uv --version
check "python 3.12 (uv)" uv python find 3.12
check "curl" curl --version
check "unzip" unzip -v
check "docker" docker --version
check "docker daemon" docker info --format '{{.ServerVersion}}'
check "docker compose" docker compose version
check "github ssh" bash -c 'ssh -T -o BatchMode=yes git@github.com 2>&1 | grep -q "successfully authenticated" && echo "authenticated"'
check "vscode (code)" code --version

echo "Result: ${ok} OK, ${fail} FAIL"
[[ "$fail" -eq 0 ]]
