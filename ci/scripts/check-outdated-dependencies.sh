#!/bin/bash

# Fails when a directly declared dependency is outdated and is not explicitly
# allowed in .allowed_outdated_dependencies.
#
# Only packages declared in pyproject.toml (project.dependencies) and
# requirements-dev.txt are checked. Transitive dependencies (nvidia-*, triton,
# ipython, ...) are skipped on purpose: their versions are a consequence of our
# own pins, so flagging them would be noise we cannot act on.

set -uo pipefail

# Define color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

ALLOWLIST_FILE=".allowed_outdated_dependencies"

# PEP 503 normalization: lowercase, and runs of -_. collapse into a single -
normalize() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[-_.]+/-/g'
}

# Collect the normalized names of every directly declared dependency
declared=$(python3 - <<'PY'
import re
import tomllib

names = set()


def add(name: str) -> None:
    name = re.split(r"[<>=!~\[; ]", name.strip(), maxsplit=1)[0]
    if name:
        names.add(re.sub(r"[-_.]+", "-", name.lower()))


with open("pyproject.toml", "rb") as handle:
    for dep in tomllib.load(handle).get("project", {}).get("dependencies", []):
        add(dep)

with open("requirements-dev.txt", encoding="utf-8") as handle:
    for line in handle:
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            add(line)

print("\n".join(sorted(names)))
PY
)

if [ -z "$declared" ]; then
  echo -e "${RED}Could not read the declared dependencies.${NC}"
  exit 1
fi

# Load the allowlist patterns, if any
allowed=""
if [ -f "$ALLOWLIST_FILE" ]; then
  allowed=$(grep -v '^[[:space:]]*#' "$ALLOWLIST_FILE" | grep -v '^[[:space:]]*$' || true)
fi

# Flag to track if we found any non-allowed outdated packages
non_compliant=false
checked=0

while read -r package rest; do
  normalized=$(normalize "$package")

  # Not a direct dependency: transitive, so its version is not ours to choose
  if ! grep -qxF "$normalized" <<<"$declared"; then
    continue
  fi

  checked=$((checked + 1))

  # Explicitly allowed to be outdated
  if grep -qE "^($(sed -E 's/[-_.]+/[-_.]/g; s/\*/.*/g' <<<"$allowed" | paste -sd'|' -))$" <<<"$normalized"; then
    echo -e "${YELLOW}Allowed outdated: $normalized ($rest)${NC}"
    continue
  fi

  echo -e "${RED}Outdated dependency: $normalized ($rest)${NC}"
  non_compliant=true
done < <(pip list --outdated 2>/dev/null | tail -n +3)

if [ "$checked" -eq 0 ]; then
  echo -e "${RED}No declared dependency was inspected, check the environment.${NC}"
  exit 1
fi

echo
echo -e "Checked $checked declared dependencies."

if [ "$non_compliant" = true ]; then
  echo -e "${RED}Found non-allowed outdated packages.${NC}"
  echo "Either bump the pin, or add the package to $ALLOWLIST_FILE."
  exit 1
fi

echo -e "${GREEN}All declared dependencies are up to date!${NC}"
exit 0
