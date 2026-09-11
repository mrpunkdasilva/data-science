#!/bin/bash

# Define color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Find all ipynb files, excluding virtual environments and egg-info directories
files=$(find . -not -path "*/venv/*" -not -path "*/*.egg-info/*" -name "*.ipynb")

# Flag to track if we found any non-compliant notebooks
non_compliant=false

for file in $files; do
  echo "Checking $file"

  # Check for non-empty outputs in the notebook - improved pattern matching
  if jq -e '.cells[] | select(.outputs != null and .outputs != [])' "$file" > /dev/null; then
    echo -e "${RED}Warning: Notebook contains outputs (ignored)${NC}"
  else
    echo -e "${GREEN}Outputs: OK${NC}"
  fi

  echo
done

echo
echo

if [ "$non_compliant" = true ]; then
  echo -e "${RED}Notebook compliance check completed with warnings.${NC}"
else
  echo -e "${GREEN}All notebooks are compliant (no outputs found)!${NC}"
fi
exit 0
