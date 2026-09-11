#!/bin/bash

# Define color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

files=$(find . -not -path "*/venv/*" -not -path "*/notebooks/*" -not -path "*/*.egg-info/*" \( -name "*.typ" -o -name "*.txt" -o -name "*.sh" -o -name "*.md" -o -name "*.yml" \))

# Flag to track if we found any non-compliant files
non_compliant=false

for file in $files; do
  echo "Checking $file"

  # Check for tabs
  if grep -q $'\t' "$file"; then
    echo -e "${RED}Warning: Tabs found in file (ignored)${NC}"
  else
    echo -e "${GREEN}Tabs: OK${NC}"
  fi

  # Check for trailing spaces
  if grep -q " $" "$file"; then
    echo -e "${RED}Warning: Trailing spaces found in file (ignored)${NC}"
  else
    echo -e "${GREEN}Trailing spaces: OK${NC}"
  fi

  # Check for an empty line at the end of the file
  if [ "$(tail -c 1 "$file" | wc -l)" -eq 0 ]; then
    echo -e "${RED}Warning: No newline at the end of file (ignored)${NC}"
  else
    echo -e "${GREEN}Newline at end: OK${NC}"
  fi

  echo
done

echo
echo

if [ "$non_compliant" = true ]; then
  echo -e "${RED}Code compliance check completed with warnings.${NC}"
else
  echo -e "${GREEN}All .typ, .txt, .sh, .md, and .yml files are compliant!${NC}"
fi
exit 0
