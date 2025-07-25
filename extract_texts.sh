#!/bin/bash

# Extract lines containing comments (//) or string literals (") from all .rs files
# Output format: filename:line_number:line_content
# This format is suitable for patching or translation mapping

# Usage: ./extract_texts.sh

# Recursive grep: search for lines containing either // or " in .rs files
# -r: recursive search
# -n: show line numbers
# --include='*.rs': limit to Rust files
# 2>/dev/null: suppress permission denied and other errors
grep -rn --include='*.rs' --include='*.html' -e '"' -e '//' . 2>/dev/null
