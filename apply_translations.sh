#!/usr/bin/env bash
# apply_translation.sh
# Usage: ./apply_translation.sh < extracted_lines.txt
# Reads lines in the format   filename:line:content
# and replaces the given line in the file with the provided content.

while IFS=: read -r file line content; do
    [[ -f $file && $line =~ ^[0-9]+$ ]] || continue
    printf '%s\n' \
        "${line}c" \
        "$content" \
        . \
        w \
        q \
        | ed -s "$file"
done
