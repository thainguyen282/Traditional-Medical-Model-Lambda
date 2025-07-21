#!/bin/bash
max_depth=4
for depth in $(seq 1 $max_depth); do
    echo -e "\nDepth $depth: Top 3 folders by file count"
    find . -mindepth $depth -maxdepth $depth -type d | while read dir; do
        count=$(find "$dir" -maxdepth 1 -type f | wc -l)
        printf "%6d %s\n" "$count" "$dir"
    done | sort -rn | head -3
done