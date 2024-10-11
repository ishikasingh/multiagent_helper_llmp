#!/bin/bash

for file in 10_10_*_agent*.txt 10_10_choose*.txt 10_10_until*.txt; do
    OUTPUT_FILE="$file"
    SUMMARY_FILE="${file%.*}_summary.txt"
    
    echo "Processing $OUTPUT_FILE"
    python processor.py "$OUTPUT_FILE" "$SUMMARY_FILE"
done

echo "Processing complete."