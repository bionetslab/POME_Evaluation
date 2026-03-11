#!/bin/bash

# Stop script on error
set -e

# Directories
INPUT_DIR="simulated_ohe_data"
OUTPUT_DIR="imputed_ohe_data"

# Ensure output directory exists

source ~/miniforge3/etc/profile.d/conda.sh

# Activate environment
conda activate ac

# Iterate over all CSV files
for file_path in "$INPUT_DIR"/*.csv; do

    # Skip if no files found
    [ -e "$file_path" ] || continue

    filename=$(basename -- "$file_path")
    name_without_ext="${filename%.csv}"
    output_file="$OUTPUT_DIR/${name_without_ext}.csv"

    echo "Processing $filename ..."

    python /home/wollerf/Projects/AutoComplete.git/fit.py \
        "$file_path" \
        --id_name ID \
        --batch_size 1024 \
        --epochs 50 \
        --lr 0.1 \
        --save_imputed \
        --device cpu:0 \
        --output "$output_file"

    echo "Finished $filename"
done

echo "All files processed."
