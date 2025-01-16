#!/bin/bash

# Define the result file as a variable
# result_file="orig_result_with_warmup.txt"
result_file="orig_result.txt"
#result_file="no_split_result.txt"
# result_file="no_split_result_with_warmup.txt"
#result_file="split_result.txt"
# result_file="split_result_with_warmup.txt"

# List of input sizes
input_sizes=(1 2 4 8 16 32 64 128 256)

# Redirect the output to the result file (overwrite if it exists)
echo "Starting program execution..." > "$result_file"

# Loop over the input sizes and run the Python script for each size
for size in "${input_sizes[@]}"
do
  echo "Running program with input size: $size" >> "$result_file"
  python no_profile_test.py --input_size $size >> "$result_file" 2>&1
  echo "-------------------------" >> "$result_file"
done

echo "Program execution finished!" >> "$result_file"
