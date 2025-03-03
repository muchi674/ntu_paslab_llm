#!/bin/bash

# List of configuration file suffixes
config_suffixes=(1 2 4 8)

# Path to your Python script
python_script="launch_nodes.py"

# Output log file
log_file="orig_output_log.txt"

# Clear the log file before starting (optional, if you want to start with a fresh log)
> "$log_file"

# Loop through each config file and run the Python script with that config
for suffix in "${config_suffixes[@]}"
do
    config_file="launch_config_v1_${suffix}.json"
    
    # Check if the config file exists
    if [ -f "$config_file" ]; then
        echo "Running with $config_file..." | tee -a "$log_file"
        
        # Run the Python script with the config file as an argument
        python3 "$python_script" --launch-config "$config_file" | tee -a "$log_file"
    else
        echo "Config file $config_file not found. Skipping..." | tee -a "$log_file"
    fi
done


