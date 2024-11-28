#!/bin/bash

# Path to the YAML template
CONFIG_TEMPLATE="./configs/template.yaml"
CONFIG_TEMP="./configs/temp.yaml"  # Temporary YAML file for modification

# Key to update in the YAML file
YAML_KEY="aggregation"  # Replace with the key you want to modify

# Values to iterate over
VALUES=(multiKrum multiKrum_pseudo krum krum_pseudo bulyan krum clustering median)  # Replace with your desired values

# Iterate over values
for VALUE in "${VALUES[@]}"
do
    echo "Running with $YAML_KEY set to $VALUE"

    # Modify the YAML file
    sed "s/^$YAML_KEY:.*/$YAML_KEY: $VALUE/" "$CONFIG_TEMPLATE" > "$CONFIG_TEMP"

    # Run the Python command with the modified YAML
    python3 -m fltk single "$CONFIG_TEMP"

    # Optional: Save results or logs for each run
    # mv output.log "output_$VALUE.log"
done

# Clean up the temporary YAML file
rm -f "$CONFIG_TEMP"

echo "All runs completed."
