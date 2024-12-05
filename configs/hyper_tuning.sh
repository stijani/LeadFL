#!/bin/bash

# Path to the YAML template
CONFIG_TEMPLATE="./configs/template.yaml"
CONFIG_TEMP="./configs/temp.yaml"  # Temporary YAML file for modification

# YAML keys to update
YAML_KEYS=("pseudo_lr" "pseudo_momentum" "pseudo_iterations")  # Replace with the keys you want to modify

experiment_prefix=PSEUDO-HYPERTUNING-LR-MOMENTUM-ITERATION

# Values for each key
VALUES_lr=(0.01 0.1 0.005 0.001)  # Replace with your desired values for 'aggregation'
VALUES_momentum=(0.99 0.95 0.999)  # Replace with your desired values for 'learning_rate'
VALUES_iterations=(10 50 100 150 200)          # Replace with your desired values for 'batch_size'

# Iterate over values for each combination
for pseudo_lr in "${VALUES_lr[@]}"
do
    for pseudo_momentum in "${VALUES_momentum[@]}"
    do
        for pseudo_iterations in "${VALUES_iterations[@]}"
        do
            echo "Running with pseudo_lr=$pseudo_lr, pseudo_momentum=$pseudo_momentum, pseudo_iterations=$pseudo_iterations"
            run_prefix="hypertune-lr:$pseudo_lr-momentum:$pseudo_momentum-iterations:$pseudo_iterations"
            # Modify the YAML file with sed for each key
            sed -e "s/^pseudo_lr:.*/pseudo_lr: $pseudo_lr/"\
                -e "s/^pseudo_momentum:.*/pseudo_momentum: $pseudo_momentum/"\
                -e "s/^pseudo_iterations:.*/pseudo_iterations: $pseudo_iterations/"\
                -e "s/^experiment_prefix:.*/experiment_prefix: $experiment_prefix/"\
                -e "s/^run_prefix:.*/run_prefix: $run_prefix/"\
                "$CONFIG_TEMPLATE" > "$CONFIG_TEMP" 

            # Run the Python command with the modified YAML
            python3 -m fltk single "$CONFIG_TEMP"
            # Clean up the temporary YAML file
            rm -f "$CONFIG_TEMP"

            # Optional: Save results or logs for each run
            # mv output.log "output_${AGGREGATION}_${LR}_${BATCH_SIZE}.log"
        done
    done
done
echo "All runs completed."
