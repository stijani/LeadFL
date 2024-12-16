#!/bin/bash

combinations=(
  "0.5 0.9 10"
  "0.01 0.75 10"
  "0.75 0.99 100"
  "0.01 0.9 100"
  "0.3 0.5 100"
  "0.75 0.5 100"
  "0.01 0.95 1000"
  "0.1 0.9 100"
  "0.75 0.5 1000"
  "0.95 0.9 1000"
  "0.75 0.99 1000"
  "0.1 0.9 10"
  "0.05 0.9 100"
  "0.5 0.75 10"
  "0.05 0.9 1000"
  "0.05 0.99 10"
  "0.75 0.95 10"
  "0.5 0.9 1000"
  "0.05 0.5 1000"
  "0.1 0.99 1000"
  "0.05 0.99 1000"
  "0.75 0.75 10"
  "0.95 0.95 10"
  "0.95 0.9 10"
  "0.95 0.75 10"
  "0.1 0.95 1000"
  "0.1 0.95 100"
  "0.3 0.75 1000"
  "0.01 0.95 10"
  "0.01 0.5 10"
  "0.75 0.99 10"
  "0.01 0.99 100"
  "0.01 0.5 1000"
  "0.95 0.95 100"
  "0.5 0.95 100"
  "0.01 0.75 100"
  "0.01 0.5 100"
  "0.75 0.5 10"
  "0.5 0.5 10"
  "0.3 0.95 100"
  "0.01 0.9 10"
  "0.95 0.75 100"
  "0.5 0.5 1000"
  "0.1 0.9 1000"
  "0.95 0.99 100"
  "0.95 0.95 1000"
  "0.3 0.9 10"
  "0.05 0.75 1000"
  "0.3 0.75 100"
  "0.1 0.5 1000"
  "0.75 0.95 1000"
  "0.75 0.95 100"
  "0.3 0.99 1000"
  "0.1 0.95 10"
  "0.1 0.99 100"
  "0.75 0.9 100"
  "0.95 0.99 10"
  "0.95 0.5 100"
  "0.05 0.95 1000"
  "0.1 0.99 10"
  "0.3 0.95 10"
  "0.05 0.75 100"
  "0.75 0.75 1000"
  "0.01 0.75 1000"
  "0.95 0.5 10"
  "0.95 0.5 1000"
  "0.5 0.75 100"
  "0.75 0.75 100"
  "0.95 0.99 1000"
  "0.1 0.5 10"
  "0.05 0.5 10"
  "0.1 0.75 1000"
  "0.5 0.95 10"
  "0.1 0.5 100"
  "0.05 0.75 10"
  "0.5 0.99 100"
  "0.05 0.95 100"
  "0.01 0.95 100"
  "0.3 0.5 10"
  "0.75 0.9 10"
  "0.95 0.9 100"
  "0.05 0.99 100"
  "0.5 0.99 10"
  "0.5 0.75 1000"
  "0.05 0.9 10"
  "0.3 0.5 1000"
  "0.95 0.75 1000"
  "0.1 0.75 100"
  "0.5 0.5 100"
  "0.3 0.99 10"
  "0.1 0.75 10"
  "0.05 0.5 100"
  "0.5 0.95 1000"
  "0.3 0.75 10"
  "0.75 0.9 1000"
  "0.3 0.9 1000"
  "0.01 0.9 1000"
  "0.3 0.95 1000"
  "0.01 0.99 1000"
  "0.01 0.99 10"
  "0.3 0.99 100"
  "0.5 0.9 100"
  "0.5 0.99 1000"
  "0.3 0.9 100"
  "0.05 0.95 10"
)

# Path to the YAML template
CONFIG_TEMPLATE="./configs/template_cifar10.yaml"
CONFIG_TEMP="./configs/temp_cifar10_5.yaml"  # Temporary YAML file for modification
experiment_prefix=HYPER-TUNING-CIFAR10-RANDOM-MAL-UNIFORM-CLIENTS-S2C
use_server_alignment=true
use_real_images=false
aggregator=multiKrum_logits

# Values for each key
# VALUES_lr=(0.01 0.75 0.05 0.1 0.3 0.95 0.5)  # Replace with your desired values for 'aggregation'
# VALUES_momentum=(0.5 0.95 0.75 0.99 0.9)  # Replace with your desired values for 'learning_rate'
# VALUES_iterations=(10 100 1000)          # Replace with your desired values for 'batch_size'
# VALUES_aggregators=(multiKrum_logits)

# Iterate over values for each combination
for combination in "${combinations[@]}"; do
    read -r pseudo_lr pseudo_momentum pseudo_iterations <<< "$combination"
    run_prefix="lr:$pseudo_lr momentum:$pseudo_momentum iterations:$pseudo_iterations"
    echo "Running with: $run_prefix"
    # Modify the YAML file with sed for each key
    sed -e "s/^pseudo_lr:.*/pseudo_lr: $pseudo_lr/"\
        -e "s/^pseudo_momentum:.*/pseudo_momentum: $pseudo_momentum/"\
        -e "s/^pseudo_iterations:.*/pseudo_iterations: $pseudo_iterations/"\
        -e "s/^experiment_prefix:.*/experiment_prefix: $experiment_prefix/"\
        -e "s/^use_server_alignment:.*/use_server_alignment: $use_server_alignment/"\
        -e "s/^use_real_images:.*/use_real_images: $use_real_images/"\
        -e "s/^aggregation:.*/aggregation: $aggregator/"\
        -e "s/^run_prefix:.*/run_prefix: $run_prefix/"\
        "$CONFIG_TEMPLATE" > "$CONFIG_TEMP" 

    # Run the Python command with the modified YAML
    python3 -m fltk single "$CONFIG_TEMP"
    # Clean up the temporary YAML file
    rm -f "$CONFIG_TEMP"

    # Optional: Save results or logs for each run
    # mv output.log "output_${AGGREGATION}_${LR}_${BATCH_SIZE}.log"
done
echo "All runs completed."
