#!/bin/bash

combinations=(
  "0.9 0.9 1"
  "0.9 0.9 5"
  "0.9 0.9 10"
  "0.9 0.9 20"
  "0.9 0.9 50"
  "0.9 0.9 100"
  "0.9 0.9 500"
  "0.9 0.9 1000"
  "0.9 0.9 2000"
  )

# Path to the YAML template
# CONFIG_TEMPLATE="./configs/template_fashion_mnist.yaml"
CONFIG_TEMPLATE="./configs/template_cifar10.yaml"
CONFIG_TEMP="./configs/temp_cifar10_iter.yaml"  # Temporary YAML file for modification
# experiment_prefix="ABLATION-NEED-FOR-PSEUDO-OPTIMIZATION (MNIST)"
experiment_prefix="ABLATION-NEED-FOR-PSEUDO-OPTIMIZATION (CIFAR10) 0.9 momentum"
use_server_alignment=false
use_real_images=false
aggregator=multiKrum_logits

# Iterate over values for each combination
for combination in "${combinations[@]}"; do
    read -r pseudo_lr pseudo_momentum pseudo_iterations <<< "$combination"
    run_prefix="lr:$pseudo_lr-momentum:$pseudo_momentum-iterations:$pseudo_iterations-"
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

done
echo "All runs completed."
