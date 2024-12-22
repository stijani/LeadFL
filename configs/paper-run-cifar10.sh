#!/bin/bash

# Path to the YAML template
CONFIG_TEMPLATE="./configs/template_cifar10.yaml"
experiment_prefix=PAPER-RUN-CIFAR10
rounds=100  # Number of rounds to set in the YAML
selection_method=consistent

# Values for each key
ATTACKS=("1-pixel" "9-pixel" "TargetedAttack" "LabelFlip")
AGGRS=("krum" "krum_logits" "multiKrum" "multiKrum_logits" "bulyan" "bulyan_logits" "median" "trmean")

# Function to get a random CUDA device
# 4 + 4 -> gpus: 4,5,6,7
get_random_cuda_device() {
    devices=(2 3 4 6)
    echo ${devices[$(( RANDOM % ${#devices[@]} ))]}
}

# Log directory for nohup outputs
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Iterate over values for each attack
for attack in "${ATTACKS[@]}"
do
    {
        if [[ "$attack" == "1-pixel" || "$attack" == "9-pixel" ]]; then
            attack_client="Backdoor"
            backdoor_type="$attack"
        else
            attack_client="$attack"
            backdoor_type=""  # Clear backdoor_type for non-backdoor attacks
        fi

        # Assign a single random CUDA device for this attack
        CUDA_DEVICE=$(get_random_cuda_device)
        echo "Using CUDA device: $CUDA_DEVICE for attack: $attack"

        # Run all aggregation strategies sequentially on the same device
        for aggr in "${AGGRS[@]}"
        do
            run_prefix="attack:$attack | aggr:${aggr}_"
            echo "Running with experiment=$run_prefix"

            # Unique temp file for each combination
            CONFIG_TEMP="./configs/cifar10_${attack}_${aggr}.yaml"

            # Modify the YAML file with sed for each key
            sed -e "s/^aggregation:.*/aggregation: $aggr/" \
                -e "s/^attack_client:.*/attack_client: $attack_client/" \
                -e "s/^backdoor_type:.*/backdoor_type: $backdoor_type/" \
                -e "s/^rounds:.*/rounds: $rounds/" \
                -e "s/^selection_method:.*/selection_method: $selection_method/" \
                -e "s/^experiment_prefix:.*/experiment_prefix: PAPER-RUN-$attack/" \
                "$CONFIG_TEMPLATE" > "$CONFIG_TEMP"

            # Run the Python command sequentially with the random CUDA device
            nohup env CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 -m fltk single "$CONFIG_TEMP" >> "$LOG_DIR/${attack}_${aggr}.log" 2>&1
            
            # clean ups
            rm -f "$CONFIG_TEMP"
        done
    } &
done

# Wait for all background processes to complete
wait
echo "All runs completed."
