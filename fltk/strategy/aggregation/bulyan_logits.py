from typing import Dict
import torch
import copy
import torch
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine
from types import SimpleNamespace

import sys
sys.path.append("./")
from fltk.strategy.helper_pseudo_logits import generate_real_samples, generate_pseudo_patterns, optimize_pseudo_patterns, apply_pseudo_patterns_to_client, server_to_client_alignment, client_to_client_alignment


def bulyan_logits(
    com_round,
    config,
    parameters: Dict[str, Dict[int, torch.Tensor]],
    client_sizes,
    model: torch.nn.Module,
    device: torch.device,
    # num_clients_to_select: int = 1
) -> List[int]:

    # keep the global params for optional use later
    global_state_dict = copy.deepcopy(model.state_dict())

 # Generate pseudo patterns
    num_label_classes = config.num_label_classes
    num_pseudo_patterns_per_label = config.num_pseudo_patterns_per_label
    # image_shape = config.image_shape
    image_shape = (config.input_channels, config.input_width, config.input_height)
    lr = config.pseudo_lr
    iter = config.pseudo_iterations
    momentum = config.pseudo_momentum

    # for the multiKrum stage
    num_clients_to_select = config.clients_per_round - config.mal_clients_per_round
    use_server_alignment = config.use_server_alignment

    # generate pseudo pattern or real images for rehearsal/KD
    if config.use_real_images:
        pseudo_patterns, labels = generate_real_samples(num_label_classes, num_pseudo_patterns_per_label, image_shape, device)
    else:
        pseudo_patterns, labels = generate_pseudo_patterns(num_label_classes, num_pseudo_patterns_per_label, image_shape, device)

    client_logits_dict = {}
    for client_id, client_params in parameters.items():
        client_model = model
        client_model.load_state_dict(copy.deepcopy(client_params), strict=True)
        client_logits_dict[client_id] = apply_pseudo_patterns_to_client(client_model, pseudo_patterns, labels, device)

    # dissimilarity_scores = {}

    # THIS DISIMILARITY IS COMPUTED BY CLIENTS
    if use_server_alignment:  # use server to client similarity criteriion
        model.load_state_dict(global_state_dict, strict=True)
        global_logits_dict = optimize_pseudo_patterns(pseudo_patterns, labels, model, device, iter, lr, momentum)
        dissimilarity_scores = server_to_client_alignment(global_logits_dict, client_logits_dict, device)
    else:
        dissimilarity_scores = client_to_client_alignment(client_logits_dict, device)

    # Sort clients by dissimilarity score (ascending order) and select top clients
    # 1. Server recieves or compute dissimilarity scores for each client
    # 2. aggregates model based on the most similar clients
    # ALTERNATIVELY: server can also use DBSCAN to cluster the clients rather than distance based similarity
    selected_client_ids = sorted(dissimilarity_scores, key=dissimilarity_scores.get)[:num_clients_to_select]
    # print(f"############### Dissimilarity scores: {dissimilarity_scores}")
    # print(f"############### Selected Client IDs: {selected_client_ids}")
    selected_params = [parameters[client_id] for client_id in selected_client_ids]
    # Step 3: Compute median for each parameter across selected clients
    medians = {}
    for name in selected_params[0].keys():
        stacked_tensors = torch.stack([param[name] for param in selected_params])
        medians[name] = torch.median(stacked_tensors, dim=0).values

    # Step 4: Average beta closest parameters to the median
    f = 3  # TODO: just for testing, this needs to be in config as: mal_clients_per_round (bulyan only was for constant f)
    beta = max(1, (config.clients_per_round - 2 * f) // 2)  # Define beta using the formula # Define beta as the number of parameters to consider for averaging

    aggregated_params = {}
    for name in selected_params[0].keys():
        # Compute distances of each candidate's parameter layerwise from the median
        if "num_batches_tracked" in name:
            aggregated_params[name] = selected_params[0][name]  # Copy from any client
            continue

        median = medians[name]
        distances_to_median = [
            torch.norm((param[name] - median).float())**2 for param in selected_params
        ]

        # Select beta closest parameters to the median
        closest_indices = torch.argsort(torch.tensor(distances_to_median))[:beta]
        closest_tensors = torch.stack([selected_params[i][name] for i in closest_indices])

        # Average the beta closest parameters for this layer
        aggregated_params[name] = torch.mean(closest_tensors, dim=0)

    return aggregated_params, selected_client_ids
