import copy
from typing import Dict, List, Tuple
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
import random
import time
from distance_calculation import cosine_distance, euclidean_distance


SEED = 42


def generate_seeded_pseudo_patterns(batch_size=1, input_shape=(1, 28, 28), device=None, seed=42):
    if seed is not None:
        torch.manual_seed(seed)  # Seed the RNG for reproducibility
        if device.type == "cuda":  # If using CUDA, seed the CUDA RNG as well
            torch.cuda.manual_seed(seed)
    pseudo_patterns = torch.rand(
        (batch_size, *input_shape), requires_grad=True, device=device
    )
    return pseudo_patterns


def optimize_pseudo_pattern_batch(
    config,
    model,
    label_batch: List[int],
    device: str,
    pseudo_pattern_batch,
):
    optimizer = torch.optim.SGD([pseudo_pattern_batch], lr=config.pseudo_lr, momentum=config.pseudo_momentum)
    # optimizer = optimizer = Adam([pseudo_pattern_batch], lr=0.0001)
    labels_tensor = torch.tensor(label_batch, device=device)

    for _ in range(config.pseudo_iterations):
        optimizer.zero_grad()
        pseudo_pattern_batch = pseudo_pattern_batch.to(device) #############
        outputs = model(pseudo_pattern_batch)

        # Compute loss: maximize activation for each label
        losses = -outputs[torch.arange(len(label_batch)), labels_tensor]
        loss = losses.mean()  # Average loss across the batch
        loss.backward()
        optimizer.step()

    # Detach abd convert to a list of nmpy arrays
    pseudo_pattern_npy = pseudo_pattern_batch.detach().cpu().unbind(0)
    # print("#######################1", [pseudo for pseudo in pseudo_pattern_npy])
    # return [pseudo for pseudo in pseudo_pattern_npy]
    return pseudo_pattern_npy 


# def get_optimized_pseudo_patterns(
#     config,
#     parameters: Dict[str, Dict[str, torch.Tensor]],
#     model: nn.Module,
#     device: torch.device,
#     label: int = 10,
#     global_params=None,
#     prev_client_pseudo_dict={}, 
#     prev_server_pseudo_patterns=[],
# ) -> Dict[int, Dict[int, torch.Tensor]]:
#     """
#     Generate pseudo patterns for each client.
#     Args:
#         parameters: Client parameters.
#         model: PyTorch model (already on device).
#         device: Device for computation.
#         labels: List of labels.
#     Returns:
#         Dictionary mapping client indices to pseudo patterns by label.
#     """
#     labels = list(range(label))

#     if global_params:
#         init_pseudo = None
#         if prev_server_pseudo_patterns:
#             init_pseudo = prev_server_pseudo_patterns
#         model.load_state_dict(copy.deepcopy(global_params), strict=True)
#         pseudo_per_label_global = optimize_pseudo_pattern_batch(config, model, labels, device, init_pseudo=init_pseudo)

#     client_pseudo_dict = {}
#     # for idx, client_id in enumerate(parameters.keys()):
#     for idx, client_id in enumerate(sorted(parameters.keys())):
#         model.load_state_dict(copy.deepcopy(parameters[client_id]), strict=True)
#         init_pseudo = None
#         if prev_client_pseudo_dict:
#             if client_id in prev_client_pseudo_dict:
#                 init_pseudo = prev_client_pseudo_dict[client_id]
#                 # print("####################", init_pseudo)
#             # model.load_state_dict(parameters[client_id])
#         pseudo_per_label = optimize_pseudo_pattern_batch(config, model, labels, device, init_pseudo=init_pseudo)
#         pseudo_per_label_dict = {lab: pseudo for lab, pseudo in enumerate(pseudo_per_label)}
#         client_pseudo_dict[idx] = pseudo_per_label_dict
#     return client_pseudo_dict, pseudo_per_label_global


def calculate_client_client_distance(
    client_pseudo_dict: Dict[int, Dict[int, torch.Tensor]],
    candidate_num: int = 6,
    num_clients: int = 10,
    num_classes: int = 10
) -> np.ndarray:
    """
    Calculate total scores for each client based on distances.
    Args:
        client_pseudo_dict: Pseudo patterns for each client and label.
        labels: List of labels.
    Returns:
        Total scores as a NumPy array.
    """

    labels = list(range(num_classes))

    num_clients = len(client_pseudo_dict)
    total_scores = np.zeros(num_clients)

    for label_id in labels:
        distances = np.zeros((num_clients, num_clients))

        # Compute pairwise distances for the current label
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                # dist = np.linalg.norm(
                #     client_pseudo_dict[i][label_id].cpu().numpy()
                #     - client_pseudo_dict[j][label_id].cpu().numpy()
                # )
                # dist = euclidean_distance(client_pseudo_dict[i][label_id], client_pseudo_dict[j])
                dist = cosine_distance(client_pseudo_dict[i][label_id], client_pseudo_dict[j][label_id])
                distances[i, j] = dist
                distances[j, i] = dist

        # Calculate scores for each client for the current label
        for i in range(num_clients):
            sorted_distances = sorted(distances[i])
            # total_scores[i] += sum(sorted_distances[:candidate_num + 1])
            total_scores[i] += sum(sorted_distances[:candidate_num])

    return total_scores


def calculate_server_client_distance(
    client_pseudo_dict: Dict[int, Dict[int, torch.Tensor]],
    server_pseudo_patterns: Dict[int, torch.Tensor],
    num_clients: int = 10,
    num_classes: int = 10,
) -> np.ndarray:
    """
    Calculate the distances between global pseudo patterns (server) and client pseudo patterns.

    Args:
        client_pseudo_dict: Pseudo patterns for each client and label.
        server_pseudo_patterns: Pseudo patterns generated from the global model.
        num_clients: Total number of clients.
        label: Number of labels.

    Returns:
        A NumPy array of shape (num_clients,) containing the server-to-client distances.
    """
    labels = list(range(num_classes))
    server_client_distances = np.zeros(num_clients)

    for i in range(num_clients):
        total_distance = 0.0

        # Compute distance for all labels
        for label_id in labels:
            # dist = np.linalg.norm(
            #     client_pseudo_dict[i][label_id].cpu().numpy()
            #     - server_pseudo_patterns[label_id].cpu().numpy()
            # )
            # dist = cosine_distance(client_pseudo_dict[i][label_id], server_pseudo_patterns[label_id])
            dist = euclidean_distance(client_pseudo_dict[i][label_id], server_pseudo_patterns[label_id])
            total_distance += dist

        server_client_distances[i] = total_distance  # ###### normalize this by dividing by the client sample count

    return server_client_distances


# def generate_pseudo_all_label(config, model, state_dict, num_classes, device, seed, init_pseudo=None):
#     pseudos_patterns = []    
#     for lab_id in range(num_classes):
#         model.load_state_dict(copy.deepcopy(state_dict), strict=True)
#         if init_pseudo:
#             single_pseudo = [init_pseudo[lab_id]]
#             single_pseudo = torch.stack(single_pseudo)
#         else:
#             single_pseudo = generate_seeded_pseudo_patterns(device=device, seed=seed)
#         single_pseudo_optimized = optimize_pseudo_pattern_batch(config, model, [lab_id], device, single_pseudo)
#         pseudos_patterns.extend(single_pseudo_optimized)
#     return pseudos_patterns


def generate_pseudo_all_label(config, model, state_dict, num_classes, device, seed, init_pseudo=None):

    x_npy = np.load("/home/stijani/projects/dataset/mnist-fashion/test_features.npy")
    y_npy = np.load("/home/stijani/projects/dataset/mnist-fashion/test_labels.npy")

    sampled_features = []
    sampled_labels = []

    for cls in range(num_classes):
        class_indices = np.where(y_npy == cls)[0]
        random_idx = np.random.choice(class_indices, size=1, replace=False)[0]

        # Append the feature and label for this index
        sampled_features.append(x_npy[random_idx])
        sampled_labels.append(y_npy[random_idx])

    pseudos_patterns = []    
    for lab_id, feature in zip(sampled_labels, sampled_features):
        model.load_state_dict(copy.deepcopy(state_dict), strict=True)
        if init_pseudo:
            single_pseudo = [init_pseudo[lab_id]]
            single_pseudo = torch.stack(single_pseudo)
        else:
            feature = feature / 255.0
            feature = feature.transpose(2, 0, 1)
            feature = torch.from_numpy(feature)
            single_pseudo = torch.stack([feature])
            single_pseudo = single_pseudo.to(torch.float32)
            single_pseudo.to(device)
        single_pseudo_optimized = optimize_pseudo_pattern_batch(config, model, [lab_id], device, single_pseudo)
        pseudos_patterns.extend(single_pseudo_optimized)
    return pseudos_patterns


#######################################
# UNCOMMENT BELOW TO USE PSEUDO PATTERNS
#######################################
# def generate_pseudo_all_label(config, model, state_dict, num_classes, device, seed, init_pseudo=None):
#     pseudos_patterns = []    
#     for lab_id in range(num_classes):
#         model.load_state_dict(copy.deepcopy(state_dict), strict=True)
#         if init_pseudo:
#             single_label_pseudo = [init_pseudo[lab_id]]
#             single_label_pseudo = torch.stack(single_label_pseudo)
#             single_label_pseudo_optimized = optimize_pseudo_pattern_batch(config, model, [lab_id], device, single_label_pseudo)
#         else:
#             # average multiple pseudo per label
#             single_label_pseudo = generate_seeded_pseudo_patterns(batch_size=1, device=device, seed=seed)  # batch_size defaults to 1
#             # generate pseudo for this label
#             single_label_pseudo_optimized = optimize_pseudo_pattern_batch(config, model, [lab_id], device, single_label_pseudo)
#             ##################################
#             # below line makes sense when multiple
#             # Pseudo patterns are generated per label and then averaged
#             # in this case we set the batch_size above to a number n > 1
#             # and set use [lab_id]*n rather than [lab_id] in the line above
#             ##################################
#             single_label_pseudo_optimized = torch.mean(torch.stack(single_label_pseudo_optimized), dim=0)
#         pseudos_patterns.extend(single_label_pseudo_optimized)
#     return pseudos_patterns


def krum_pseudo(
    config,
    parameters: Dict[str, Dict[str, torch.Tensor]],
    client_sizes: List[int],
    model: nn.Module,
    device: torch.device,
    prev_client_pseudo_dict,
    prev_server_pseudo_patterns,
    candidate_num: int = 6,
    num_classes: int = 10,
    num_clients: int = 10,
    seed: int = 42,
    # seed: int = None,
    use_prev_pseudos: bool = False,  # @TODO: move to config
    client2server: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Krum: Select the most representative client.
    """
    # map client id to ints
    client_id_to_int = {i: client_id for i, client_id in enumerate(sorted(parameters.keys()))}

    if not use_prev_pseudos:
        prev_client_pseudo_dict = {}
        prev_server_pseudo_patterns = None

    # get server pseudo
    server_state_dict = copy.deepcopy(model.state_dict())
    init_pseudo = prev_server_pseudo_patterns
    server_pseudos = generate_pseudo_all_label(config, model, server_state_dict, num_classes, device, seed, init_pseudo=init_pseudo)

    # get client pseudos
    client_pseudo_dict = {}
    for client_idx, client_id in client_id_to_int.items():
        client_state_dict = copy.deepcopy(parameters[client_id])
        init_pseudo = prev_client_pseudo_dict.get(client_id, None)
        client_pseudos = generate_pseudo_all_label(config, model, client_state_dict, num_classes, device, seed, init_pseudo=init_pseudo)
        client_pseudo_dict[client_idx] = client_pseudos

    # Calculate total scores

    if client2server:
        final_scores = calculate_server_client_distance(client_pseudo_dict, server_pseudos, num_clients=num_clients, num_classes=num_classes)
    else:
        final_scores = calculate_client_client_distance(client_pseudo_dict, candidate_num=candidate_num, num_clients=num_clients, num_classes=num_classes)

    # alpha = 0.5
    # final_scores = scores_client2client + alpha * scores_client2server  # TODO: 1. make optional to add scores_client2server 2. make alpha an argument

    # Select the client with the lowest total score
    best_client_index = np.argmin(final_scores)
    best_client_id = client_id_to_int[best_client_index]

    print("##################1", best_client_index, best_client_id)
    print("##################2", final_scores)


    # map client pseudo to client id
    client_id_vs_pseudo = {client_id: client_pseudo_dict[idx] for idx, client_id in client_id_to_int.items()}

    return parameters[best_client_id], client_id_vs_pseudo, server_pseudos


def multiKrum_pseudo(
    config,
    parameters: Dict[str, Dict[str, torch.Tensor]],
    client_sizes: List[int],
    model: nn.Module,
    device: torch.device,
    prev_client_pseudo_dict,
    prev_server_pseudo_patterns,
    candidate_num: int = 6,
    num_classes: int = 10,
    num_selected: int = 5,
    num_clients: int = 10,
    # seed: int = None,
    seed: int = 42,
    use_prev_pseudos: bool = False,  # @TODO: move to config
    client2server: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Krum: Select the most representative client.
    """
    # map client id to ints
    client_id_to_int = {i: client_id for i, client_id in enumerate(sorted(parameters.keys()))}

    if not use_prev_pseudos:
        prev_client_pseudo_dict = {}
        prev_server_pseudo_patterns = None

    # get server pseudo
    server_state_dict = copy.deepcopy(model.state_dict())
    init_pseudo = prev_server_pseudo_patterns
    server_pseudos = generate_pseudo_all_label(config, model, server_state_dict, num_classes, device, seed, init_pseudo=init_pseudo)

    # get client pseudos
    client_pseudo_dict = {}
    for client_idx, client_id in client_id_to_int.items():
        client_state_dict = copy.deepcopy(parameters[client_id])
        init_pseudo = prev_client_pseudo_dict.get(client_id, None)
        client_pseudos = generate_pseudo_all_label(config, model, client_state_dict, num_classes, device, seed, init_pseudo=init_pseudo)
        client_pseudo_dict[client_idx] = client_pseudos

    # Calculate total scores

    if client2server:
        final_scores = calculate_server_client_distance(client_pseudo_dict, server_pseudos, num_clients=num_clients, num_classes=num_classes)
    else:
        final_scores = calculate_client_client_distance(client_pseudo_dict, candidate_num=candidate_num, num_clients=num_clients, num_classes=num_classes)

    # alpha = 0.5
    # final_scores = scores_client2client + alpha * scores_client2server  # TODO: 1. make optional to add scores_client2server 2. make alpha an argument

    # Select the client with the lowest total score
    # best_client_index = np.argmin(final_scores)
    # best_client_id = client_id_to_int[best_client_index]

    # selected_indices = np.argsort(final_scores)[:num_selected]
    best_client_idxs = [idx for idx, val in sorted(enumerate(final_scores), key=lambda x: x[1])[:num_selected]]
    best_client_ids = [client_id_to_int[client_idx] for client_idx in best_client_idxs]
    best_parameters = [parameters[best_client_id] for best_client_id in best_client_ids]

    print("##################1", best_client_idxs)
    print("##################1", final_scores)

    # average the candidate params
    new_params = {}
    for name in parameters[best_client_ids[0]].keys():
        new_params[name] = sum([param[name].data for param in best_parameters]) / len(best_parameters)

    # map client pseudo to client id
    client_id_vs_pseudo = {client_id: client_pseudo_dict[idx] for idx, client_id in client_id_to_int.items()}

    return new_params, client_id_vs_pseudo, server_pseudos


if __name__ == "__main__":
    # test the median function
    params = {
        '1': {'1': torch.Tensor([[1, 2, 3], [4, 5, 6]]), '2': torch.Tensor([[7, 8, 9], [10, 11, 12]])},
        '2': {'1': torch.Tensor([[13, 14, 15], [16, 17, 18]]), '2': torch.Tensor([[19, 20, 21], [22, 23, 24]])},
        '3': {'1': torch.Tensor([[25, 26, 27], [28, 29, 30]]), '2': torch.Tensor([[31, 32, 33], [34, 35, 36]])},
        '4': {'1': torch.Tensor([[37, 38, 39], [40, 41, 42]]), '2': torch.Tensor([[43, 44, 45], [46, 47, 48]])},
        '5': {'1': torch.Tensor([[49, 50, 51], [52, 53, 54]]), '2': torch.Tensor([[55, 56, 57], [58, 59, 60]])},
        '6': {'1': torch.Tensor([[61, 62, 63], [64, 65, 66]]), '2': torch.Tensor([[67, 68, 69], [70, 71, 72]])},
        '7': {'1': torch.Tensor([[73, 74, 75], [76, 77, 78]]), '2': torch.Tensor([[79, 80, 81], [82, 83, 84]])},
        '8': {'1': torch.Tensor([[85, 86, 87], [88, 89, 90]]), '2': torch.Tensor([[91, 92, 93], [94, 95, 96]])},
        '9': {'1': torch.Tensor([[97, 98, 99], [100, 101, 102]]), '2': torch.Tensor([[103, 104, 105], [106, 107, 108]])},
        '10': {'1': torch.Tensor([[109, 110, 111], [112, 113, 114]]), '2': torch.Tensor([[115, 116, 117], [118, 119, 120]])}}
    sizes = {'1': 2, '2': 2, '3': 3}

    print(krum_pseudo(params, sizes))
    # calculate_client_client_distance
