import copy
from typing import Dict, List, Tuple
import torch
import numpy as np
from torch import nn
from torch.optim import Adam


def generate_seeded_pseudo_patterns(batch_size, input_shape, device, seed=42):
    if seed is not None:
        torch.manual_seed(seed)  # Seed the RNG for reproducibility
        if device.type == "cuda":  # If using CUDA, seed the CUDA RNG as well
            torch.cuda.manual_seed(seed)

    pseudo_patterns = torch.rand(
        (batch_size, *input_shape), requires_grad=True, device=device
    )
    return pseudo_patterns


def optimize_single_batch(
    config,
    model,
    labels: List[int],
    device: str,
    input_shape: Tuple = (1, 28, 28),
    # iterations: int = 100,  # 10(best so far) # 100,
    # lr: float = 0.001  # 0.01 SGD, serveronly(best so far), #0.5,
):
    """
    Optimize a single batch of pseudo patterns for a given set of labels.

    Args:
        model: The PyTorch model used for computing activations.
        input_shape: Shape of the input tensor (channel, height, width).
        iterations: Number of optimization iterations.
        lr: Learning rate for the optimizer.
        labels: A list of labels corresponding to the patterns in this batch.
        device: The device to run the computations on ('cuda' or 'cpu').

    Returns:
        A tensor of optimized pseudo patterns for this batch.
    """
    # Adjust input shape to channel-first if required
    if input_shape[0] > 3:
        input_shape = input_shape[::-1]

    # Batch size is determined by the number of labels
    batch_size = len(labels)

    # Initialize pseudo patterns for the current batch
    # pseudo_patterns = torch.rand(
    #     (batch_size, *input_shape), requires_grad=True, device=device
    # )

    pseudo_patterns = generate_seeded_pseudo_patterns(batch_size, input_shape, device)

    # Create an optimizer for the current batch
    # optimizer = Adam([pseudo_patterns], lr=lr)
    optimizer = torch.optim.SGD([pseudo_patterns], lr=config.pseudo_lr, momentum=config.pseudo_momentum)

    # Labels tensor for the batch
    labels_tensor = torch.tensor(labels, device=device)

    # Optimize the pseudo patterns for the current batch
    for _ in range(config.pseudo_iterations):
        optimizer.zero_grad()

        # Forward pass: Compute model outputs for the batch
        outputs = model(pseudo_patterns)

        # Compute loss: maximize activation for each label
        losses = -outputs[torch.arange(batch_size), labels_tensor]
        loss = losses.mean()  # Average loss across the batch
        loss.backward()

        # Update the pseudo patterns
        optimizer.step()

    # Detach abd convert to a list of nmpy arrays
    pseudo_pattern_list = pseudo_patterns.detach().cpu().unbind(0)
    return pseudo_pattern_list


def generate_pseudo_patterns(
    config,
    parameters: Dict[str, Dict[str, torch.Tensor]],
    model: nn.Module,
    device: torch.device,
    label: int = 10,
    global_params=None,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """
    Generate pseudo patterns for each client.
    Args:
        parameters: Client parameters.
        model: PyTorch model (already on device).
        device: Device for computation.
        labels: List of labels.
    Returns:
        Dictionary mapping client indices to pseudo patterns by label.
    """
    labels = list(range(label))

    if global_params:
        model.load_state_dict(copy.deepcopy(global_params), strict=True)
        pseudo_per_label_global = optimize_single_batch(config, model, labels, device)

    client_pseudo_dict = {}
    # for idx, client_id in enumerate(parameters.keys()):
    for idx, client_id in enumerate(sorted(parameters.keys())):
        model.load_state_dict(copy.deepcopy(parameters[client_id]), strict=True)
        # model.load_state_dict(parameters[client_id])
        pseudo_per_label = optimize_single_batch(config, model, labels, device)
        pseudo_per_label_dict = {lab: pseudo for lab, pseudo in enumerate(pseudo_per_label)}
        client_pseudo_dict[idx] = pseudo_per_label_dict
    return client_pseudo_dict, pseudo_per_label_global


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
                dist = np.linalg.norm(
                    client_pseudo_dict[i][label_id].cpu().numpy()
                    - client_pseudo_dict[j][label_id].cpu().numpy()
                )
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
            dist = np.linalg.norm(
                client_pseudo_dict[i][label_id].cpu().numpy()
                - server_pseudo_patterns[label_id].cpu().numpy()
            )
            total_distance += dist

        server_client_distances[i] = total_distance  ####### normalize this by dividing by the client sample count

    return server_client_distances


def krum_pseudo(
    config,
    parameters: Dict[str, Dict[str, torch.Tensor]],
    client_sizes: List[int],
    model: nn.Module,
    device: torch.device,
    candidate_num: int = 6,
    num_classes: int = 10,
    num_clients: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Krum: Select the most representative client.
    """
    client_id_to_int = {i: client_id for i, client_id in enumerate(sorted(parameters.keys()))}

    global_params = copy.deepcopy(model.state_dict())
    # Generate pseudo patterns
    client_pseudo_dict, server_pseudo_patterns = generate_pseudo_patterns(config, parameters, model, device, num_classes, global_params)

    # Calculate total scores
    # scores_client2client = calculate_client_client_distance(client_pseudo_dict, candidate_num=candidate_num, num_clients=num_clients, num_classes=num_classes)
    scores_client2server = calculate_server_client_distance(client_pseudo_dict, server_pseudo_patterns, num_clients=num_clients, num_classes=num_classes)

    ######################################################
    # print("#############1", client_sizes)
    # print("#############2", list(parameters.keys()))
    # raise ValueError
    # # normalize = True
    # # if normalize:
    # #     scores_client2client = 
    # #     scores_client2client
    ######################################################

    alpha = 0.5
    # final_scores = scores_client2client + alpha * scores_client2server  # TODO: 1. make optional to add scores_client2server 2. make alpha an argument
    final_scores = scores_client2server
    # final_scores = scores_client2client
    # final_scores = 0.5 * scores_client2client + 0.5 * scores_client2server

    # Select the client with the lowest total score
    best_client_index = np.argmin(final_scores)
    best_client_id = client_id_to_int[best_client_index]
    best_client_id = list(parameters.keys())[best_client_index]

    return parameters[best_client_id]
    ###### return parameters[best_client_id] # fix


def multiKrum_pseudo(
    config,
    parameters: Dict[str, Dict[str, torch.Tensor]],
    client_sizes: List[int],
    model: nn.Module,
    device: torch.device,
    candidate_num: int = 7,
    num_classes: int = 10,
    num_selected: int = 5,
    num_clients: int = 10,
) -> List[Dict[str, torch.Tensor]]:
    """
    Multi-Krum: Select multiple representative clients.
    """
    # labels = list(range(10))

    global_params = copy.deepcopy(model.state_dict())

    # Generate pseudo patterns
    client_pseudo_dict, server_pseudo_patterns = generate_pseudo_patterns(config, parameters, model, device, num_classes, global_params)

    # Calculate total scores
    # scores_client2client = calculate_client_client_distance(client_pseudo_dict, candidate_num=candidate_num, num_clients=num_clients, num_classes=num_classes)
    scores_client2server = calculate_server_client_distance(client_pseudo_dict, server_pseudo_patterns, num_clients=num_clients, num_classes=num_classes)

    alpha = 1
    # final_scores = scores_client2client + alpha * scores_client2server
    # final_scores = scores_client2client
    final_scores = scores_client2server

    # Select the indices of the `num_selected` clients with the lowest total scores
    selected_indices = np.argsort(final_scores)[:num_selected]

    # Map selected indices back to client IDs and parameters
    candidate_parameters = [
        parameters[list(parameters.keys())[idx]] for idx in selected_indices
    ]

    # average the candidate params
    new_params = {}
    for name in candidate_parameters[0].keys():
        new_params[name] = sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters)

    return new_params


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
