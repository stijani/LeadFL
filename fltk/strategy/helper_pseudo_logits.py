import copy
import torch
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine
from types import SimpleNamespace


def normalize_client_sample_quantities(client_sizes):
    # min_val = min(client_sizes.values())
    max_val = max(client_sizes.values())

    # Normalize the values
    # normalized_data = {k: (v - min_val) / (max_val - min_val) for k, v in client_sizes.items()}
    normalized_data = {k: v / max_val for k, v in client_sizes.items()}
    return normalized_data

def generate_real_samples(
    num_classes,
    num_pseudo_patterns,
    input_shape,
    device,
    dataset_name,
    seed: int = 42,
):
    """
    Generate real samples as pseudo patterns using existing dataset.

    Args:
        config: Configuration object.
        model: PyTorch model (unused here but kept for interface compatibility).
        state_dict: Model state dictionary (unused here but kept for interface compatibility).
        num_classes: Number of classes in the dataset.
        device: Device to move the data to (e.g., 'cuda' or 'cpu').
        seed: Random seed for reproducibility.
        num_pseudo_patterns: Number of pseudo patterns to sample per class.
        input_shape: Shape of the input data.

    Returns:
        pseudo_patterns: Tensor containing the real samples as pseudo patterns.
        labels: Tensor containing the corresponding labels.
    """
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Load the dataset
    # x_npy = np.load("/home/stijani/projects/dataset/fashion-mnist/test_features.npy")
    # y_npy = np.load("/home/stijani/projects/dataset/fashion-mnist/test_labels.npy")

    x_npy = np.load(f"/home/stijani/projects/dataset/{dataset_name}/test_features.npy")
    y_npy = np.load(f"/home/stijani/projects/dataset/{dataset_name}/test_labels.npy")

    sampled_features = []
    sampled_labels = []

    # Sample the specified number of patterns per class
    for cls in range(num_classes):
        class_indices = np.where(y_npy == cls)[0]
        selected_indices = np.random.choice(class_indices, size=num_pseudo_patterns, replace=False)

        # Append the features and labels for the selected indices
        sampled_features.extend(x_npy[selected_indices])
        sampled_labels.extend(y_npy[selected_indices])

    # Convert sampled features and labels to tensors
    pseudo_patterns = torch.tensor(sampled_features, dtype=torch.float32).view(-1, *input_shape).to(device)
    labels = torch.tensor(sampled_labels, dtype=torch.long).to(device)
    # print("########################1111", pseudo_patterns) not normalized up tp this point
    pseudo_patterns = pseudo_patterns / 255.0
    # print("########################1111", pseudo_patterns)

    return pseudo_patterns, labels


def generate_pseudo_patterns(
    num_classes: int,
    num_pseudo_patterns: int,
    input_shape: Tuple[int, int, int],
    device: torch.device,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random pseudo patterns and corresponding labels.

    Args:
        num_classes: Number of classes to generate pseudo patterns for.
        num_pseudo_patterns: Number of pseudo patterns per class.
        input_shape: Shape of the input data (channel, height, width).
        device: Device to move the data to (e.g., 'cuda' or 'cpu').

    Returns:
        pseudo_patterns: Tensor of randomly initialized pseudo patterns.
        labels: Tensor of corresponding labels.
    """

    if seed is not None:
        torch.manual_seed(seed)

    # Generate random pseudo patterns
    pseudo_patterns = torch.randn(
        (num_classes * num_pseudo_patterns, *input_shape),
        requires_grad=True,
        device=device
    )

    # Generate corresponding labels
    labels = torch.tensor(
        [i for i in range(num_classes) for _ in range(num_pseudo_patterns)],
        device=device
    )

    return pseudo_patterns, labels


def optimize_pseudo_patterns(pseudo_patterns, labels, model, device, iterations, lr, momentum) -> Dict[int, torch.Tensor]:
    """
    Generates and optimizes pseudo patterns at the server.
    
    Args:
        global_model: The global model at the server.
        device: Device to perform computations on (e.g., 'cuda' or 'cpu').
        num_classes: Number of classes in the dataset.
        num_pseudo_patterns: Number of pseudo patterns to generate per class.
        input_shape: Shape of the input images.
        iterations: Number of optimization iterations.
        lr: Learning rate for optimization.
        momentum: Momentum for the optimizer.

    Returns:
        pseudo_patterns: Optimized pseudo patterns as a tensor.
        logits: Dictionary containing logits for each class.
    """
    
    model = model.to(device)
    model.eval()

    optimizer = torch.optim.SGD([pseudo_patterns], lr=lr, momentum=momentum)

    for _ in range(iterations):
        optimizer.zero_grad()
        outputs = model(pseudo_patterns)
        losses = -outputs[torch.arange(len(labels)), labels]
        loss = losses.mean()
        loss.backward()
        optimizer.step()

    num_classes = torch.unique(labels).numel()
    logits = model(pseudo_patterns).detach().cpu()
    logits_dict = {label: logits[labels == label].mean(dim=0) for label in range(num_classes)}

    return logits_dict


def apply_pseudo_patterns_to_client(
    client_model: torch.nn.Module,
    pseudo_patterns: torch.Tensor,
    labels: List[int],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """
    Applies server-provided pseudo patterns to the client model to compute logits.
    
    Args:
        client_model: The client's model.
        pseudo_patterns: Pseudo patterns provided by the server.
        labels: List of labels corresponding to the pseudo patterns.
        device: Device to perform computations on.

    Returns:
        client_logits: Dictionary of logits for each label, generated by the client model.
    """
    client_model = client_model.to(device)
    labels = labels.to(device)
    client_model.eval()
    client_logits = {}

    # Move pseudo patterns to device
    pseudo_patterns = pseudo_patterns.to(device)

    # Forward pass through the client model
    outputs = client_model(pseudo_patterns)
    # outputs = copy.deepcopy(client_model(pseudo_patterns))

    # Collect logits per label
    labels = labels.detach().cpu().numpy()
    for idx, label in enumerate(labels):
        if label not in client_logits:
            client_logits[label] = []
        client_logits[label].append(outputs[idx].detach())

    # Average logits for each label
    client_logits = {label: torch.stack(logits).mean(dim=0).cpu() for label, logits in client_logits.items()}
    # client_logits = {label: logits.detach().cpu() for label, logits in client_logits.items()}

    return client_logits


def server_to_client_alignment(global_logits_dict, client_logits_dict, device):
    dissimilarity_scores = {}
    for client_id, client_logits in client_logits_dict.items():
        distances = []
        for label, server_logit in global_logits_dict.items():
            if label in client_logits:
                server_logit = server_logit.to(device)
                client_logit = client_logits[label].to(device)
                # Compute squared Euclidean distance
                dist = torch.sum((server_logit - client_logit) ** 2)
                distances.append(dist.item())  # TODO: limit the number of dis to sum to the (n-f), all the distances need to be calculated, sorted, sliced and summed.
        distances.sort()
        print("####################WWWWWW", distances)
        sum_distance = sum(distances)
        dissimilarity_scores[client_id] = sum_distance
    return dissimilarity_scores


def client_to_client_alignment(client_logits_dict, device):
    dissimilarity_scores = {}
    client_ids = list(client_logits_dict.keys())
    for client_id in client_ids:
        distances = []
        for other_client_id in client_ids:
            if client_id == other_client_id:
                continue
            for label, client_logit in client_logits_dict[client_id].items():
                if label in client_logits_dict[other_client_id]:
                    other_client_logit = client_logits_dict[other_client_id][label].to(device)
                    client_logit = client_logit.to(device)
                    # Compute squared Euclidean distance
                    dist = torch.sum((client_logit - other_client_logit) ** 2)
                    distances.append(dist.item())  # TODO: limit the number of dis to sum to the (n-f), all the distances need to be calculated, sorted, sliced and summed.
        distances.sort()
        print("####################WWWWWW", distances)
        # sum_distance = sum(distances[:-num_byzantine])
        sum_distance = sum(distances)
        dissimilarity_scores[client_id] = sum_distance
    return dissimilarity_scores
