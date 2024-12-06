from typing import Tuple, List
import torch
import numpy as np
from torch.optim import Adam


def tensor2uint8(tensor):
    """
    Convert a PyTorch tensor to a uint8 NumPy array.
    """
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = (tensor * 255).to(torch.uint8)
    return tensor.cpu().numpy()


def optimize_single_batch(
    model,
    input_shape: Tuple,
    iterations: int,
    lr: float,
    labels: List[int],
    device: str,
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
    pseudo_patterns = torch.rand(
        (batch_size, *input_shape), requires_grad=True, device=device
    )

    # Create an optimizer for the current batch
    optimizer = Adam([pseudo_patterns], lr=lr)

    # Labels tensor for the batch
    labels_tensor = torch.tensor(labels, device=device)

    # Optimize the pseudo patterns for the current batch
    for _ in range(iterations):
        optimizer.zero_grad()

        # Forward pass: Compute model outputs for the batch
        outputs = model(pseudo_patterns)

        # Compute loss: maximize activation for each label
        losses = -outputs[torch.arange(batch_size), labels_tensor]
        loss = losses.mean()  # Average loss across the batch
        loss.backward()

        # Update the pseudo patterns
        optimizer.step()

    # Detach and return the optimized pseudo patterns
    return pseudo_patterns.detach()


def generate_large_number_of_pseudo_patterns(
    input_shape: Tuple,
    iterations: int,
    lr: float,
    labels: List[int],
    batch_size: int,
    model,
    device: str,
    to_uint: bool = False,
):
    """
    Generate a large number of pseudo patterns for multiple labels by processing in batches.

    Args:
        input_shape: Shape of the input tensor (channel, height, width).
        iterations: Number of optimization iterations.
        lr: Learning rate for the optimizer.
        labels: A list of all labels for which patterns need to be generated.
        batch_size: The number of pseudo patterns to process per batch.
        model: The PyTorch model used for computing activations.
        device: The device to run the computations on ('cuda' or 'cpu').

    Returns:
        A tensor of all optimized pseudo patterns.
    """
    model = model.to(device)

    # List to hold all pseudo patterns
    all_pseudo_patterns = []

    # Process labels in batches
    for i in range(0, len(labels), batch_size):
        # Get the current batch of labels
        batch_labels = labels[i : i + batch_size]

        # Optimize pseudo patterns for the current batch
        batch_patterns = optimize_single_batch(
            model=model,
            input_shape=input_shape,
            iterations=iterations,
            lr=lr,
            labels=batch_labels,
            device=device,
        )
        # Append the batch patterns to the results
        all_pseudo_patterns.extend(batch_patterns.cpu().unbind(0))
    if to_uint:
        all_pseudo_patterns = [tensor2uint8(tensor) for tensor in all_pseudo_patterns]
    else:
        all_pseudo_patterns = [tensor.numpy() for tensor in all_pseudo_patterns]
    return np.array(all_pseudo_patterns), np.array(labels)