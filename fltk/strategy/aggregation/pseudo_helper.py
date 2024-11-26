from typing import Tuple, List, Dict
from torch.optim import Adam
import torch


def optimize_single_batch(
    model,
    labels: List[int],
    device: str,
    input_shape: Tuple = (1, 28, 28),
    iterations: int = 100,
    lr: float = 0.5,
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

    # Detach abd convert to a list of nmpy arrays
    pseudo_pattern_list = pseudo_patterns.detach().cpu().unbind(0)
    return pseudo_pattern_list
