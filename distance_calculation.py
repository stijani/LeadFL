from scipy.spatial.distance import cosine, mahalanobis
from scipy.stats import wasserstein_distance
import numpy as np
import torch

def euclidean_distance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    Compute the Euclidean distance between two tensors.
    """
    return np.linalg.norm(tensor1.cpu().numpy().flatten() - tensor2.cpu().numpy().flatten())

def cosine_distance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    Compute the cosine distance between two tensors.
    """
    return cosine(tensor1.cpu().numpy().flatten(), tensor2.cpu().numpy().flatten())

def manhattan_distance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    Compute the Manhattan (L1) distance between two tensors.
    """
    return np.sum(np.abs(tensor1.cpu().numpy().flatten() - tensor2.cpu().numpy().flatten()))

def mahalanobis_distance(tensor1: torch.Tensor, tensor2: torch.Tensor, inv_cov_matrix: np.ndarray) -> float:
    """
    Compute the Mahalanobis distance between two tensors.
    """
    return mahalanobis(tensor1.cpu().numpy().flatten(), tensor2.cpu().numpy().flatten(), inv_cov_matrix)

def wasserstein_distance_wrapper(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """
    Compute the Wasserstein distance between two tensors.
    """
    return wasserstein_distance(tensor1.cpu().numpy().flatten(), tensor2.cpu().numpy().flatten())
