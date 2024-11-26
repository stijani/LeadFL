from typing import Dict, Tuple
import torch
from torch.optim import Adam
import sys

sys.path.append("./fltk/nets")
from fashion_mnist_cnn import FashionMNISTCNN


import flwr as fl
from flwr.common.parameter import parameters_to_ndarrays, ndarrays_to_parameters
import numpy as np
import json
from typing import List, Tuple, Dict
# from flwr.common import Parameters, Weights
from flwr.common import Parameters
from flwr.server.client_proxy import ClientProxy
from collections import OrderedDict
import torch
from torch import nn
# from hydra.utils import instantiate
from pseudo_helper import optimize_single_batch


def krum_pseudo(parameters: Dict[str, Dict[str, torch.Tensor]],
                model: nn.Module,
                # pseudo_patterns: Dict[str, Dict[str, np.ndarray]],
                device: torch.device,
                ) -> int:
    """
    Selects the most representative client based on pseudo patterns.
    model: model should alread be on device
    """
    # TODO:
    # 1. load the model with a clients parameter and generate pseudo for all classes
    # 2. a list of dicts [label: pseudo] for each client
    # 3. continues as below

    labels = list(range(10))
    client_pseudo_dict = {}

    for idx, client_id in enumerate(parameters.keys()):
        model = model.load_state_dict(parameters[client_id]).to(device)
        pseudo_per_label = optimize_single_batch(model, labels, device)

        # convert to dict with labels as keys, pseudo are already in the order of labels
        pseudo_per_label_dict = {lab: pseudo for lab, pseudo in enumerate(pseudo_per_label)}
        client_pseudo_dict[idx] = pseudo_per_label_dict  # client id here are there index
    
    num_clients = len(client_pseudo_dict)
    label_ids = labels

    # Initialize an array to store the total scores for each client
    total_scores = np.zeros(num_clients)

    for label_id in label_ids:
        # Compute pairwise distances for the current label
        distances = np.zeros((num_clients, num_clients))

        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                # Calculate the distance between client i and client j for the current label
                dist = np.linalg.norm(
                    client_pseudo_dict[i][label_id] - client_pseudo_dict[j][label_id]
                )
                distances[i, j] = dist
                distances[j, i] = dist

        # Calculate scores for each client for the current label
        for i in range(num_clients):
            # Sort distances from client i to others
            sorted_distances = sorted(distances[i])
            # Add the sum of distances to the closest N-2 clients to the total score
            total_scores[i] += sum(sorted_distances[:num_clients - 2])

    # Select the client with the lowest total score
    best_client_index = np.argmin(total_scores)
    return best_client_index


























# def get_single_pesudo_pattern(model, shape: Tuple, iterations, lr: float, label: int, device=None):
#     """
#     1. initialize a model
#     2. load the model with the parameter
#     3. generate pseudo pattern (one per label)
#     4. optimize each pesudo pattern
#     5. take average of all 10 pseudo patterns
#     6. return a single pseudo pattern
#     """
#     model = model.to(device)
#     pseudo_pattern = torch.rand((1, *shape), requires_grad=True, device=device)
#     optimizer = Adam([pseudo_pattern], lr=lr)

#     for _ in range(iterations):
#         optimizer.zero_grad()
#         pseudo_pattern = pseudo_pattern.to(device)
#         output = model(pseudo_pattern)
#         loss = -output[0, label]  # Maximize activation for the current class
#         loss.backward()
#         optimizer.step()
#     return pseudo_pattern.detach().clone()[0]


# def get_averaged_pseudo_patterns(client_model_params, shape: Tuple, iterations, lr: float, num_classes: int, device=None):
#     """
#     get pseudo pattern for each client
#     """
#     pseudo_patterns_per_client = {}
#     for idx, state_dict in client_model_params.items():
#         model = FashionMNISTCNN() ##########
#         model.load_state_dict(state_dict, strict=True)
#         per_label_pesudo_patterns = []
#         for label in range(num_classes):
#             label_pseudo = get_single_pesudo_pattern(model, shape, iterations, lr, label, device)
#             per_label_pesudo_patterns.append(label_pseudo)
#         pseudo_patterns_per_client[idx] = torch.mean(torch.stack(per_label_pesudo_patterns), dim=0)
#     return pseudo_patterns_per_client


# def krum_with_pseudo(parameters: Dict[str, Dict[str, torch.Tensor]], shape: Tuple=(1, 28, 28), iterations: int=100, lr: float=0.1, num_classes: int=10, device: str="cuda") -> Dict[str, torch.Tensor]:
#     """
#     multi krum passed parameters.
#     Use the model parameters to compute pseudo patterns, estimat krum distance based on pseudo patterns return updated parameters based on this.

#     :param parameters: nn model named parameters with client index
#     :type parameters: list
#     """
#     pseudo_patterns = get_averaged_pseudo_patterns(parameters, shape, iterations, lr, num_classes, device) # these are representatative pseudo-patterns rather that client parameters.

#     candidate_num = 6  # we select the top n (candidate_num) lowest ditances measured for this client from every other client, the candiate's distance becomes the sum of these top n ditances. Krum used 6
#     distances = {}
#     tmp_parameters = {}
#     for idx, pseudo_pattern in pseudo_patterns.items():
#         distance = []
#         for _idx, _pseudo_pattern in pseudo_patterns.items():
#             # dis = [torch.norm((parameter[name].data - _parameter[name].data).float()) ** 2 for name in parameter.keys()]
#             dis = torch.norm((pseudo_pattern - _pseudo_pattern).float()) ** 2
#             distance.append(dis)
#             tmp_parameters[idx] = parameters[idx]
#         # distance = sum(torch.Tensor(distance).float())
#         distance.sort()
#         # print("benign distance: " + str(distance))
#         distances[idx] = sum(distance[:candidate_num])  # sum the top n (candidate_num) lowest distances

#     sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))  # sort the distances from lowest to highest
#     candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][:1]  # pick the candidate with the minimum distance (if single Krum), for multi-Krum, pick the top n clients and average their parameters (could also be weighted by the individual distances)
#     print("#################1", sorted_distance)

#     new_params = {}
#     for name in candidate_parameters[0].keys():
#         new_params[name] = sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters) # for Krum, we are just taking average over a single client (same as just returning the parameters of that client directly)

#     return new_params


# def krum(parameters: Dict[str, Dict[str, torch.Tensor]], sizes: Dict[str, int]) -> Dict[str, torch.Tensor]:
#     """
#     multi krum passed parameters.

#     :param parameters: nn model named parameters with client index
#     :type parameters: list
#     """

#     candidate_num = 6
#     distances = {}
#     tmp_parameters = {}
#     for idx, parameter in parameters.items():
#         distance = []
#         for _idx, _parameter in parameters.items():
#             dis = [torch.norm((parameter[name].data - _parameter[name].data).float()) ** 2 for name in parameter.keys()]
#             distance.append(sum(dis))
#             tmp_parameters[idx] = parameter
#         # distance = sum(torch.Tensor(distance).float())
#         distance.sort()
#         # print("benign distance: " + str(distance))
#         distances[idx] = sum(distance[:candidate_num])

#     sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))
#     candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][:1]
#     print("#################2", sorted_distance)

#     new_params = {}
#     for name in candidate_parameters[0].keys():
#         new_params[name] = sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters)

#     return new_params


# if __name__ == "__main__":
#     # test the median function
#     # model = FashionMNISTCNN()
#     parameters = {idx: FashionMNISTCNN().state_dict() for idx in range(10)}
#     device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#     new_params_pseudo = krum_with_pseudo(parameters, shape=(1, 28, 28), iterations=100, lr=0.1, num_classes=10, device=device)
#     new_params_krum = krum(parameters, None)