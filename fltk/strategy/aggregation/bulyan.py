# pylint: disable=invalid-name
from typing import Dict

import numpy as np
import torch


def bulyan(config, parameters: Dict[str, Dict[str, torch.Tensor]], sizes: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """
    Implements the Bulyan aggregation strategy for robust federated learning.

    :param parameters: Dictionary of model parameters with client indices as keys.
                       Each value is a dictionary of parameter tensors.
    :param sizes: Dictionary mapping client indices to their respective data sizes.
    :return: Aggregated model parameters as a dictionary of tensors.
    """
    # for the multiKrum stage
    num_clients_to_select = config.clients_per_round - config.mal_clients_per_round
    multi_krum = num_clients_to_select  # Number of clients to consider after Krum selection

    candidate_num = 7  # Number of closest clients to consider for distance sum
    f = 3  # assuming and average if 3 mal clients per round
    beta = max(3, (config.clients_per_round - 2 * f) // 2)  # Define beta using the formula

    # Step 1: Compute pairwise distances
    distances = {}
    for client_id, parameter in parameters.items():
        distances[client_id] = []
        for _client_id, _parameter in parameters.items():
            if client_id != _client_id:
                distance = sum(
                    torch.norm((parameter[name] - _parameter[name]).float())**2
                    for name in parameter.keys()
                )
                distances[client_id].append(distance)

    # Step 2: Select top candidates using Krum
    krum_scores = {}
    for client_id, dist_list in distances.items():
        sorted_distances = sorted(dist_list)[:candidate_num]
        krum_scores[client_id] = sum(sorted_distances)

    # Sort clients by Krum scores and select the top `multi_krum`
    selected_client_ids = sorted(krum_scores, key=krum_scores.get)[:multi_krum]
    selected_params = [parameters[client_id] for client_id in selected_client_ids]

    # Step 3: Compute median for each parameter across selected clients
    medians = {}
    for name in selected_params[0].keys():
        stacked_tensors = torch.stack([param[name] for param in selected_params])
        medians[name] = torch.median(stacked_tensors, dim=0).values

    # Step 4: Average beta closest parameters to the median
    aggregated_params = {}
    count_tracking_layers = 0
    for name in selected_params[0].keys():
        ###################################
        # Skip non-learnable parameters
        # print("##############", name)
        # raise ValueError
        if "num_batches_tracked" in name:
            aggregated_params[name] = selected_params[0][name]  # Copy from any client
            # print("##############1", name, selected_params[0][name])
            # print("##############-1", name, selected_params[0][name])
            count_tracking_layers += 1
            continue
        ###################################
        # Compute distances of each candidate's layer parameter from the median
        median = medians[name]
        distances_to_median = [
            torch.norm((param[name] - median).float())**2 for param in selected_params
        ]

        # Select beta closest parameters to the median
        closest_indices = torch.argsort(torch.tensor(distances_to_median))[:beta]
        closest_tensors = torch.stack([selected_params[i][name] for i in closest_indices])

        # Average the beta closest parameters for this layer
        # print("#################", selected_params[closest_indices[0]])
        aggregated_params[name] = torch.mean(closest_tensors, dim=0)
        # print("##############", count_tracking_layers)
    return aggregated_params, selected_client_ids


# def bulyan(parameters: Dict[str, Dict[str, torch.Tensor]], sizes: Dict[str, int]) -> Dict[str, torch.Tensor]:
def bulyan_original(parameters: Dict[str, Dict[str, torch.Tensor]], useless=None) -> Dict[str, torch.Tensor]:
    """
    bulyan passed parameters.

    :param parameters: nn model named parameters with client index
    :type parameters: list
    :param uselss: added to ensure that the function has a similar signature to others  ########
    :type parameters: None
    """
    multi_krum = 5
    candidate_num = 7
    distances = {}
    tmp_parameters = {}
    for idx, parameter in parameters.items():
        distance = []
        for _idx, _parameter in parameters.items():
            dis = [torch.norm((parameter[name].data - _parameter[name].data).float())**2 for name in parameter.keys()]
            distance.append(sum(dis))
            tmp_parameters[idx] = parameter
        # distance = sum(torch.Tensor(distance).float())
        distance.sort()
        distances[idx] = sum(distance[:candidate_num])

    sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))
    candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][1:multi_krum-1]
    new_params = {}
    for name in candidate_parameters[0].keys():
        new_params[name] = sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters)

    # return new_params
    normalized_client_distances = None  # so that the output is always compatible
    return new_params, normalized_client_distances


if __name__ == "__main__":
    # test the median function
    params = {'1': {'1': torch.Tensor([[1, 2, 3], [4, 5, 6]]), '2': torch.Tensor([[7, 8, 9], [10, 11, 12]])},
              '2': {'1': torch.Tensor([[13, 14, 15], [16, 17, 18]]), '2': torch.Tensor([[19, 20, 21], [22, 23, 24]])},
              '3': {'1': torch.Tensor([[25, 26, 27], [28, 29, 30]]), '2': torch.Tensor([[31, 32, 33], [34, 35, 36]])},
                '4': {'1': torch.Tensor([[37, 38, 39], [40, 41, 42]]), '2': torch.Tensor([[43, 44, 45], [46, 47, 48]])},
                '5': {'1': torch.Tensor([[49, 50, 51], [52, 53, 54]]), '2': torch.Tensor([[55, 56, 57], [58, 59, 60]])},
                '6': {'1': torch.Tensor([[61, 62, 63], [64, 65, 66]]), '2': torch.Tensor([[67, 68, 69], [70, 71, 72]])},
                '7': {'1': torch.Tensor([[73, 74, 75], [76, 77, 78]]), '2': torch.Tensor([[79, 80, 81], [82, 83, 84]])},
                '8': {'1': torch.Tensor([[85, 86, 87], [88, 89, 90]]), '2': torch.Tensor([[91, 92, 93], [94, 95, 96]])},
                '9': {'1': torch.Tensor([[97, 98, 99], [100, 101, 102]]), '2': torch.Tensor([[103, 104, 105], [106, 107, 108]])},
                '10': {'1': torch.Tensor([[109, 110, 111], [112, 113, 114]]), '2': torch.Tensor([[115, 116, 117], [118, 119, 120]])}}
    # sizes = {'1': 2, '2': 2, '3': 3}
    # print(bulyan(params, sizes))
    sizes = {'1': 2, '2': 2, '3': 3}
    print(bulyan(params))

