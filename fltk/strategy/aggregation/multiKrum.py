# pylint: disable=invalid-name
from typing import Dict

import numpy as np
import torch


def multiKrum(parameters: Dict[str, Dict[str, torch.Tensor]], sizes: Dict[str, int], config) -> Dict[str, torch.Tensor]:
    """
    multi krum passed parameters.

    :param parameters: nn model named parameters with client index
    :type parameters: list
    """
    multi_krum = config.clients_per_round - config.mal_clients_per_round
    candidate_num = int(0.7 * config.clients_per_round)
    distances = {}
    tmp_parameters = {}
    pre_distance = []
    for idx, parameter in parameters.items():
        distance = []
        for _idx, _parameter in parameters.items():

            dis = 0
            for key in parameter.keys():
                dis = dis + (torch.norm(_parameter[key].float() - parameter[key].float()) ** 2)
            distance.append(dis)

            # pre_distance.append(sum(pre_dis))
            tmp_parameters[idx] = parameter
        # pre_dis = [torch.norm((_parameter[name].data - previous_weight[name].data).float()) for name in parameter.keys()]
        # pre_distance.append(sum(pre_dis))
        distance.sort()

        distances[idx] = sum(distance[:candidate_num+1])


    sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))
    candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][:multi_krum]
    selected_client_ids = [idx for idx in sorted_distance.keys()][:multi_krum]
    new_params = {}
    for name in candidate_parameters[0].keys():
        new_params[name] = sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters)

    return new_params, selected_client_ids



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
    sizes = {'1': 2, '2': 2, '3': 3}
    print(multiKrum(params, sizes))
    # test torch stack

    mal_list = np.random.choice([1,2,3], 0, replace=False)
    print(mal_list)

