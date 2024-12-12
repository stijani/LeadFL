from typing import Dict, List
import torch
# from krum_logits import krum_logits

import sys
sys.path.append("./")
from fltk.strategy.aggregation.krum_logits import krum_logits


def multiKrum_logits(
    com_round,
    config,
    client_sizes,
    parameters: Dict[str, Dict[int, torch.Tensor]],
    model: torch.nn.Module,
    device: torch.device,
) -> List[int]:
    # num_clients_to_select = config.num_clients_to_select
    num_clients_to_select = config.clients_per_round - config.mal_clients_per_round
    aggr_params, selected_client_ids = krum_logits(com_round, config, client_sizes, parameters, model, device, num_clients_to_select)
    return aggr_params, selected_client_ids