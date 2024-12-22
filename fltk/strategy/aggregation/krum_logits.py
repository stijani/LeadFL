import copy
import torch
from typing import Dict, List
from scipy.spatial.distance import cosine
from types import SimpleNamespace
import sys
sys.path.append("./")
from fltk.strategy.helper_pseudo_logits import generate_real_samples, generate_pseudo_patterns, optimize_pseudo_patterns, apply_pseudo_patterns_to_client, server_to_client_alignment, client_to_client_alignment, normalize_client_sample_quantities


def krum_logits(
    com_round,
    config,
    client_sizes,
    parameters: Dict[str, Dict[int, torch.Tensor]],
    model: torch.nn.Module,
    device: torch.device,
    num_clients_to_select: int = 1
) -> List[int]:
    """
    Selects clients based on a Krum-like approach by comparing either their logits with the server's logits
    or among themselves for dissimilarity.
    
    Args:
        config: Configuration dictionary (currently unused).
        client_sizes: List of client sizes (currently unused).
        parameters: Dictionary of client model state dicts by client ID.
        model: Global model from the previous round.
        device: Device to perform computations on.
        use_server_alignment: Boolean to indicate whether to use client-to-server dissimilarity (True) 
                              or client-to-client dissimilarity (False).
    
    Returns:
        aggr_params: Aggregated parameters from selected clients.
    """

    client_sizes_norm = normalize_client_sample_quantities(client_sizes)

    # Generate pseudo patterns
    num_label_classes = config.num_label_classes
    num_pseudo_patterns_per_label = config.num_pseudo_patterns_per_label
    # image_shape = config.image_shape
    image_shape = (config.input_channels, config.input_height, config.input_width)
    lr = config.pseudo_lr
    iter = config.pseudo_iterations
    momentum = config.pseudo_momentum
    # num_clients_to_select = config.num_clients_to_select
    use_server_alignment = config.use_server_alignment
    dataset_name = config.pseudo_real_dataset_name

    # generate pseudo pattern or real images for rehearsal/KD
    if config.use_real_images:
        pseudo_patterns, labels = generate_real_samples(num_label_classes, num_pseudo_patterns_per_label, image_shape, device, dataset_name)
    else:
        pseudo_patterns, labels = generate_pseudo_patterns(num_label_classes, num_pseudo_patterns_per_label, image_shape, device)

    global_logits_dict, server_optimized_pseudo_patterns = optimize_pseudo_patterns(copy.deepcopy(pseudo_patterns), labels, model, device, iter, lr, momentum)
    # global_logits_dict, server_optimized_pseudo_patterns = optimize_pseudo_patterns(copy.deepcopy(pseudo_patterns), labels, model, device, iter, lr, momentum, use_softmax=True)
    # global_logits_dict = optimize_pseudo_patterns(copy.deepcopy(pseudo_patterns), labels, model, device, iter, lr, momentum)

    client_logits_dict = {}
    for client_id, client_params in parameters.items():
        client_model = model
        client_model.load_state_dict(copy.deepcopy(client_params), strict=True)
        # client_logits_dict[client_id] = apply_pseudo_patterns_to_client(client_model, pseudo_patterns, labels, device)
        # client_logits_dict[client_id] = apply_pseudo_patterns_to_client(client_model, server_optimized_pseudo_patterns, labels, device)
        ###############################
        client_logits, _ = optimize_pseudo_patterns(copy.deepcopy(pseudo_patterns), labels, client_model, device, iter, lr, momentum)
        # client_logits, _ = optimize_pseudo_patterns(copy.deepcopy(pseudo_patterns), labels, client_model, device, iter, lr, momentum, use_softmax=True)
        client_logits_dict[client_id] = client_logits
        ##############################

    dissimilarity_scores = {}

    # THIS DISIMILARITY IS COMPUTED BY CLIENTS
    if use_server_alignment:  # use server to client similarity criteriion
        dissimilarity_scores = server_to_client_alignment(global_logits_dict, client_logits_dict, device)

    # THIS DISIMILARITY IS COMPUTED BY SERVER
    else:
        dissimilarity_scores = client_to_client_alignment(client_logits_dict, device)
    # ####################################################

    # ALTERNATIVELY: server can also use DBSCAN to cluster the clients rather than distance based similarity
    selected_client_ids = sorted(dissimilarity_scores, key=dissimilarity_scores.get)[:num_clients_to_select]
    print(f"############### Dissimilarity scores: {dissimilarity_scores}")
    print(f"############### Selected Client IDs: {selected_client_ids}")

    # Aggregate parameters from selected clients
    best_parameters = [parameters[client_id] for client_id in selected_client_ids]
    aggr_params = {}
    for name in best_parameters[0].keys():
        aggr_params[name] = sum([param[name].data for param in best_parameters]) / len(best_parameters)

    return aggr_params, selected_client_ids


# ######## Testing #########
if __name__ == "__main__":
    import sys
    sys.path.append("../LeadFL")
    from fltk.nets.fashion_mnist_cnn import FashionMNISTCNN

    global_model = FashionMNISTCNN()
    num_label_classes = 10
    num_pseudo_patterns_per_label = 10
    image_shape = [1, 28, 28]
    dataset_name = "cifar10"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_samples, labels_ = generate_real_samples(num_label_classes, num_pseudo_patterns_per_label, image_shape, device, dataset_name)
    pseudo_patterns, labels = generate_pseudo_patterns(num_label_classes, num_pseudo_patterns_per_label, image_shape, device)

    print(f"##### Real: {real_samples.shape}, {labels_.shape}")
    print(f"##### Pseudo: {pseudo_patterns.shape}, {labels.shape}")

    global_logits_dict = optimize_pseudo_patterns(pseudo_patterns, labels, global_model, device, 100, 0.5, 0.9)
    print(f"##### Global Logits: {global_logits_dict}")

    client_0_logits_dict = apply_pseudo_patterns_to_client(FashionMNISTCNN(), pseudo_patterns, labels, device)
    client_1_logits_dict = apply_pseudo_patterns_to_client(FashionMNISTCNN(), pseudo_patterns, labels, device)
    client_2_logits_dict = apply_pseudo_patterns_to_client(FashionMNISTCNN(), pseudo_patterns, labels, device)

    # print(global_logits_dict[0])
    # print(client_0_logits_dict[0])
    # print(client_1_logits_dict[0])
    # print(client_2_logits_dict[0])

    parameters = {
        "client_0": FashionMNISTCNN().state_dict(),
        "client_1": FashionMNISTCNN().state_dict(),
        "client_2": FashionMNISTCNN().state_dict(),
    }

    global_model = FashionMNISTCNN()
    num_clients_to_select = 2

    config = {"num_label_classes": 10,
              "num_pseudo_patterns_per_label": 10,
              "image_shape": [1, 28, 28],
              "pseudo_lr": 0.5,
              "pseudo_iterations": 100,
              "pseudo_momentum": 0.9
              }
    config = SimpleNamespace(**config)

    client_sizes = []

    aggr_params = krum_logits(
        config,
        client_sizes,
        parameters,
        global_model,
        num_clients_to_select,
        device,
    )

    print(aggr_params)