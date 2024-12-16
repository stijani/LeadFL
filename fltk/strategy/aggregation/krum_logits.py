import copy
import torch
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine
from types import SimpleNamespace
import sys
sys.path.append("./")
from fltk.strategy.helper_pseudo_logits import generate_real_samples, generate_pseudo_patterns, optimize_pseudo_patterns, apply_pseudo_patterns_to_client, server_to_client_alignment, client_to_client_alignment, normalize_client_sample_quantities


# def generate_real_samples(
#     num_classes,
#     num_pseudo_patterns,
#     input_shape,
#     device,
#     seed: int = 42,
# ):
#     """
#     Generate real samples as pseudo patterns using existing dataset.

#     Args:
#         config: Configuration object.
#         model: PyTorch model (unused here but kept for interface compatibility).
#         state_dict: Model state dictionary (unused here but kept for interface compatibility).
#         num_classes: Number of classes in the dataset.
#         device: Device to move the data to (e.g., 'cuda' or 'cpu').
#         seed: Random seed for reproducibility.
#         num_pseudo_patterns: Number of pseudo patterns to sample per class.
#         input_shape: Shape of the input data.

#     Returns:
#         pseudo_patterns: Tensor containing the real samples as pseudo patterns.
#         labels: Tensor containing the corresponding labels.
#     """
#     # Set random seed for reproducibility
#     if seed is not None:
#         torch.manual_seed(seed)

#     # Load the dataset
#     x_npy = np.load("/home/stijani/projects/dataset/mnist-fashion/test_features.npy")
#     y_npy = np.load("/home/stijani/projects/dataset/mnist-fashion/test_labels.npy")

#     sampled_features = []
#     sampled_labels = []

#     # Sample the specified number of patterns per class
#     for cls in range(num_classes):
#         class_indices = np.where(y_npy == cls)[0]
#         selected_indices = np.random.choice(class_indices, size=num_pseudo_patterns, replace=False)

#         # Append the features and labels for the selected indices
#         sampled_features.extend(x_npy[selected_indices])
#         sampled_labels.extend(y_npy[selected_indices])

#     # Convert sampled features and labels to tensors
#     pseudo_patterns = torch.tensor(sampled_features, dtype=torch.float32).view(-1, *input_shape).to(device)
#     labels = torch.tensor(sampled_labels, dtype=torch.long).to(device)

#     return pseudo_patterns, labels


# def generate_pseudo_patterns(
#     num_classes: int,
#     num_pseudo_patterns: int,
#     input_shape: Tuple[int, int, int],
#     device: torch.device,
#     seed: int = 42,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Generate random pseudo patterns and corresponding labels.

#     Args:
#         num_classes: Number of classes to generate pseudo patterns for.
#         num_pseudo_patterns: Number of pseudo patterns per class.
#         input_shape: Shape of the input data (channel, height, width).
#         device: Device to move the data to (e.g., 'cuda' or 'cpu').

#     Returns:
#         pseudo_patterns: Tensor of randomly initialized pseudo patterns.
#         labels: Tensor of corresponding labels.
#     """

#     if seed is not None:
#         torch.manual_seed(seed)

#     # Generate random pseudo patterns
#     pseudo_patterns = torch.randn(
#         (num_classes * num_pseudo_patterns, *input_shape),
#         requires_grad=True,
#         device=device
#     )

#     # Generate corresponding labels
#     labels = torch.tensor(
#         [i for i in range(num_classes) for _ in range(num_pseudo_patterns)],
#         device=device
#     )

#     return pseudo_patterns, labels


# def optimize_pseudo_patterns(pseudo_patterns, labels, model, device, iterations, lr, momentum) -> Dict[int, torch.Tensor]:
#     """
#     Generates and optimizes pseudo patterns at the server.
    
#     Args:
#         global_model: The global model at the server.
#         device: Device to perform computations on (e.g., 'cuda' or 'cpu').
#         num_classes: Number of classes in the dataset.
#         num_pseudo_patterns: Number of pseudo patterns to generate per class.
#         input_shape: Shape of the input images.
#         iterations: Number of optimization iterations.
#         lr: Learning rate for optimization.
#         momentum: Momentum for the optimizer.

#     Returns:
#         pseudo_patterns: Optimized pseudo patterns as a tensor.
#         logits: Dictionary containing logits for each class.
#     """
#     model = model.to(device)
#     model.eval()

#     optimizer = torch.optim.SGD([pseudo_patterns], lr=lr, momentum=momentum)

#     for _ in range(iterations):
#         optimizer.zero_grad()
#         outputs = model(pseudo_patterns)
#         losses = -outputs[torch.arange(len(labels)), labels]
#         loss = losses.mean()
#         loss.backward()
#         optimizer.step()

#     num_classes = torch.unique(labels).numel()
#     logits = model(pseudo_patterns).detach().cpu()
#     logits_dict = {label: logits[labels == label].mean(dim=0) for label in range(num_classes)}

#     return logits_dict


# def apply_pseudo_patterns_to_client(
#     client_model: torch.nn.Module,
#     pseudo_patterns: torch.Tensor,
#     labels: List[int],
#     device: torch.device,
# ) -> Dict[int, torch.Tensor]:
#     """
#     Applies server-provided pseudo patterns to the client model to compute logits.
    
#     Args:
#         client_model: The client's model.
#         pseudo_patterns: Pseudo patterns provided by the server.
#         labels: List of labels corresponding to the pseudo patterns.
#         device: Device to perform computations on.

#     Returns:
#         client_logits: Dictionary of logits for each label, generated by the client model.
#     """
#     client_model = client_model.to(device)
#     labels = labels.to(device)
#     client_model.eval()
#     client_logits = {}

#     # Move pseudo patterns to device
#     pseudo_patterns = pseudo_patterns.to(device)

#     # Forward pass through the client model
#     outputs = client_model(pseudo_patterns)

#     # Collect logits per label
#     labels = labels.detach().cpu().numpy()
#     for idx, label in enumerate(labels):
#         if label not in client_logits:
#             client_logits[label] = []
#         client_logits[label].append(outputs[idx].detach())

#     # Average logits for each label
#     client_logits = {label: torch.stack(logits).mean(dim=0).cpu() for label, logits in client_logits.items()}
#     # client_logits = {label: logits.detach().cpu() for label, logits in client_logits.items()}

#     return client_logits


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

    # get server logits - server does:
    # 1. generates pseudo pattern (pp) for the next round
    # 2. optimize the pp
    # 3. generates generates server logits using the optimized pp as input
    # 4. broad cast the pp to the clients
    global_logits_dict = optimize_pseudo_patterns(pseudo_patterns, labels, model, device, iter, lr, momentum)

    # Each client:
    # 1. compute its own logit for the current iteration using it's local model (after training)
    # 2. anonymised its computed weigths
    # 3. shares its weights with the server:
    #    i) also shares and its computed logits (is computing dimisalarity based on client2client)
    #    ii) compute dissimilarity score between its logit and the server on if dissilarity is server2client
    # share its logit or dissimlarity score with the server 
    client_logits_dict = {}
    for client_id, client_params in parameters.items():
        client_model = model
        client_model.load_state_dict(copy.deepcopy(client_params), strict=True)
        client_logits_dict[client_id] = apply_pseudo_patterns_to_client(client_model, pseudo_patterns, labels, device)

    dissimilarity_scores = {}

    # #############################################################
    # THIS DISIMILARITY IS COMPUTED BY CLIENTS
    if use_server_alignment:  # use server to client similarity criteriion
    #     for client_id, client_logits in client_logits_dict.items():
    #         total_distance = 0.0
    #         for label, server_logit in global_logits_dict.items():
    #             if label in client_logits:
    #                 server_logit = server_logit.to(device)
    #                 client_logit = client_logits[label].to(device)
    #                 # Compute squared Euclidean distance
    #                 distance = torch.sum((server_logit - client_logit) ** 2)
    #                 total_distance += distance.item()
    #         dissimilarity_scores[client_id] = total_distance
        dissimilarity_scores = server_to_client_alignment(global_logits_dict, client_logits_dict, device) #############
    # ###############################################################

    # THIS DISIMILARITY IS COMPUTED BY SERVER
    else:
        # ##################################################
        # use client to client similarity criteriion
        # client_ids = list(client_logits_dict.keys())
        # for client_id in client_ids:
        #     total_distance = 0.0
        #     for other_client_id in client_ids:
        #         if client_id == other_client_id:
        #             continue
        #         for label, client_logit in client_logits_dict[client_id].items():
        #             if label in client_logits_dict[other_client_id]:
        #                 other_client_logit = client_logits_dict[other_client_id][label].to(device)
        #                 client_logit = client_logit.to(device)
        #                 # Compute squared Euclidean distance
        #                 distance = torch.sum((client_logit - other_client_logit) ** 2)
        #                 total_distance += distance.item()
        #     dissimilarity_scores[client_id] = total_distance
        dissimilarity_scores = client_to_client_alignment(client_logits_dict, device)
    # ####################################################

    # Sort clients by dissimilarity score (ascending order) and select top clients
    # 1. Server recieves or compute dissimilarity scores for each client
    # 2. aggregates model based on the most similar clients
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





# import copy
# import torch
# import numpy as np
# from typing import Dict, List, Tuple
# from scipy.spatial.distance import cosine
# from types import SimpleNamespace


# def generate_real_samples(
#     num_classes,
#     num_pseudo_patterns,
#     input_shape,
#     device,
#     seed: int = 42,
# ):
#     """
#     Generate real samples as pseudo patterns using existing dataset.

#     Args:
#         config: Configuration object.
#         model: PyTorch model (unused here but kept for interface compatibility).
#         state_dict: Model state dictionary (unused here but kept for interface compatibility).
#         num_classes: Number of classes in the dataset.
#         device: Device to move the data to (e.g., 'cuda' or 'cpu').
#         seed: Random seed for reproducibility.
#         num_pseudo_patterns: Number of pseudo patterns to sample per class.
#         input_shape: Shape of the input data.

#     Returns:
#         pseudo_patterns: Tensor containing the real samples as pseudo patterns.
#         labels: Tensor containing the corresponding labels.
#     """
#     # Set random seed for reproducibility
#     if seed is not None:
#         torch.manual_seed(seed)

#     # Load the dataset
#     x_npy = np.load("/home/stijani/projects/dataset/mnist-fashion/test_features.npy")
#     y_npy = np.load("/home/stijani/projects/dataset/mnist-fashion/test_labels.npy")

#     sampled_features = []
#     sampled_labels = []

#     # Sample the specified number of patterns per class
#     for cls in range(num_classes):
#         class_indices = np.where(y_npy == cls)[0]
#         selected_indices = np.random.choice(class_indices, size=num_pseudo_patterns, replace=False)

#         # Append the features and labels for the selected indices
#         sampled_features.extend(x_npy[selected_indices])
#         sampled_labels.extend(y_npy[selected_indices])

#     # Convert sampled features and labels to tensors
#     pseudo_patterns = torch.tensor(sampled_features, dtype=torch.float32).view(-1, *input_shape).to(device)
#     labels = torch.tensor(sampled_labels, dtype=torch.long).to(device)

#     return pseudo_patterns, labels


# def generate_pseudo_patterns(
#     num_classes: int,
#     num_pseudo_patterns: int,
#     input_shape: Tuple[int, int, int],
#     device: torch.device,
#     seed: int = 42,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Generate random pseudo patterns and corresponding labels.

#     Args:
#         num_classes: Number of classes to generate pseudo patterns for.
#         num_pseudo_patterns: Number of pseudo patterns per class.
#         input_shape: Shape of the input data (channel, height, width).
#         device: Device to move the data to (e.g., 'cuda' or 'cpu').

#     Returns:
#         pseudo_patterns: Tensor of randomly initialized pseudo patterns.
#         labels: Tensor of corresponding labels.
#     """

#     if seed is not None:
#         torch.manual_seed(seed)

#     # Generate random pseudo patterns
#     pseudo_patterns = torch.randn(
#         (num_classes * num_pseudo_patterns, *input_shape),
#         requires_grad=True,
#         device=device
#     )

#     # Generate corresponding labels
#     labels = torch.tensor(
#         [i for i in range(num_classes) for _ in range(num_pseudo_patterns)],
#         device=device
#     )

#     return pseudo_patterns, labels


# def optimize_pseudo_patterns(pseudo_patterns, labels, model, device, iterations, lr, momentum) -> Dict[int, torch.Tensor]:
#     """
#     Generates and optimizes pseudo patterns at the server.
    
#     Args:
#         global_model: The global model at the server.
#         device: Device to perform computations on (e.g., 'cuda' or 'cpu').
#         num_classes: Number of classes in the dataset.
#         num_pseudo_patterns: Number of pseudo patterns to generate per class.
#         input_shape: Shape of the input images.
#         iterations: Number of optimization iterations.
#         lr: Learning rate for optimization.
#         momentum: Momentum for the optimizer.

#     Returns:
#         pseudo_patterns: Optimized pseudo patterns as a tensor.
#         logits: Dictionary containing logits for each class.
#     """
#     model = model.to(device)
#     model.eval()

#     optimizer = torch.optim.SGD([pseudo_patterns], lr=lr, momentum=momentum)

#     for _ in range(iterations):
#         optimizer.zero_grad()
#         outputs = model(pseudo_patterns)
#         losses = -outputs[torch.arange(len(labels)), labels]
#         loss = losses.mean()
#         loss.backward()
#         optimizer.step()

#     num_classes = torch.unique(labels).numel()
#     logits = model(pseudo_patterns).detach().cpu()
#     logits_dict = {label: logits[labels == label].mean(dim=0) for label in range(num_classes)}

#     return logits_dict


# def apply_pseudo_patterns_to_client(
#     client_model: torch.nn.Module,
#     pseudo_patterns: torch.Tensor,
#     labels: List[int],
#     device: torch.device,
# ) -> Dict[int, torch.Tensor]:
#     """
#     Applies server-provided pseudo patterns to the client model to compute logits.
    
#     Args:
#         client_model: The client's model.
#         pseudo_patterns: Pseudo patterns provided by the server.
#         labels: List of labels corresponding to the pseudo patterns.
#         device: Device to perform computations on.

#     Returns:
#         client_logits: Dictionary of logits for each label, generated by the client model.
#     """
#     client_model = client_model.to(device)
#     labels = labels.to(device)
#     client_model.eval()
#     client_logits = {}

#     # Move pseudo patterns to device
#     pseudo_patterns = pseudo_patterns.to(device)

#     # Forward pass through the client model
#     outputs = client_model(pseudo_patterns)

#     # Collect logits per label
#     labels = labels.detach().cpu().numpy()
#     for idx, label in enumerate(labels):
#         if label not in client_logits:
#             client_logits[label] = []
#         client_logits[label].append(outputs[idx].detach())

#     # Average logits for each label
#     client_logits = {label: torch.stack(logits).mean(dim=0).cpu() for label, logits in client_logits.items()}
#     # client_logits = {label: logits.detach().cpu() for label, logits in client_logits.items()}

#     return client_logits


# def krum_logits(
#     com_round,
#     config,
#     client_sizes,
#     parameters: Dict[str, Dict[int, torch.Tensor]],
#     model: torch.nn.Module,
#     device: torch.device,
#     num_clients_to_select: int = 1
# ) -> List[int]:
#     """
#     Selects clients based on a Krum-like approach by comparing either their logits with the server's logits
#     or among themselves for dissimilarity.
    
#     Args:
#         config: Configuration dictionary (currently unused).
#         client_sizes: List of client sizes (currently unused).
#         parameters: Dictionary of client model state dicts by client ID.
#         model: Global model from the previous round.
#         device: Device to perform computations on.
#         use_server_alignment: Boolean to indicate whether to use client-to-server dissimilarity (True) 
#                               or client-to-client dissimilarity (False).
    
#     Returns:
#         aggr_params: Aggregated parameters from selected clients.
#     """

#     # Generate pseudo patterns
#     num_label_classes = config.num_label_classes
#     num_pseudo_patterns_per_label = config.num_pseudo_patterns_per_label
#     image_shape = config.image_shape
#     lr = config.pseudo_lr
#     iter = config.pseudo_iterations
#     momentum = config.pseudo_momentum
#     # num_clients_to_select = config.num_clients_to_select
#     use_server_alignment = config.use_server_alignment

#     # generate pseudo pattern or real images for rehearsal/KD
#     if config.use_real_images:
#         pseudo_patterns, labels = generate_real_samples(num_label_classes, num_pseudo_patterns_per_label, image_shape, device)
#     else:
#         pseudo_patterns, labels = generate_pseudo_patterns(num_label_classes, num_pseudo_patterns_per_label, image_shape, device)

#     # get server logits - server does:
#     # 1. generates pseudo pattern (pp) for the next round
#     # 2. optimize the pp
#     # 3. generates generates server logits using the optimized pp as input
#     # 4. broad cast the pp to the clients
#     global_logits_dict = optimize_pseudo_patterns(pseudo_patterns, labels, model, device, iter, lr, momentum)

#     # Each client:
#     # 1. compute its own logit for the current iteration using it's local model (after training)
#     # 2. anonymised its computed weigths
#     # 3. shares its weights with the server:
#     #    i) also shares and its computed logits (is computing dimisalarity based on client2client)
#     #    ii) compute dissimilarity score between its logit and the server on if dissilarity is server2client
#     # share its logit or dissimlarity score with the server 
#     client_logits_dict = {}
#     for client_id, client_params in parameters.items():
#         client_model = model
#         client_model.load_state_dict(copy.deepcopy(client_params), strict=True)
#         client_logits_dict[client_id] = apply_pseudo_patterns_to_client(client_model, pseudo_patterns, labels, device)

#     dissimilarity_scores = {}

#     # THIS DISIMILARITY IS COMPUTED BY CLIENTS
#     if use_server_alignment:  # use server to client similarity criteriion
#         for client_id, client_logits in client_logits_dict.items():
#             total_distance = 0.0
#             for label, server_logit in global_logits_dict.items():
#                 if label in client_logits:
#                     server_logit = server_logit.to(device)
#                     client_logit = client_logits[label].to(device)
#                     # Compute squared Euclidean distance
#                     distance = torch.sum((server_logit - client_logit) ** 2)
#                     total_distance += distance.item()
#             dissimilarity_scores[client_id] = total_distance

#     # THIS DISIMILARITY IS COMPUTED BY SERVER
#     else:
#         # use client to client similarity criteriion
#         client_ids = list(client_logits_dict.keys())
#         for client_id in client_ids:
#             total_distance = 0.0
#             for other_client_id in client_ids:
#                 if client_id == other_client_id:
#                     continue
#                 for label, client_logit in client_logits_dict[client_id].items():
#                     if label in client_logits_dict[other_client_id]:
#                         other_client_logit = client_logits_dict[other_client_id][label].to(device)
#                         client_logit = client_logit.to(device)
#                         # Compute squared Euclidean distance
#                         distance = torch.sum((client_logit - other_client_logit) ** 2)
#                         total_distance += distance.item()
#             dissimilarity_scores[client_id] = total_distance

#     # Sort clients by dissimilarity score (ascending order) and select top clients
#     # 1. Server recieves or compute dissimilarity scores for each client
#     # 2. aggregates model based on the most similar clients
#     # ALTERNATIVELY: server can also use DBSCAN to cluster the clients rather than distance based similarity
#     selected_client_ids = sorted(dissimilarity_scores, key=dissimilarity_scores.get)[:num_clients_to_select]
#     print(f"############### Dissimilarity scores: {dissimilarity_scores}")
#     print(f"############### Selected Client IDs: {selected_client_ids}")

#     # Aggregate parameters from selected clients
#     best_parameters = [parameters[client_id] for client_id in selected_client_ids]
#     aggr_params = {}
#     for name in best_parameters[0].keys():
#         aggr_params[name] = sum([param[name].data for param in best_parameters]) / len(best_parameters)

#     return aggr_params, selected_client_ids


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
