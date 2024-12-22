import torch
import numpy as np
from typing import Dict
import numpy as np
import torch
from collections import Counter
import hdbscan
from sklearn.metrics import accuracy_score, precision_score, recall_score


def clustering_not_used(stored_gradients, client_index, clients_per_round=10, stored_rounds=10):
    client_index = np.array(client_index)
    labels, major_label = clustering_(stored_gradients[-stored_rounds*clients_per_round:])
    aggregation_id = np.where(labels[-clients_per_round:] == major_label)[0]
    clients_id = client_index[-clients_per_round:][aggregation_id]
    return aggregation_id, clients_id

def clustering_(stored_gradients):
    # transform the stored gradients into a numpy array
    stored_gradients = np.concatenate(stored_gradients, axis=0)
    #clf = DBSCAN(eps=0.5, min_samples=5).fit(stored_gradients)
    clf = hdbscan.HDBSCAN(min_cluster_size=3).fit(stored_gradients)
    major_label = find_majority_label(clf)
    labels = clf.labels_
    return labels, major_label

def find_majority_label(clf):
    counts = Counter(clf.labels_)
    major_label = max(counts, key=counts.get)
    # major_id = set(major_id.reshape(-1))
    return major_label

def calculate_cluster_metrics(client_index, mal_index, candidates):
    y_true = [1 if i in mal_index else 0 for i in client_index]
    y_pred = [0 if i in candidates else 1 for i in client_index]
    # calculate the metrics
    # acc score
    acc = accuracy_score(y_true, y_pred)
    # precision score
    pre = precision_score(y_true, y_pred)
    # recall score
    rec = recall_score(y_true, y_pred)
    return acc, pre, rec


def clustering(clients_params, sizes):
    """
    Cluster clients using HDBSCAN, aggregate parameters for the largest cluster,
    and return the client IDs of clients in that cluster.

    :param clients_params: Dict of client IDs mapped to parameter tensors
    :return: Aggregated parameters, largest cluster ID, and client IDs in the best cluster
    """
    # Step 1: Concatenate client parameters into a single tensor
    client_ids = list(clients_params.keys())
    concatenated_params = [
        torch.cat([param.flatten() for param in params.values()], dim=0).numpy()
        for params in clients_params.values()
    ]
    
    # Convert to a 2D NumPy array for clustering
    data_for_clustering = np.stack(concatenated_params)

    # Step 2: Apply HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    cluster_labels = clusterer.fit_predict(data_for_clustering)
    
    # Step 3: Organize clients into clusters
    cluster_dict = {}
    for client_id, label in zip(client_ids, cluster_labels):
        if label != -1:  # Exclude noise points (label = -1)
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(client_id)
    
    # Check if there are any valid clusters
    if not cluster_dict:
        print("No valid clusters found. All points are classified as noise.")
        return None

    # Step 4: Find the largest cluster
    largest_cluster_id = max(cluster_dict, key=lambda x: len(cluster_dict[x]))
    largest_cluster_client_ids = cluster_dict[largest_cluster_id]
    
    # Step 5: Aggregate parameters by averaging within the largest cluster
    aggregated_params = {}
    num_clients = len(largest_cluster_client_ids)
    
    # Sum up parameters across all clients in the largest cluster
    for client_id in largest_cluster_client_ids:
        client_params = clients_params[client_id]
        for name, param in client_params.items():
            if name not in aggregated_params:
                aggregated_params[name] = torch.zeros_like(param)
            aggregated_params[name] += param
    
    # Average the summed parameters
    for name in aggregated_params:
        aggregated_params[name] = aggregated_params[name].float() / float(num_clients)
    
    return aggregated_params, largest_cluster_client_ids


# Example input: Simulated client parameters (grouped for valid clusters)
if __name__ == "__main__ ":
    clients_params = {
        "client1": {
            "param1": torch.tensor([[1.1, 2.3, 3.6], [4.2, 5.1, 6.3]], dtype=torch.float32),
        },
        "client2": {
            "param1": torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=torch.float32),
        },
        "client3": {
            "param1": torch.tensor([[13.0, 14.0, 15.0], [16.0, 17.12, 18.2]], dtype=torch.float32),
        },
        "client4": {
            "param1": torch.tensor([[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]], dtype=torch.float32),
        },
        "client5": {
            "param1": torch.tensor([[25.0, 26.0, 27.0], [28.0, 29.12, 30.1]], dtype=torch.float32),
        },
        "client6": {
            "param1": torch.tensor([[31.0, 32.0, 33.0], [34.0, 35.0, 36.0]], dtype=torch.float32),
        }
    }

    # Perform clustering and aggregate parameters for the best cluster
    aggregated_params, client_ids_in_cluster = aggregate_best_cluster(clients_params)

    # Print the results if clustering is successful
    if aggregated_params:
        print(f"Best Cluster ID: {best_cluster_id}")
        print("Clients in the Best Cluster:")
        print(client_ids_in_cluster)
        print("Aggregated Parameters:")
        for name, value in aggregated_params.items():
            print(f"  {name}: {value}")


# if __name__ == '__main__':
#     #test the clustering function
#     stored_gradients = [torch.Tensor([[1.1, 2.3, 3.6], [4.2, 5.1, 6.3]]), torch.Tensor([[7, 8, 9], [10, 11, 12]]), torch.Tensor([[13, 14, 15], [16, 17.12, 18.2]]), torch.Tensor([[19, 20, 21], [22, 23, 24]]), torch.Tensor([[25, 26, 27], [28, 29.12, 30.1]]), torch.Tensor([[31, 32, 33], [34, 35, 36]])]
#     print(clustering(stored_gradients))
