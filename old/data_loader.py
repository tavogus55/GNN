import numpy as np
import scipy.sparse as sp
import scipy.io as scio
import pickle as pkl
import networkx as nx
import sys
# from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
import json
from torch_geometric.datasets import (Planetoid, Reddit, Flickr, FacebookPagePage, Actor, LastFMAsia, DeezerEurope,
                                      Amazon, Yelp)

# from old.sgc2 import test_mask


def get_data(data_dir, name):
    dataset = None
    test_mask = None
    train_mask = None
    if name == 'Cora':
        dataset = Planetoid(root=f'{data_dir}{name}', name=name)
        train_mask = dataset.train_mask
        test_mask = dataset.test_mask
    elif name == 'CiteSeer':
        dataset = Planetoid(root=f'{data_dir}{name}', name=name)
        train_mask = dataset.train_mask
        test_mask = dataset.test_mask
    elif name == 'PubMed':
        dataset = Planetoid(root=f'{data_dir}{name}', name=name)
        train_mask = dataset.train_mask
        test_mask = dataset.test_mask
    elif name == 'LastFMAsia':
        dataset = LastFMAsia(root=f'{data_dir}{name}')
        _, _, _, train_mask, _, test_mask = load_lastfmasia_dataset(name)
    elif name == 'FacebookPagePage':
        dataset = FacebookPagePage(root=f'{data_dir}{name}')
        _, _, _, train_mask, _, test_mask = load_facebook_pagepage_dataset(name)
    elif name == 'DeezerEurope':
        dataset = DeezerEurope(root=f'{data_dir}{name}')
        _, _, _, train_mask, _, test_mask = load_deezereurope_dataset(name)
    elif name == 'Amazon Photo' or name == 'Amazon Computers':
        dataset_path = f'{data_dir}{name.split()[0]}'
        dataset_name = name.split()[0]
        dataset_type = name.split()[1]
        dataset = Amazon(root=dataset_path, name=dataset_type)
        _, _, _, train_mask, _, test_mask = load_amazon_dataset(dataset_name, dataset_type)
    elif name == 'Actor':
        dataset = Actor(root=f'{data_dir}{name}')
        _, _, _, train_mask, _, test_mask = load_actor_dataset(name)
    elif name == 'Flickr':
        dataset = Flickr(root=f'{data_dir}{name}')
        _, _, _, train_mask, _, test_mask = load_flickr_data(name)
    elif name == 'Reddit':
        dataset = Reddit(root=f'{data_dir}{name}')

    return dataset, train_mask, test_mask


def load_facebook_pagepage_dataset(dataset):
    """
    Loads the FacebookPagePage dataset from the specified path.

    :param dataset_path: Path to the directory containing the raw dataset (e.g., "FacebookPagePage/raw").
    :return: adjacency (sparse matrix), features (numpy array), labels (numpy array), train_mask, val_mask, test_mask
    """
    # Load the raw data
    data = np.load(f"data/{dataset}/raw/facebook.npz", allow_pickle=True)

    # Extract edges, features, and target (labels)
    edges = data["edges"]  # Edge list
    features = data["features"]  # Node features
    labels = data["target"]  # Node labels

    # Create adjacency matrix from edge list
    num_nodes = features.shape[0]
    adjacency = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    ).tocsc()

    # Assuming no train/val/test masks in this dataset, split manually (if needed)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    # Example: First 70% for training, next 15% for validation, last 15% for testing
    train_mask[: int(0.7 * num_nodes)] = True
    val_mask[int(0.7 * num_nodes): int(0.85 * num_nodes)] = True
    test_mask[int(0.85 * num_nodes):] = True

    return adjacency, features, labels, train_mask, val_mask, test_mask


def load_actor_dataset(dataset_path):
    """
    Loads the Actor dataset from the specified path.

    :param dataset_path: Path to the directory containing the raw dataset (e.g., "Actor/raw").
    :return: adjacency (sparse matrix), features (numpy array), labels (numpy array)
    """
    # File paths
    edges_file = f"data/{dataset_path}/raw/out1_graph_edges.txt"
    features_labels_file = f"data/{dataset_path}/raw/out1_node_feature_label.txt"

    # Load edges (tab-separated values)
    edges = np.loadtxt(edges_file, dtype=int, delimiter="\t", skiprows=1)

    # Process features and labels manually due to mixed delimiter
    features = []
    labels = []
    max_feature_length = 0  # Track the maximum length of feature vectors

    with open(features_labels_file, "r") as f:
        lines = f.readlines()[1:]  # Skip the header line
        for line in lines:
            parts = line.strip().split("\t")  # Split by tab
            feature_values = list(map(float, parts[1].split(",")))  # Split features by comma
            label = int(parts[-1])  # Last column is the label

            features.append(feature_values)
            labels.append(label)

            # Update maximum feature length
            max_feature_length = max(max_feature_length, len(feature_values))

    # Pad features to the maximum feature length
    padded_features = np.zeros((len(features), max_feature_length), dtype=np.float32)
    for i, feature_row in enumerate(features):
        padded_features[i, :len(feature_row)] = feature_row

    # Convert labels to numpy array
    labels = np.array(labels, dtype=np.int64)  # Ensure int64 for compatibility with PyTorch

    # Create adjacency matrix
    num_nodes = len(features)
    adjacency = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    ).tocsc()

    # Generate train/validation/test masks manually
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    # Example: First 70% for training, next 15% for validation, last 15% for testing
    train_mask[: int(0.7 * num_nodes)] = True
    val_mask[int(0.7 * num_nodes): int(0.85 * num_nodes)] = True
    test_mask[int(0.85 * num_nodes):] = True

    return adjacency, padded_features, labels, train_mask, val_mask, test_mask


def load_amazon_dataset(dataset_name, dataset_type):
    """
    Loads the Amazon dataset from the specified path.

    :param dataset_path: Path to the directory containing the raw dataset (e.g., "Amazon/Computers/raw").
    :param dataset_type: Type of dataset, e.g., "Computers" or "Photos".
    :return: adjacency (sparse matrix), features (numpy array), labels (numpy array), train_mask, val_mask, test_mask
    """
    # Load the raw data
    data = np.load(f"data/{dataset_name}/{dataset_type}/raw/amazon_electronics_{dataset_type.lower()}.npz", allow_pickle=True)

    # Extract adjacency matrix components
    adj_data = data["adj_data"]
    adj_indices = data["adj_indices"]
    adj_indptr = data["adj_indptr"]
    adj_shape = tuple(data["adj_shape"])
    adjacency = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)

    # Extract features
    attr_data = data["attr_data"]
    attr_indices = data["attr_indices"]
    attr_indptr = data["attr_indptr"]
    attr_shape = tuple(data["attr_shape"])
    features = sp.csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape).todense()  # Ensure dense matrix

    # Extract labels
    labels = np.array(data["labels"], dtype=np.int64)  # Convert labels to int64 for compatibility

    # Create train, validation, and test masks
    num_nodes = features.shape[0]
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    # Example split: First 70% for training, next 15% for validation, last 15% for testing
    train_mask[: int(0.7 * num_nodes)] = True
    val_mask[int(0.7 * num_nodes): int(0.85 * num_nodes)] = True
    test_mask[int(0.85 * num_nodes):] = True

    features = np.array(features, dtype=np.float32)

    return adjacency, features, labels, train_mask, val_mask, test_mask

def load_lastfmasia_dataset(dataset):
    """
    Loads the LastFMAsia dataset from the specified path.

    :param dataset_path: Path to the directory containing the raw dataset (e.g., "LastFMAsia/raw").
    :return: adjacency (sparse matrix), features (numpy array), labels (numpy array), train_mask, val_mask, test_mask
    """
    # Load the raw data
    data = np.load(f"data/{dataset}/raw/lastfm_asia.npz", allow_pickle=True)

    # Extract edges, features, and target (labels)
    edges = data["edges"]  # Edge list
    features = data["features"]  # Node features
    labels = data["target"]  # Node labels

    # Create adjacency matrix from edge list
    num_nodes = features.shape[0]
    adjacency = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    ).tocsc()

    # Assuming no train/val/test masks in this dataset, split manually (if needed)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    # Example: First 70% for training, next 15% for validation, last 15% for testing
    train_mask[: int(0.7 * num_nodes)] = True
    val_mask[int(0.7 * num_nodes): int(0.85 * num_nodes)] = True
    test_mask[int(0.85 * num_nodes):] = True

    return adjacency, features, labels, train_mask, val_mask, test_mask

# def load_ogbn_dataset(dataset_n):
#     # Load the OGBN-Arxiv dataset
#     dataset_name = f'ogbn-{dataset_n}'
#     dataset = PygNodePropPredDataset(name=dataset_name, root='data/')
#     data = dataset[0]  # Get the graph data object
#     split_idx = dataset.get_idx_split()  # Get train/val/test splits
#
#     # Adjacency matrix
#     full_adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
#
#     # Features (node features)
#     features = data.x.numpy()  # Convert PyTorch tensor to numpy array
#
#     # Labels
#     labels = data.y.squeeze().numpy()  # Convert to 1D array
#
#     # Train/validation/test indices
#     train_index = split_idx['train']
#     val_index = split_idx['valid']
#     test_index = split_idx['test']
#
#     return full_adj, data.num_nodes, features, labels, train_index, val_index, test_index

def load_deezereurope_dataset(dataset):
    """
    Loads the DeezerEurope dataset from the specified path.

    :param dataset_path: Path to the directory containing the raw dataset (e.g., "DeezerEurope/raw").
    :return: adjacency (sparse matrix), features (numpy array), labels (numpy array), train_mask, val_mask, test_mask
    """
    # Load the raw data
    data = np.load(f"data/{dataset}/raw/deezer_europe.npz", allow_pickle=True)

    # Extract edges, features, and target (labels)
    edges = data["edges"]  # Edge list
    features = data["features"]  # Node features
    labels = data["target"]  # Node labels

    # Create adjacency matrix from edge list
    num_nodes = features.shape[0]
    adjacency = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    ).tocsc()

    # Assuming no train/val/test masks in this dataset, split manually (if needed)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    # Example: First 70% for training, next 15% for validation, last 15% for testing
    train_mask[: int(0.7 * num_nodes)] = True
    val_mask[int(0.7 * num_nodes): int(0.85 * num_nodes)] = True
    test_mask[int(0.85 * num_nodes):] = True

    return adjacency, features, labels, train_mask, val_mask, test_mask

def load_flickr_data(dataset):
    """
    Loads the Flickr dataset from the given file structure.

    Returns:
        adj: Sparse adjacency matrix (scipy.sparse.csr_matrix).
        features: Feature matrix (numpy.ndarray).
        labels: Labels (numpy.ndarray).
        train_mask: Training mask (numpy.ndarray).
        val_mask: Validation mask (numpy.ndarray).
        test_mask: Test mask (numpy.ndarray).
    """
    # Paths to the raw Flickr data
    raw_dir = f"./data/{dataset}/raw/"

    # Load data
    adj_full = sp.load_npz(raw_dir + "adj_full.npz")
    features = np.load(raw_dir + "feats.npy")
    with open(raw_dir + "class_map.json") as f:
        class_map = json.load(f)
    with open(raw_dir + "role.json") as f:
        roles = json.load(f)

    # Convert class_map to labels
    labels = np.array([class_map[str(i)] for i in range(len(class_map))])

    # Create train, val, and test masks
    num_nodes = len(labels)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    train_mask[roles["tr"]] = True
    val_mask[roles["va"]] = True
    test_mask[roles["te"]] = True

    # Create adjacency matrix
    adj = nx.adjacency_matrix(nx.from_scipy_sparse_matrix(adj_full))

    return adj, features, labels, train_mask, val_mask, test_mask