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


def get_data(data_dir, name, type=''):
    dataset = None
    if name == 'Cora':
        dataset = Planetoid(root=f'{data_dir}{name}', name=name)
    elif name == 'CiteSeer':
        dataset = Planetoid(root=f'{data_dir}{name}', name=name)
    elif name == 'PubMed':
        dataset = Planetoid(root=f'{data_dir}{name}', name=name)
    elif name == 'LastFMAsia':
        dataset = LastFMAsia(root=f'{data_dir}{name}')
    elif name == 'FacebookPagePage':
        dataset = FacebookPagePage(root=f'{data_dir}{name}')
    elif name == 'DeezerEurope':
        dataset = DeezerEurope(root=f'{data_dir}{name}')
    elif name == 'Amazon Photo' or name == 'Amazon Computers':
        dataset = Amazon(root=f'{data_dir}{name.split()[0]}', name=name.split()[1])
    elif name == 'Actor':
        dataset = Actor(root=f'{data_dir}{name}')
    elif name == 'Flickr':
        dataset = Flickr(root=f'{data_dir}{name}')
    elif name == 'Reddit':
        dataset = Reddit(root=f'{data_dir}{name}')

    if name == 'Cora' or name == 'CiteSeer' or name == 'PubMed' or name == 'Reddit':
        return dataset, dataset.train_mask, dataset.test_mask
    elif (name == 'LastFMAsia' or name == 'FacebookPagePage'
              or name == 'DeezerEurope'
              or name == 'Amazon Photo'
              or name == 'Actor'
              or name == 'Flickr'
              or name == 'Amazon Computers'):
        return dataset, get_train_mask(name), get_test_mask(name)


def get_data_from_npz(dataset_name, dataset_filename):
    if 'Amazon' in dataset_name:
        data = np.load(f"data/{dataset_name.split()[0]}/{dataset_name.split()[1]}/raw/{dataset_filename}.npz",
                       allow_pickle=True)
    else:
        data = np.load(f"data/{dataset_name}/raw/{dataset_filename}.npz", allow_pickle=True)

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

def get_train_mask(dataset_name):
    data = ''
    if dataset_name == 'LastFMAsia':
        data = np.load(f"data/{dataset_name}/raw/lastfm_asia.npz", allow_pickle=True)
    elif dataset_name == 'FacebookPagePage':
        data = np.load(f"data/{dataset_name}/raw/facebook.npz", allow_pickle=True)
    elif dataset_name == 'DeezerEurope':
        data = np.load(f"data/{dataset_name}/raw/deezer_europe.npz", allow_pickle=True)
    elif 'Amazon' in dataset_name:
        data = np.load(f"data/{dataset_name.split()[0]}/{dataset_name.split()[1]}/raw/amazon_electronics_{dataset_name.split()[1].lower()}.npz", allow_pickle=True)

    if 'Amazon' in dataset_name:
        # Extract features
        attr_data = data["attr_data"]
        attr_indices = data["attr_indices"]
        attr_indptr = data["attr_indptr"]
        attr_shape = tuple(data["attr_shape"])
        features = sp.csr_matrix((attr_data, attr_indices, attr_indptr),
                                 shape=attr_shape).todense()  # Ensure dense matrix
    elif dataset_name == 'Actor':
        # File paths
        edges_file = f"data/{dataset_name}/raw/out1_graph_edges.txt"
        features_labels_file = f"data/{dataset_name}/raw/out1_node_feature_label.txt"
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

        features = padded_features
    elif dataset_name == 'Flickr':
        features = np.load(f"data/{dataset_name}/raw/" + "feats.npy")
    else:
        features = data["features"]

    if dataset_name == 'Flickr':
        with open(f"data/{dataset_name}/raw/" + "role.json") as f:
            roles = json.load(f)
        with open(f"data/{dataset_name}/raw/" + "class_map.json") as f:
            class_map = json.load(f)
        labels = np.array([class_map[str(i)] for i in range(len(class_map))])
        num_nodes = len(labels)
        train_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[roles["tr"]] = True
    else:
        num_nodes = features.shape[0]
        train_mask = np.zeros(num_nodes, dtype=bool)
        train_mask[: int(0.7 * num_nodes)] = True
    return train_mask

def get_test_mask(dataset_name):
    data = ''
    if dataset_name == 'LastFMAsia':
        data = np.load(f"data/{dataset_name}/raw/lastfm_asia.npz", allow_pickle=True)
    elif dataset_name == 'FacebookPagePage':
        data = np.load(f"data/{dataset_name}/raw/facebook.npz", allow_pickle=True)
    elif dataset_name == 'DeezerEurope':
        data = np.load(f"data/{dataset_name}/raw/deezer_europe.npz", allow_pickle=True)
    elif 'Amazon' in dataset_name:
        data = np.load(
            f"data/{dataset_name.split()[0]}/{dataset_name.split()[1]}/raw/amazon_electronics_{dataset_name.split()[1].lower()}.npz",
            allow_pickle=True)

    if 'Amazon' in dataset_name:
        # Extract features
        attr_data = data["attr_data"]
        attr_indices = data["attr_indices"]
        attr_indptr = data["attr_indptr"]
        attr_shape = tuple(data["attr_shape"])
        features = sp.csr_matrix((attr_data, attr_indices, attr_indptr),
                                 shape=attr_shape).todense()  # Ensure dense matrix
    elif dataset_name == 'Actor':
        # File paths
        edges_file = f"data/{dataset_name}/raw/out1_graph_edges.txt"
        features_labels_file = f"data/{dataset_name}/raw/out1_node_feature_label.txt"
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

        features = padded_features
    elif dataset_name == 'Flickr':
        features = np.load(f"data/{dataset_name}/raw/" + "feats.npy")
    else:
        features = data["features"]

    if dataset_name == 'Flickr':
        with open(f"data/{dataset_name}/raw/" + "role.json") as f:
            roles = json.load(f)
        with open(f"data/{dataset_name}/raw/" + "class_map.json") as f:
            class_map = json.load(f)
        labels = np.array([class_map[str(i)] for i in range(len(class_map))])
        num_nodes = len(labels)
        test_mask = np.zeros(num_nodes, dtype=bool)
        test_mask[roles["tr"]] = True
    else:
        num_nodes = features.shape[0]
        test_mask = np.zeros(num_nodes, dtype=bool)
        test_mask[int(0.85 * num_nodes):] = True
    return test_mask


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

    return adjacency, features, labels, train_mask, val_mask, test_mask, data

