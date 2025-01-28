import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SGC(torch.nn.Module):
    def __init__(self, dataset, use_precomputed=True):
        super().__init__()
        self.use_precomputed = use_precomputed
        if not use_precomputed:
            self.conv = GCNConv(dataset.num_node_features, dataset.num_classes)
        else:
            # Add a linear layer to classify precomputed features
            self.linear = torch.nn.Linear(dataset.num_node_features, dataset.num_classes)

    def forward(self, data):
        if self.use_precomputed:
            # Use precomputed features with linear layer
            x = data.x
            x = self.linear(x)
        else:
            # Perform single convolution step if not precomputed
            x, edge_index = data.x, data.edge_index
            x = self.conv(x, edge_index)

        return F.log_softmax(x, dim=1)


# Precompute features outside the model
def precompute_features(data, num_node_features, num_classes, k=2, device='cpu'):
    # Move the GCNConv layer to the correct device
    conv = GCNConv(num_node_features, num_node_features).to(device)
    x, edge_index = data.x.to(device), data.edge_index.to(device)  # Move data to the device

    for _ in range(k - 1):  # Perform k-1 propagations
        x = conv(x, edge_index)

    data.x = x  # Update the features in the dataset
    return data
