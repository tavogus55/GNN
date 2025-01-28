import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import numpy as np

# Load Cora dataset
dataset = Planetoid(root='./data/Cora', name='Cora')

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Load data
data = dataset[0]

# Initialize model, optimizer, and hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluate clustering accuracy
def clustering_accuracy(pred, true):
    cost_matrix = -np.eye(len(np.unique(true)))[true] @ np.eye(len(np.unique(pred)))[pred].T
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    acc = accuracy_score(true, [col_ind[label] for label in pred])
    return acc

# Training the GCN model
for epoch in range(100):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Perform clustering on the embeddings
model.eval()
with torch.no_grad():
    embeddings = model(data).cpu().numpy()

kmeans = KMeans(n_clusters=dataset.num_classes, random_state=0).fit(embeddings)
predicted_labels = kmeans.labels_

# Compute accuracy
ground_truth_labels = data.y.cpu().numpy()
accuracy = clustering_accuracy(predicted_labels, ground_truth_labels)
print(f'Clustering Accuracy: {accuracy:.4f}')
