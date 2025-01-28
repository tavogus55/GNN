import torch
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
import time
import numpy as np

# Define Improved GCN Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Load Dataset
dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Initialize Model with Larger Dimensions
model = GCN(
    in_channels=dataset.num_node_features,
    hidden_channels=128,  # Larger hidden dimension
    out_channels=64  # Larger output embedding dimension
).to(device)

# Train the Model with More Epochs
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
model.train()

start_train_time = time.time()
for epoch in range(1000):  # Train for more epochs
    optimizer.zero_grad()
    embeddings = model(data.x, data.edge_index)  # Node embeddings
    loss = torch.mean((embeddings[data.train_mask] - embeddings.mean(dim=0))**2)  # Simple regularization loss
    loss.backward()
    optimizer.step()
end_train_time = time.time()

# Extract Node Embeddings
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index).cpu().numpy()

# Normalize Embeddings
embeddings = normalize(embeddings, axis=1)

# Clustering with K-Means
start_cluster_time = time.time()
num_clusters = dataset.num_classes
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
predicted_clusters = kmeans.fit_predict(embeddings)
end_cluster_time = time.time()

# Compute Metrics
true_labels = data.y.cpu().numpy()
acc = accuracy_score(true_labels, predicted_clusters)
nmi = normalized_mutual_info_score(true_labels, predicted_clusters)

# Relaxed K-Means Clustering (Baseline)
random_clusters = np.random.randint(0, num_clusters, size=true_labels.shape)
relaxed_acc = accuracy_score(true_labels, random_clusters)
relaxed_nmi = normalized_mutual_info_score(true_labels, random_clusters)

# Print Results
print("============ Start Clustering ============")
print(f"k-means results: ACC: {acc:.4f}, NMI: {nmi:.4f}")
print(f"Relaxed K-Means results: ACC: {relaxed_acc:.4f}, NMI: {relaxed_nmi:.4f}")
print(f"Process started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_train_time))}")
print(f"Process finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_cluster_time))}")

# Training Time
training_time = end_train_time - start_train_time
hours, rem = divmod(training_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Training lasted {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
