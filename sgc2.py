import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from datetime import datetime
from data_loader import get_data, get_train_mask, get_test_mask

# SGC Model Definition
class SGC(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_precomputed=True):
        super(SGC, self).__init__()
        self.use_precomputed = use_precomputed
        if use_precomputed:
            self.linear = torch.nn.Linear(in_channels, out_channels)
        else:
            self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        if self.use_precomputed:
            # Use precomputed features with linear layer
            x = self.linear(x)
        else:
            # Perform single convolution step if not precomputed
            x = self.conv(x, edge_index)
        return F.log_softmax(x, dim=1)

# Precompute Features
def precompute_features(data, in_channels, k=2, device='cpu'):
    conv = GCNConv(in_channels, in_channels).to(device)  # Self-loop preserving propagation
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    for _ in range(k):
        x = conv(x, edge_index)
    return x.detach()  # Detach to avoid retaining computation graph

available_datasets = ['Cora', 'CiteSeer', 'PubMed', 'LastFMAsia', 'FacebookPagePage', 'DeezerEurope', 'Amazon', 'Actor']

# Prompt the user to select a dataset
print("Available datasets:")
for i, dataset_name in enumerate(available_datasets, start=1):
    print(f"{i}: {dataset_name}")

choice = int(input("Enter the number corresponding to the dataset you want to use: "))

if 1 <= choice <= len(available_datasets):
    selected_dataset = available_datasets[choice - 1]
    dataset = get_data('./data/', selected_dataset)

data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Precompute features
k = 2
precomputed_x = precompute_features(data, dataset.num_node_features, k=k, device=device)
data.x = precomputed_x.to(device)
data = data.to(device)

# Initialize Model, Optimizer, and Loss Function
model = SGC(dataset.num_node_features, dataset.num_classes, use_precomputed=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

start_time = datetime.now()
# Training Loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # Use precomputed features
    if selected_dataset == 'Cora' or selected_dataset == 'PubMed' or selected_dataset == 'CiteSeer':
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    elif (selected_dataset == 'LastFMAsia' or selected_dataset == 'FacebookPagePage'
          or selected_dataset == 'DeezerEurope' or selected_dataset == 'Amazon' or selected_dataset == 'Actor'):
        train_mask = get_train_mask(selected_dataset)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
out = model(data.x, data.edge_index)
pred = out.argmax(dim=1)
if selected_dataset == 'Cora' or selected_dataset == 'PubMed' or selected_dataset == 'CiteSeer':
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
elif (selected_dataset == 'LastFMAsia' or selected_dataset == 'FacebookPagePage' or selected_dataset == 'DeezerEurope'
      or selected_dataset == 'Amazon' or selected_dataset == 'Actor'):
    test_mask = get_test_mask(selected_dataset)
    correct = (pred[test_mask] == data.y[test_mask]).sum()
    acc = int(correct) / int(test_mask.sum())

print(f'Accuracy: {acc:.4f}')

finish_time = datetime.now()
print(start_time.strftime("Process started at: " + "%Y-%m-%d %H:%M:%S"))
print(finish_time.strftime("Process started at: " + "%Y-%m-%d %H:%M:%S"))
time_difference = finish_time - start_time

# Extract hours, minutes, and seconds
total_seconds = int(time_difference.total_seconds())
hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Training lasted {hours} hours, {minutes} minutes, {seconds} seconds")

