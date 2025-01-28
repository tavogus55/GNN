import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Load the Cora dataset
dataset = Planetoid(root="data/Planetoid", name="Cora")

# Define the GCN model with two layers
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim1, hidden_dim2, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.fc = torch.nn.Linear(hidden_dim2, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x


# Training and evaluation function
def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dataset[0].to(device)

    # Model and optimizer
    model = GCN(
        num_features=dataset.num_node_features,
        hidden_dim1=128,
        hidden_dim2=64,
        num_classes=dataset.num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # Training settings
    num_epochs = 100

    # Training
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        out = model(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    efficiency = total_time / num_epochs

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
        acc = correct / data.test_mask.sum().item()

    print(f"Classification Accuracy: {acc:.4f}")
    print(f"Efficiency (time per iteration): {efficiency:.4f} seconds")


if __name__ == "__main__":
    train_and_evaluate()
