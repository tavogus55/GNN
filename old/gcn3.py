import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        """
        Two-layer GCN model:
        - First layer: size = hidden_size (128 as per paper).
        - Second layer: size = num_classes (for classification tasks).
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Training function (no mini-batching)
def train(model, data, optimizer, epochs):
    model.train()
    for epoch in range(epochs):  # Number of epochs
        optimizer.zero_grad()
        out = model(data)
        # Compute loss using only training nodes
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss: {loss.item():.4f}")

# Evaluation function
def test(model, data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # Predicted class for each node
    test_mask = data.test_mask
    correct = pred[test_mask].eq(data.y[test_mask]).sum().item()
    acc = correct / test_mask.sum().item()
    return acc

def main():
    # Ask the user for inputs
    dataset_name = input("Select dataset (Cora/Citeseer/PubMed): ").strip()
    num_experiments = int(input("Enter the number of times to repeat the experiment: ").strip())



    # Variables to track metrics
    accuracies = []
    efficiencies = []

    for exp in range(num_experiments):

        print(f"\nRunning Experiment {exp + 1}...")
        start_time = time.time()  # Start timing from data load to training completion

        # Load the dataset once
        dataset = Planetoid(root=f"./data/{dataset_name}", name=dataset_name)
        data = dataset[0]

        # Define model parameters
        num_features = dataset.num_node_features
        num_classes = dataset.num_classes
        hidden_size = 128  # First layer size as per paper
        epochs = 100
        learning_rate = 0.001


        # Initialize the model and optimizer
        model = GCN(num_features, hidden_size, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train and evaluate
        train(model, data, optimizer, epochs)
        accuracy = test(model, data)

        # Calculate efficiency
        end_time = time.time()
        total_time = end_time - start_time
        efficiency = total_time / epochs

        # Store metrics
        accuracies.append(accuracy)
        efficiencies.append(efficiency)

        print(f"Experiment {exp + 1}: Test Accuracy: {accuracy:.4f}, Efficiency: {efficiency:.4f} seconds/epoch")

    # Compute overall averages
    avg_accuracy = sum(accuracies) / num_experiments
    avg_efficiency = sum(efficiencies) / num_experiments

    print("\n===== Final Results =====")
    print(f"Average Test Accuracy: {avg_accuracy:.4f}")
    print(f"Average Efficiency: {avg_efficiency:.4f} seconds/epoch")

if __name__ == "__main__":
    main()
