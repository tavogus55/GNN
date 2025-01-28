import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv
from torch_geometric.loader import NeighborLoader
from utils import get_selected_dataset, get_dataset_data
import json
import gc

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

# Define the SGC model
class SGC(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        """
        Two-layer SGC model:
        - First layer: size = hidden_size (128 as per paper).
        - Second layer: size = num_classes (for classification tasks).
        """
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, hidden_size)
        self.conv2 = SGConv(hidden_size, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

# Training function with batching support
def train(model, data, optimizer, epochs, test_mask, use_batching=False, batch_size=128):
    model.train()
    if use_batching:
        loader = NeighborLoader(data, num_neighbors=[25, 10], batch_size=batch_size, input_nodes=data.train_mask)
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                out = model(batch)
                loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss: {total_loss:.4f}")
    else:
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out[test_mask], data.y[test_mask])
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss: {loss.item():.4f}")

# Evaluation function
def test(model, data, test_mask):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # Predicted class for each node
    correct = pred[test_mask].eq(data.y[test_mask]).sum().item()
    acc = correct / test_mask.sum().item()
    return acc

def main():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Current Device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available. Running on CPU.")

    with open("./config.json", "r") as file:
        CONFIG = json.load(file)

    DATASET_OPTIONS = CONFIG["datasets"]
    selected_dataset = get_selected_dataset(DATASET_OPTIONS)
    dataset, selected_dataset, start_time, train_mask, test_mask = get_dataset_data(selected_dataset)

    # Ask the user for inputs
    model_type = input("Select model (GCN/SGC): ").strip().upper()
    num_experiments = int(input("Enter the number of times to repeat the experiment: ").strip())

    # Variables to track metrics
    accuracies = []
    efficiencies = []

    for exp in range(num_experiments):
        print(f"\nRunning Experiment {exp + 1}...")

        # Load the dataset once
        data = dataset[0]

        # Define model parameters
        num_features = dataset.num_node_features
        num_classes = dataset.num_classes
        hidden_size = 128  # First layer size as per paper
        epochs = 100
        learning_rate = 0.001

        # Initialize the model and optimizer
        if model_type == "GCN":
            model = GCN(num_features, hidden_size, num_classes)
        elif model_type == "SGC":
            model = SGC(num_features, hidden_size, num_classes)
        else:
            print("Invalid model type. Please select GCN or SGC.")
            return

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Determine if batching should be used
        use_batching = selected_dataset == 11  # Use batching only for Reddit dataset
        batch_size = 512 if use_batching else None

        # Train and evaluate
        train(model, data, optimizer, epochs, test_mask, use_batching=use_batching, batch_size=batch_size)
        accuracy = test(model, data, test_mask)

        # Calculate efficiency
        end_time = time.time()
        total_time = end_time - start_time
        efficiency = total_time / epochs

        # Store metrics
        accuracies.append(accuracy)
        efficiencies.append(efficiency)

        torch.cuda.empty_cache()
        gc.collect()

        print(f"Experiment {exp + 1}: Test Accuracy: {accuracy:.4f}, Efficiency: {efficiency:.4f} seconds/epoch")

    # Compute overall averages
    avg_accuracy = sum(accuracies) / num_experiments
    avg_efficiency = sum(efficiencies) / num_experiments

    print("\n===== Final Results =====")
    print(f"All Efficiencies: {efficiencies}")
    print(f"All Accuracies: {accuracies}")
    print(f"Experiment count: {num_experiments}")
    print(f"Average Test Accuracy: {avg_accuracy:.4f}")
    print(f"Average Efficiency: {avg_efficiency:.4f} seconds/epoch")
    print(f"Model Type: {model_type}")
    print(f"Dataset Name: {selected_dataset}")

if __name__ == "__main__":
    main()
