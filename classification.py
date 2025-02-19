import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv
from utils import *
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
        return F.log_softmax(x, dim=1)

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

# Training function (no mini-batching)
def train(model, data, optimizer, epochs, train_mask):
    model.train()
    for epoch in range(epochs):  # Number of epochs
        optimizer.zero_grad()
        out = model(data)
        # Compute loss using only training nodes
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss: {loss.item():.4f}")

# Evaluation function
def test(model, data, test_mask):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # Predicted clasgcs for each node
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

    cuda_num, dataset_decision, selected_model, exp_times = set_arg_parser()

    device = torch.device(f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu')

    # Variables to track metrics
    accuracies = []
    efficiencies = []
    run_times = []

    for exp in range(exp_times):

        print(f"\nRunning Experiment {exp + 1}...")

        dataset, selected_dataset, start_time, train_mask, test_mask = get_dataset_data(dataset_decision)

        # Load the dataset once

        data = dataset[0].to(device)

        # Define model parameters
        num_features = dataset.num_node_features
        num_classes = dataset.num_classes
        hidden_size = CONFIG["hidden_size"]  # First layer size as per paper
        epochs = CONFIG["epochs"]
        learning_rate = CONFIG["learning_rate"]
        weight_decay = CONFIG["weight_decay"]

        # Initialize the model and optimizer
        if selected_model == "GCN":
            model = GCN(num_features, hidden_size, num_classes).to(device)
        elif selected_model == "SGC":
            model = SGC(num_features, hidden_size, num_classes).to(device)
        else:
            print("Invalid model type. Please select GCN or SGC.")
            return

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Train and evaluate
        train(model, data, optimizer, epochs, train_mask)
        accuracy = test(model, data, test_mask)

        # Calculate efficiency
        end_time = time.time()
        total_time = end_time - start_time
        efficiency = total_time / epochs

        # Store metrics
        accuracies.append(accuracy)
        run_times.append(total_time)
        efficiencies.append(efficiency)

        torch.cuda.empty_cache()
        gc.collect()

        print(f"Experiment {exp + 1}: Test Accuracy: {accuracy:.4f}, Time: {total_time:.4f} "
              f"Efficiency: {efficiency:.4f} seconds/epoch")

    # Compute overall averages
    avg_accuracy = sum(accuracies) / exp_times
    avg_efficiency = sum(efficiencies) / exp_times
    avg_run_time = sum(run_times) / exp_times

    print("\n===== Final Results =====")
    print(f"Experiment count: {exp_times}")
    print(f"Model Type: {selected_model}")
    print(f"Dataset Name: {dataset_decision}")
    print(f"All Accuracies: {efficiencies}")
    print(f"All Run Times: {run_times}")
    print(f"All Efficiencies: {accuracies}")
    print(f"Average Accuracies: {avg_accuracy:.4f}")
    print(f"Average Run Times: {avg_run_time:.4f} seconds")
    print(f"Average Efficiency: {avg_efficiency:.4f} seconds/epoch")


if __name__ == "__main__":

    main()
