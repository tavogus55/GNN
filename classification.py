import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv
from utils import *
import json
import gc
from torch.nn import Linear
import logging

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


class SGC(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_classes, K=2):
        """
        Standard SGC model:
        - First SGConv layer: K-step feature propagation.
        - One non-linearity (ReLU).
        - Final Linear layer for classification.
        """
        super(SGC, self).__init__()
        self.conv = SGConv(num_features, hidden_size, K=K, cached=True)  # Feature propagation
        self.fc = Linear(hidden_size, num_classes)  # Classification layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv(x, edge_index)  # Propagate features
        x = F.relu(x)  # Single non-linearity
        x = self.fc(x)  # Final classification
        return F.log_softmax(x, dim=1)  # Standard output format


def train(model, data, optimizer, epochs, train_mask, logger=None):
    model.train()
    for epoch in range(epochs):  # Number of epochs
        optimizer.zero_grad()
        out = model(data)
        # Compute loss using only training nodes
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            logger.debug(f"Epoch {epoch}: Loss: {loss.item():.4f}")


def test(model, data, test_mask):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # Predicted clasgcs for each node
    correct = pred[test_mask].eq(data.y[test_mask]).sum().item()
    acc = correct / test_mask.sum().item()
    return acc


def main(device, dataset_decision, selected_model, exp_times, logger=None):

    # Variables to track metrics
    accuracies = []
    efficiencies = []
    run_times = []

    for exp in range(exp_times):

        logger.info(f"\nRunning Experiment {exp + 1}...")

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
            logger.warn("Invalid model type. Please select GCN or SGC.")
            return

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Train and evaluate
        train(model, data, optimizer, epochs, train_mask, logger=logger)
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

        logger.info(f"Experiment {exp + 1}: Test Accuracy: {accuracy:.4f}, Time: {total_time:.4f} "
              f"Efficiency: {efficiency:.4f} seconds/epoch")

    # Compute overall averages
    avg_accuracy = sum(accuracies) / exp_times
    avg_efficiency = sum(efficiencies) / exp_times
    avg_run_time = sum(run_times) / exp_times

    logger.info("\n===== Final Results =====")
    logger.info(f"Experiment count: {exp_times}")
    logger.info(f"Model Type: {selected_model}")
    logger.info(f"Dataset Name: {dataset_decision}")
    logger.info(f"All Accuracies: {accuracies}")
    logger.info(f"All Run Times: {run_times}")
    logger.info(f"All Efficiencies: {efficiencies}")
    logger.info(f"Average Accuracies: {avg_accuracy:.4f}")
    logger.info(f"Average Run Times: {avg_run_time:.4f} seconds")
    logger.info(f"Average Efficiency: {avg_efficiency:.4f} seconds/epoch")


if __name__ == "__main__":

    cuda_num, dataset_decision, selected_model, exp_times, log_path = set_arg_parser()

    log_path = f"{log_path}{selected_model}_{dataset_decision}.log"

    logger = get_logger(f"{selected_model}", log_path)


    if torch.cuda.is_available():
        logger.info("CUDA is available!")
        logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Current Device: {torch.cuda.current_device()}")
    else:
        logger.info("CUDA is not available. Running on CPU.")

    with open("./config.json", "r") as file:
        CONFIG = json.load(file)

    device = torch.device(f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu')

    main(device, dataset_decision, selected_model, exp_times, logger=logger)
