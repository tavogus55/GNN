import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv
from sklearn.cluster import KMeans
from metric import cal_clustering_metric
from utils import get_data

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_size, output_size):
        """
        Two-layer GCN model:
        - First layer: size = hidden_size (128 as per paper).
        - Second layer: size = output_size (e.g., embedding size for clustering).
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Define the SGC model
class SGC(torch.nn.Module):
    def __init__(self, num_features, hidden_size, output_size):
        """
        Two-layer SGC model:
        - First layer: size = hidden_size (128 as per paper).
        - Second layer: size = output_size (e.g., embedding size for clustering).
        """
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, hidden_size)
        self.conv2 = SGConv(hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

# Training function (no mini-batching)
def train(model, data, optimizer, epochs):
    model.train()
    for epoch in range(epochs):  # Number of epochs
        optimizer.zero_grad()
        out = model(data)  # Output embeddings
        # No supervised loss for clustering tasks
        loss = torch.tensor(0.0, requires_grad=True)  # Placeholder
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Embedding computation complete.")

# Clustering evaluation function using custom metrics
def evaluate_clustering(model, data):
    model.eval()
    embeddings = model(data).detach().cpu().numpy()

    # Perform K-Means clustering
    n_clusters = data.y.max().item() + 1  # Number of clusters = number of classes
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(embeddings)
    pred_labels = kmeans.predict(embeddings)

    # Calculate clustering metrics using the provided method
    acc, nmi = cal_clustering_metric(data.y.cpu().numpy(), pred_labels)
    return acc, nmi

def main():

    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Current Device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available. Running on CPU.")

    # Ask the user for inputs
    model_type = input("Select model (GCN/SGC): ").strip().upper()
    dataset_name = input("Select dataset (Cora/Citeseer/PubMed/Flickr/Reddit/LastFMAsia/...): ").strip()
    num_experiments = int(input("Enter the number of times to repeat the experiment: ").strip())

    # Variables to track metrics
    acc_scores = []
    nmi_scores = []
    efficiencies = []

    for exp in range(num_experiments):

        print(f"\nRunning Experiment {exp + 1}...")
        start_time = time.time()  # Start timing from data load to training completion

        # Load the dataset dynamically using utils.get_data
        dataset, train_mask, test_mask = get_data("./data/", dataset_name)
        data = dataset[0]

        # Define model parameters
        num_features = dataset.num_node_features
        output_size = 64  # Embedding size for clustering as per paper
        hidden_size = 128  # First layer size as per paper
        epochs = 100
        learning_rate = 0.001

        # Initialize the model and optimizer
        if model_type == "GCN":
            model = GCN(num_features, hidden_size, output_size)
        elif model_type == "SGC":
            model = SGC(num_features, hidden_size, output_size)
        else:
            print("Invalid model type. Please select GCN or SGC.")
            return

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train and evaluate
        train(model, data, optimizer, epochs)
        acc, nmi = evaluate_clustering(model, data)

        # Calculate efficiency
        end_time = time.time()
        total_time = end_time - start_time
        efficiency = total_time / epochs

        # Store metrics
        acc_scores.append(acc)
        nmi_scores.append(nmi)
        efficiencies.append(efficiency)

        print(f"Experiment {exp + 1}: ACC: {acc:.4f}, NMI: {nmi:.4f}, Efficiency: {efficiency:.4f} seconds/epoch")

    # Compute overall averages
    avg_acc = sum(acc_scores) / num_experiments
    avg_nmi = sum(nmi_scores) / num_experiments
    avg_efficiency = sum(efficiencies) / num_experiments

    print("\n===== Final Results =====")
    print(f"All Accuracies: {acc_scores}")
    print(f"All NMIs: {nmi_scores}")
    print(f"All Efficiencies: {efficiencies}")
    print(f"Average ACC: {avg_acc:.4f}")
    print(f"Average NMI: {avg_nmi:.4f}")
    print(f"Average Efficiency: {avg_efficiency:.4f} seconds/epoch")
    print(f"Model Type: {model_type}")
    print(f"Dataset Name: {dataset_name}")

if __name__ == "__main__":
    main()
