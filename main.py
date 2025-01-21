from GCN import GCN
from SGC import SGC, precompute_features
from torch_geometric.datasets import (Planetoid, Reddit, Flickr, FacebookPagePage, Actor, LastFMAsia, DeezerEurope,
                                      Amazon, Yelp)
import torch
import torch.nn.functional as F
from datetime import datetime


available_datasets = ['Cora', 'CiteSeer', 'PubMed', 'Reddit', 'Flickr', 'FacebookPagePage', 'Actor',
                      'LastFMAsia', 'DeezerEurope', 'Amazon Computers', 'Amazon Photo', 'Yelp']

# Prompt the user to select a dataset
print("Available datasets:")
for i, dataset_name in enumerate(available_datasets, start=1):
    print(f"{i}: {dataset_name}")

choice = int(input("Enter the number corresponding to the dataset you want to use: "))


# Validate the choice
if 1 <= choice <= len(available_datasets):
    selected_dataset = available_datasets[choice - 1]
    print(f"Downloading the {selected_dataset} dataset...")

    # Download the selected dataset
    if selected_dataset == 'Reddit':
        dataset = Reddit(root='data/Reddit')
    elif selected_dataset == 'Flickr':
        dataset = Flickr(root='data/Flickr')
    elif selected_dataset == 'FacebookPagePage':
        dataset = FacebookPagePage(root='data/FacebookPagePage')
    elif selected_dataset == 'Actor':
        dataset = Actor(root='data/Actor')
    elif selected_dataset == 'LastFMAsia':
        dataset = LastFMAsia(root='data/LastFMAsia')
    elif selected_dataset == 'DeezerEurope':
        dataset = DeezerEurope(root='data/DeezerEurope')
    elif selected_dataset == 'Amazon Computers':
        dataset = Amazon(root='data/Amazon', name='Computers')
    elif selected_dataset == 'Amazon Photo':
        dataset = Amazon(root='data/Amazon', name='Photo')
    elif selected_dataset == 'Yelp':
        dataset = Yelp(root='data/Yelp')
    elif selected_dataset == 'Cora':
        dataset = Planetoid(root=f'data/{selected_dataset}', name=selected_dataset)
    elif selected_dataset == 'Citeseer':
        dataset = Planetoid(root=f'data/{selected_dataset}', name=selected_dataset)
    elif selected_dataset == 'Pubmed':
        dataset = Planetoid(root=f'data/{selected_dataset}', name=selected_dataset)
    else:
        dataset = Planetoid(root='data/', name=selected_dataset)

    print(f"The {selected_dataset} dataset has been downloaded successfully!")
else:
    print("Invalid choice. Please run the script again and select a valid dataset.")

available_archs = ['GCN', 'SGC']

# Prompt the user to select a dataset
print("Available architectures:")
for i, archs_name in enumerate(available_archs, start=1):
    print(f"{i}: {archs_name}")

choice = int(input("Enter the number corresponding to the architecture you want to use: "))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)

if 1 <= choice <= len(available_archs):
    selected_arch = available_archs[choice - 1]
    print(f"Executing with {selected_arch} architecture...")

    # Download the selected dataset
    if selected_arch == 'GCN':
        model = GCN(dataset).to(device)
    elif selected_arch == 'SGC':
        # Use precomputed features with a linear layer
        num_node_features = dataset.num_node_features
        num_classes = dataset.num_classes
        data = precompute_features(data, num_node_features, num_classes, k=2, device=device)
        model = SGC(dataset, use_precomputed=True).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
start_time = datetime.now()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
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
