from torch_geometric.datasets import Planetoid
import torch
from data_loader import get_data, get_train_mask, get_test_mask, get_data_from_npz
import torch.nn.functional as F
from GCN import GCN
from datetime import datetime

available_datasets = ['Cora', 'CiteSeer', 'PubMed', 'LastFMAsia', 'FacebookPagePage', 'DeezerEurope', 'Amazon Photo',
                      'Amazon Computers', 'Actor', 'Flickr', 'Reddit']

# Prompt the user to select a dataset
print("Available datasets:")
for i, dataset_name in enumerate(available_datasets, start=1):
    print(f"{i}: {dataset_name}")

choice = int(input("Enter the number corresponding to the dataset you want to use: "))

if 1 <= choice <= len(available_datasets):
    selected_dataset = available_datasets[choice - 1]
    if 'Amazon' in selected_dataset:
        dataset = get_data('./data/', selected_dataset, type=selected_dataset.split()[1])
    else:
        dataset = get_data('./data/', selected_dataset)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

start_time = datetime.now()
model.train()

npz_dict = {"LastFMAsia": "lastfm_asia",
            "FacebookPagePage": "facebook",
            "DeezerEurope": "deezer_europe",
            "Amazon Photo": "amazon_electronics_photo",
            "Amazon Computers": "amazon_electronics_computers"
            }

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    if selected_dataset == 'Cora' or selected_dataset == 'PubMed' or selected_dataset == 'CiteSeer':
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    elif (selected_dataset == 'LastFMAsia' or selected_dataset == 'FacebookPagePage'
          or selected_dataset == 'DeezerEurope'
          or selected_dataset == 'Amazon Photo'
          or selected_dataset == 'Amazon Computers'):
        _, _, _, train_mask, _, _ = get_data_from_npz(selected_dataset, npz_dict[selected_dataset])
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
if selected_dataset == 'Cora' or selected_dataset == 'PubMed' or selected_dataset == 'CiteSeer':
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
elif (selected_dataset == 'LastFMAsia' or selected_dataset == 'FacebookPagePage' or selected_dataset == 'DeezerEurope'
      or selected_dataset == 'Amazon' or selected_dataset == 'Actor' or selected_dataset == 'Flickr'):
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

