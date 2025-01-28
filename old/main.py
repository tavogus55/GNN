import torch
from old.data_loader import get_test_mask, get_data_from_npz
from utils import get_dataset
import torch.nn.functional as F
from old.GCN import GCN
from datetime import datetime
import json

def main(available_datasets, learning_rate, weight_decay, epochs, npz_dict):
    # Prompt the user to select a dataset
    dataset, selected_dataset, start_time = get_dataset(available_datasets)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(dataset).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()

    for epoch in range(epochs):
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
    elif (
            selected_dataset == 'LastFMAsia' or selected_dataset == 'FacebookPagePage' or selected_dataset == 'DeezerEurope'
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

    effiency = total_seconds / (epochs)

    print(f"Training lasted {hours} hours, {minutes} minutes, {seconds} seconds")
    print(f"Official efficiency: {effiency}")



if __name__ == "__main__":
    with open("../config.json", "r") as file:
        CONFIG = json.load(file)
    DATASET_OPTIONS = CONFIG["datasets"]
    LEARNING_RATE = CONFIG["learning_rate"]
    WEIGHT_DECAY = CONFIG["weight_decay"]
    EPOCHS = CONFIG["epochs"]
    NPZ_DICT = CONFIG["npz_mapping"]


    main(DATASET_OPTIONS, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, NPZ_DICT)

