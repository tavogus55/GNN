from old.data_loader import get_data
# from datetime import datetime
import time

def get_selected_dataset(available_datasets):
    dataset = None
    selected_dataset = None
    start_time = None

    print("Available datasets:")
    for i, dataset_name in enumerate(available_datasets, start=1):
        print(f"{i}: {dataset_name}")

    choice = int(input("Enter the number corresponding to the dataset you want to use: "))

    if 1 <= choice <= len(available_datasets):
        selected_dataset = available_datasets[choice - 1]

    return selected_dataset

def get_dataset_data(selected_dataset):

        start_time = time.time()
        if 'Amazon' in selected_dataset:
            dataset, train_mask, test_mask = get_data('./data/', selected_dataset, type=selected_dataset.split()[1])
        else:
            dataset, train_mask, test_mask = get_data('./data/', selected_dataset)

        return dataset, selected_dataset, start_time, train_mask, test_mask