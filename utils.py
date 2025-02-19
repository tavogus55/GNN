from old.data_loader import get_data
# from datetime import datetime
import argparse
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

def set_arg_parser():
    ALLOWED_DATASETS = [
        "Cora",
        "Citeseer",
        "PubMed",
        "Flickr",
        "FacebookPagePage",
        "Actor",
        "LastFMAsia",
        "DeezerEurope",
        "Amazon Computers",
        "Amazon Photo",
        "Reddit",
        "Arxiv",
        "Products"
    ]

    parser = argparse.ArgumentParser(description="GNN script")
    parser.add_argument("--cuda_num", type=str, required=True, help="GPU to use")
    parser.add_argument(
        "--data",
        type=str,
        choices=ALLOWED_DATASETS,  # Restricts choices
        required=True,
        help=f"Dataset name (choices: {', '.join(ALLOWED_DATASETS)})"
    )
    parser.add_argument("--model", type=str, required=True, help="GCN or SGC")
    parser.add_argument("--exp", type=int, required=True, help="How many times do you want to run the exercise")
    # parser.add_argument("--log_path", type=str, help="Where you want to store the logs")
    # parser.add_argument("--tuning", type=int, help="How many times you want to tune the hyperparameters")
    args = parser.parse_args()

    cuda_num = args.cuda_num
    dataset_decision = args.data
    model = args.model
    exp_times = args.exp
    # log_path = args.log_path
    # is_tuning = args.tuning

    return cuda_num, dataset_decision, model, exp_times