from old.data_loader import get_data
# from datetime import datetime
import argparse
import time
import  logging

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
            dataset, train_mask, test_mask = get_data('./data/', selected_dataset)
        else:
            dataset, train_mask, test_mask = get_data('./data/', selected_dataset)

        return dataset, selected_dataset, start_time, train_mask, test_mask

def set_arg_parser():
    ALLOWED_DATASETS = [
        "Cora",
        "CiteSeer",
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
    parser.add_argument("--log_path", type=str, required=True, help="Where you want to store the logs")
    # parser.add_argument("--tuning", type=int, help="How many times you want to tune the hyperparameters")
    args = parser.parse_args()

    cuda_num = args.cuda_num
    dataset_decision = args.data
    model = args.model
    exp_times = args.exp
    log_path = args.log_path
    # is_tuning = args.tuning

    return cuda_num, dataset_decision, model, exp_times, log_path

class CustomFormatter(logging.Formatter):

    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name, log_path):

    logger = logging.getLogger(name)
    logging.basicConfig(
        filename=log_path,
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    ch.setFormatter(CustomFormatter())

    logger.addHandler(ch)

    return logger