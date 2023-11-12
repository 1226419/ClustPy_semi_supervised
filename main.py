import argparse
import pathlib
import os
import yaml

from clustpy.utils.read_config import get_dataloaders_from_config, get_algorithmns_from_config, get_metrics_from_config, get_autoencoders_from_config
from clustpy.utils import evaluate_multiple_datasets


def argument_parser():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Read YAML file from command-line argument')

    # Add a positional argument for the YAML file path
    parser.add_argument('config_file_path', type=str, help='Path to the YAML file')
    args = parser.parse_args()
    return args


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def main():
    arguments = argument_parser()
    # Parse the command-line arguments
    try:
        parameters = read_yaml_file(arguments.config_file_path)
        print("YAML Data:")
        print(parameters)
    except FileNotFoundError:
        print(f"Error: File not found at path {arguments.config_file_path}")
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")

    from datetime import datetime

    # Get the current date and time
    now = datetime.now()

    # Format the date and time as YYYY-MM-DD-hh:mm
    formatted_datetime = now.strftime("%Y_%m_%d_%H_%M")

    # Create a folder with the formatted datetime
    folder_name = formatted_datetime
    save_path = parameters["general"]["save_path"]
    results_path = os.path.join(save_path, folder_name)
    pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)
    config_file_path = os.path.join(results_path, "config.yaml")
    with open(config_file_path, "w") as yaml_file:
        yaml.dump(parameters, yaml_file, default_flow_style=False)

    list_of_dataloaders, list_of_unique_datasets = get_dataloaders_from_config(parameters["dataset"])

    for unique_dataset in list_of_unique_datasets:
        list_of_autoencoders = get_autoencoders_from_config(parameters["autoencoder"], unique_dataset)
        list_of_algorithmns = get_algorithmns_from_config(parameters["algorithm"], list_of_autoencoders)
        list_of_metrics = get_metrics_from_config(parameters["metric"])
        df = evaluate_multiple_datasets(list_of_dataloaders, list_of_algorithmns, list_of_metrics,
                                        n_repetitions=parameters["evaluation"]["n_repetitions"],
                                        aggregation_functions=parameters["evaluation"]["aggregation_functions"],
                                        add_runtime=parameters["evaluation"]["add_runtinme"],
                                        add_n_clusters=parameters["evaluation"]["add_n_clusters"],
                                        save_path=f"multi_dataset_result.csv",
                                        save_intermediate_results=parameters["evaluation"]["save_intermediatie_results"])


if __name__ == "__main__":
    main()
