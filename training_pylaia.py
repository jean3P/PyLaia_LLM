import argparse
import yaml
import subprocess
import os

from constants import outputs_path_washington, base_path, models_path


def update_config_yaml_train(config_path, tr_txt_table, experiment_dirname):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Update the training text table path and experiment directory name
    config['tr_txt_table'] = tr_txt_table
    config['common']['experiment_dirname'] = experiment_dirname

    # Write the updated configuration back to the file
    with open(config_path, 'w') as file:
        yaml.dump(config, file)


def run_training(config_path):
    # Command to run the training
    training_command = f"pylaia-htr-train-ctc --config {config_path} | tee training_output.log"
    try:
        subprocess.run(training_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")


# if __name__ == "__main__":
#     print(base_path)
#     parser = argparse.ArgumentParser(description="Run PyLaia HTR training with dynamic configuration.")
#
#     parser.add_argument("--config", default=os.path.join(base_path, 'config_train.yaml'),
#                         help="Path to the YAML configuration file.")
#     parser.add_argument("--tr_txt_table", default=os.path.join(outputs_path_washington, 'training_subset_25.txt'),
#                         help="Path to the training text table.")
#     parser.add_argument("--experiment_dirname", default=os.path.join(models_path, 'experiments_25'),
#                         help="Name of the experiment directory.")
#
#     args = parser.parse_args()
#
#     # Update the YAML configuration
#     update_config_yaml(args.config, args.tr_txt_table, args.experiment_dirname)
#
#     # Run the training
#     run_training(args.config)
