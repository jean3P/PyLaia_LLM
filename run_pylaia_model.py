import subprocess


def run_pylaia_create_model(config_file_path, log_file_path):
    """
    Runs the pylaia-htr-create-model command with the given configuration file
    and logs the output to the specified log file.

    :param config_file_path: Path to the PyLaia YAML configuration file
    :param log_file_path: Path to save the log output
    """
    command = 'pylaia-htr-create-model'
    args = [command, '--config', config_file_path]

    with open(log_file_path, 'w') as log_file:
        subprocess.run(args, stdout=log_file, stderr=subprocess.STDOUT)

    print(f"Model creation completed. Log saved to {log_file_path}")


# # Paths for the configuration file and the log file
# config_file = 'config_create_model.yaml'  # Update this to your config file path
# log_file = 'pylaia_create_model.log'  # Update this to your desired log file path
#
# # Run the function
# run_pylaia_create_model(config_file, log_file)
