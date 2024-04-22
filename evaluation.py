import argparse
import subprocess
import jiwer
import os
import yaml

from constants import outputs_path_washington, save_to_json, outputs_evaluation_pylaia, models_path, base_path


def run_decoding_process(config_path):
    # Command to run the decoding
    decode_command = f"pylaia-htr-decode-ctc --config {config_path} | tee predict.txt"
    try:
        subprocess.run(decode_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to run the decoding process:", e)


def update_config_yaml(config_path, experiment_dirname, img_list):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config['common']['experiment_dirname'] = experiment_dirname
    config['img_list'] = img_list

    with open(config_path, 'w') as file:
        yaml.dump(config, file)


def calculate_cer(predictions_file, ground_truths_file):
    # Load predictions
    def load_predictions(filename):
        predictions = {}
        try:
            with open(filename, "r") as file:
                for line in file:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        image_path, text = parts
                        predictions[image_path] = text
        except FileNotFoundError:
            print(f"Error: The file {filename} was not found.")
        return predictions

    # Load ground truths
    def load_ground_truths(filename):
        truths = {}
        try:
            with open(filename, "r") as file:
                for line in file:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        image_path, text = parts
                        truths[image_path] = text
        except FileNotFoundError:
            print(f"Error: The file {filename} was not found.")
        return truths

    predictions = load_predictions(predictions_file)
    ground_truths = load_ground_truths(ground_truths_file)

    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveWhiteSpace(replace_by_space=True)
    ])

    results = []

    # Calculate CER for each prediction
    for img_path, predicted_text in predictions.items():
        if img_path in ground_truths:
            true_text = transformation(ground_truths[img_path])
            predicted_text = transformation(predicted_text)

            if not true_text or not predicted_text:
                print(f"Skipping empty text for image: {img_path}")
                continue

            cer = jiwer.wer(true_text, predicted_text)
            print(f"Image: {img_path}, CER: {cer:.2%}")
            results.append({
                "file_name": os.path.basename(img_path),
                "ground_truth_label": true_text,
                "predicted_label": predicted_text,
                "cer": round(cer * 100, 2)  # CER as percentage with 2 decimal places
            })
        else:
            print(f"Missing ground truth for {img_path}")
    return results


# Main execution
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run PyLaia HTR training with dynamic configuration.")
#     parser.add_argument("--experiment_dirname", default=os.path.join(models_path, 'experiments_25'), help="Name of the experiment directory.")
#     args = parser.parse_args()
#
#     config_path = os.path.join(base_path, 'config_decode.yaml')
#     update_config_yaml(config_path, args.experiment_dirname)
#
#     # Run the decoding process to generate predict.txt
#     run_decoding_process(config_path)
#
#     # Calculate CER using predict.txt and the ground truths
#     grund_thruths_file = os.path.join(outputs_path_washington, 'test_eval.txt')
#     results = calculate_cer("predict.txt", grund_thruths_file)
#     path = os.path.join(outputs_evaluation_pylaia, "results_25.json")
#     save_to_json(results, path)
