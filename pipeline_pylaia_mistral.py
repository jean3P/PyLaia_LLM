import gc

import torch
import transformers
from transformers import AutoTokenizer, pipeline

from _psw_token import TOKEN
from concatenate_text_files_script import concatenate_files
from constants import outputs_path_washington, base_path, models_path, outputs_evaluation_pylaia, save_to_json, \
    load_from_json, outputs_evaluation_mistral
from evaluation import update_config_yaml, calculate_cer, run_decoding_process
from extract_labels_script import extract_mistral_labels
from mistral import evaluate_test_data_mistral7B
from run_pylaia_model import run_pylaia_create_model

import argparse
import os

from split_train import select_subset_of_train_data
from training_pylaia import run_training, update_config_yaml_train


def clear_cuda_cache():
    torch.cuda.empty_cache()  # Clear CUDA cache
    gc.collect()


def file_exists(file_name_path):
    # Check if the file exists and return the result
    return os.path.isfile(file_name_path)


def evaluation_pylaia(test_data_path, subset, type_test, test_ids, base=False):
    parser = argparse.ArgumentParser(description="Run PyLaia HTR training with dynamic configuration.")
    parser.add_argument("--experiment_dirname", default=os.path.join(models_path, f'experiments_{subset}'),
                        help="Name of the experiment directory.")
    if base:
        parser.add_argument("--img_list", default=os.path.join(outputs_path_washington, test_ids),
                            help="Name of the experiment directory.")
    else:
        parser.add_argument("--img_list", default=os.path.join(base_path, test_ids),
                        help="Name of the experiment directory.")
    args = parser.parse_args()

    config_path = os.path.join(base_path, 'config_decode.yaml')
    update_config_yaml(config_path, args.experiment_dirname, args.img_list)

    # Run the decoding process to generate predict.txt
    run_decoding_process(config_path)

    # Calculate CER using predict.txt and the ground truths
    grund_thruths_file = test_data_path
    results = calculate_cer("predict.txt", grund_thruths_file)
    path = os.path.join(outputs_evaluation_pylaia, f"evaluation_from_pylaia_{subset}_with_{type_test}.json")
    save_to_json(results, path)
    return path


def split_trainingset(training_data_path, iteration, train_ids):
    input_file_path = training_data_path
    train_percentage = iteration  # Modify as needed
    output_file_prefix = os.path.join(outputs_path_washington, f"training_subset")
    return select_subset_of_train_data(input_file_path, output_file_prefix, train_ids, train_percentage)


def run_pylaia_training_with_dynamic_config(start_percentage, name_train_file):
    """
    Function to run PyLaia HTR training with dynamic configuration.

    :param start_percentage: Surname for the experiment directory.
    :param tr_txt_table_name: Name of the training text table file.
    """
    config_file_path = os.path.join(base_path, 'config_train.yaml')
    tr_txt_table_path = os.path.join(outputs_path_washington, name_train_file)
    experiment_dirname = os.path.join(models_path, f'experiments_{start_percentage}')

    # Update the YAML configuration
    update_config_yaml_train(config_file_path, tr_txt_table_path, experiment_dirname)

    # Run the training
    run_training(config_file_path)
    clear_cuda_cache()
    return tr_txt_table_path


def run_mistral(mistral_model_name, short_system, tr_txt_table_path, start_percentage, type_test):
    mistral_model = transformers.AutoModelForCausalLM.from_pretrained(mistral_model_name,
                                                                      torch_dtype=torch.float16,
                                                                      device_map="auto",
                                                                      token=TOKEN
                                                                      )
    mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name, token=TOKEN, max_length=32)
    mistral_pipe = pipeline("text-generation", model=mistral_model, tokenizer=mistral_tokenizer, batch_size=10)

    # print(tr_txt_table_path)
    # results_path_from_ocr = os.path.join(outputs_evaluation_pylaia, tr_txt_table_path)
    loaded_data = load_from_json(tr_txt_table_path)
    ## Example usage
    name_file = f'evaluation_from_mistral_{start_percentage}_with_{type_test}.json'
    evaluate_test_data_mistral7B(loaded_data, mistral_pipe, name_file, short_system)
    del mistral_model
    del mistral_tokenizer
    del mistral_pipe
    clear_cuda_cache()
    return name_file


def extract_from_mistral_json(input_json_file, name_txt):
    json_file_path = os.path.join(outputs_evaluation_mistral, input_json_file)
    output_text_file = os.path.join(outputs_path_washington, name_txt)
    # Call the function with the file paths
    extract_mistral_labels(json_file_path, output_text_file)


def mixed_datasets(mistral_labels_file, remaining_subset, final_name):
    first_file_path = os.path.join(outputs_path_washington,
                                   mistral_labels_file)  # Replace with your first file path
    second_file_path = os.path.join(outputs_path_washington,
                                    remaining_subset)  # Replace with your second file path
    combined_output_file_path = os.path.join(outputs_path_washington, final_name)
    # Call the function with the file paths
    concatenate_files(first_file_path, second_file_path, combined_output_file_path)


def directory_exists(directory_path):
    return os.path.isdir(directory_path)


def automate_workflow(start_percentage=25, increments=25, max_iterations=3, training_data_path='',
                      test_data_path='', mistral_model_name=''):
    # First run the PyLaia model
    model_path = os.path.join(base_path, 'config_create_model.yaml')
    model_log_path = os.path.join(base_path, 'pylaia_create_model.log')
    print(f"=== RUNING - MODEL - {start_percentage} ===")
    run_pylaia_create_model(model_path, model_log_path)
    train_ids = os.path.join(outputs_path_washington, 'train_ids.txt')
    for iteration in range(max_iterations):
        # Save subset
        print(f"=== SPLIT - TRAINING SET - {start_percentage} ===")
        training_subset, subset_remain, remaining_subset = split_trainingset(training_data_path, start_percentage, train_ids)

        # Run Training Model
        print(f"=== TRAINING - MODEL - {start_percentage} ===")
        directory_model_ = os.path.join(models_path,  f'experiments_{start_percentage}')
        if not directory_exists(directory_model_):
            name_train_file = f'training_subset_{start_percentage}.txt'
            run_pylaia_training_with_dynamic_config(start_percentage, name_train_file)
        else:
            print(f"=== TRAINING - MODEL - {start_percentage} EXISTS PREVIOUSLY ===")

        print(f"=== EVALUATE PYLAIA WITH FINAL TEST - {start_percentage} ===")
        path_eval_pylaia_final_test_without = evaluation_pylaia(test_data_path, start_percentage, 'final_test', 'test_ids.txt', True)

        print(f"=== MISTRAL WITHOUT SELF TRAINING WITH FINAL TEST - {start_percentage} ===")
        run_mistral(mistral_model_name, True, path_eval_pylaia_final_test_without, start_percentage, 'final_test')

        if start_percentage < 100:
            print(
                f"=== EVALUATE PYLAIA SELF TRAINING - {start_percentage} WITH REMAINING TEST {100 - start_percentage} ===")
            path_remaining = os.path.join(outputs_path_washington, 'train_eval.txt')
            print(path_remaining)
            eval_self_training_test_remaining_path = evaluation_pylaia(path_remaining, start_percentage,
                                                                       f'remaining_{100 - start_percentage}_test', remaining_subset, False)

            print(f"=== MISTRAL SELF TRAINING - {start_percentage} ===")
            mistral_file = run_mistral(mistral_model_name, False, eval_self_training_test_remaining_path,
                                       start_percentage, f'remaining_{100 - start_percentage}_test')
            # mistral_file = os.path.join(outputs_evaluation_mistral, 'evaluation_from_mistral_25_with_remaining_75_test.json')

            print(f"=== MERGE - {start_percentage}-{100 - start_percentage} ===")
            extact_txt_file = f"extract_labels_from_mistral_{start_percentage}_{100 - start_percentage}.txt"
            extract_from_mistral_json(mistral_file, extact_txt_file)
            name_mixed_file = f"mixed_train_{start_percentage}_{100 - start_percentage}.txt"
            mixed_datasets(extact_txt_file, training_subset, name_mixed_file)

            print(f"=== RUN PYLAIA MODEL SELF - {start_percentage}-{100 - start_percentage} ===")
            run_pylaia_training_with_dynamic_config(f"{start_percentage}_{100 - start_percentage}", name_mixed_file)

            print(f"=== EVALUATE PYLAIA SELF TRAINING MIXED FILE - {start_percentage}  ===")
            final_eval_from_mixed = evaluation_pylaia(test_data_path, f"{start_percentage}_{100 - start_percentage}",
                                                      'final_test_after_mixed', 'test_ids.txt', True)
            print(f"=== FINAL TEST MISTRAL - {start_percentage}-{100 - start_percentage} ===")
            run_mistral(mistral_model_name, True, final_eval_from_mixed,
                                       start_percentage, f'final_test_{start_percentage}_{100 - start_percentage}_test')

        start_percentage = start_percentage + increments


if __name__ == "__main__":
    training_data_path = os.path.join(outputs_path_washington, 'train.txt')
    test_data_path = os.path.join(outputs_path_washington, 'test_eval.txt')
    mistral_model_name = "mistralai/Mistral-7B-v0.1"
    automate_workflow(75, 25, 1, training_data_path, test_data_path, mistral_model_name)
