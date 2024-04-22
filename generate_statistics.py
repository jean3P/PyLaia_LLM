import json
import os

from constants import outputs_evaluation_mistral, outputs_evaluation_pylaia


def calculate_cer_values(directory_path_mistral, filename_mistral, path_directory_ocr, other_filename_ocr):
    """
    Calculates and returns the mean CER values for a specified Mistral file and an OCR file.

    Args:
        directory_path_mistral (str): The directory where the Mistral JSON file is located.
        filename_mistral (str): The name of the Mistral JSON file.
        path_directory_ocr (str): The directory where the OCR JSON file is located.
        other_filename_ocr (str): The name of the OCR JSON file.

    Returns:
        dict: A dictionary with mean CER values for both Mistral and OCR files, rounded to two decimal places.
    """
    # Load and calculate mean CER for the Mistral file
    mistral_file_path = os.path.join(directory_path_mistral, filename_mistral)
    mistral_data = load_from_json(mistral_file_path)
    mean_cer_mistral = calculate_mean(mistral_data, 'MISTRAL', 'cer')
    mean_confidence_mistral = calculate_mean(mistral_data, 'MISTRAL', 'confidence')

    # Load and calculate mean CER for the OCR file
    ocr_file_path = os.path.join(path_directory_ocr, other_filename_ocr)
    ocr_data = load_from_json(ocr_file_path)
    # Calculate the total CER and total confidence
    total_cer = sum(item['cer'] for item in ocr_data)

    # Calculate the number of items
    num_items = len(ocr_data)

    # Calculate the mean CER and mean confidence
    mean_cer_ocr = total_cer / num_items
    results = {
        'Mean CER OCR': round(mean_cer_ocr, 2) if mean_cer_ocr is not None else None,
        'Mean CER Mistral': round(mean_cer_mistral, 2) if mean_cer_mistral is not None else None,
        'Mean Confidence Mistral': round(mean_confidence_mistral, 2) if mean_confidence_mistral is not None else None,
    }

    return results


# Function to load JSON data from a file
def load_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Function to calculate mean values given data, a key, and a field
def calculate_mean(data, key, field):
    values = [item[key][field] for item in data if key in item and field in item[key]]
    return sum(values) / len(values) if values else None

# Example usage
# directory_path = 'path_to_mistral_data'
# filename = 'mistral_file.json'
# path_directory = 'path_to_ocr_data'
# other_filename = 'ocr_file.json'
# results = calculate_cer_values(directory_path, filename, path_directory, other_filename)
# print(results)


def calculate_label_change_percentages(directory, file_name):
    """
    Calculate percentages of label changes based on CER values from a specified JSON file, formatted to two decimal places, and return them.

    Args:
        directory (str): The directory where the JSON file is located.
        file_name (str): The name of the JSON file.

    Returns:
        dict: A dictionary containing the percentages of label changes for CER=0 and CER>0, formatted to two decimal places.
    """
    file_path = os.path.join(directory, file_name)

    with open(file_path, 'r') as file:
        json_data = json.load(file)

    same_label_cer_0_count = 0
    modified_label_cer_0_count = 0
    same_label_cer_greater_than_0_count = 0
    modified_label_cer_greater_than_0_count = 0

    for item in json_data:
        ocr_cer = item["OCR"]["cer"]
        ocr_label = item["OCR"]["predicted_label"]
        mistral_label = item["MISTRAL"]["predicted_label"]
        label_changed = (ocr_label != mistral_label)

        if ocr_cer == 0:
            if label_changed:
                modified_label_cer_0_count += 1
            else:
                same_label_cer_0_count += 1
        else:
            if label_changed:
                modified_label_cer_greater_than_0_count += 1
            else:
                same_label_cer_greater_than_0_count += 1

    total_cer_0 = same_label_cer_0_count + modified_label_cer_0_count
    total_cer_greater_than_0 = same_label_cer_greater_than_0_count + modified_label_cer_greater_than_0_count

    # Calculate and format percentages
    same_label_cer_0_percentage = round((same_label_cer_0_count / total_cer_0 * 100) if total_cer_0 else 0, 2)
    modified_label_cer_0_percentage = round((modified_label_cer_0_count / total_cer_0 * 100) if total_cer_0 else 0, 2)
    same_label_cer_greater_than_0_percentage = round((same_label_cer_greater_than_0_count / total_cer_greater_than_0 * 100) if total_cer_greater_than_0 else 0, 2)
    modified_label_cer_greater_than_0_percentage = round((modified_label_cer_greater_than_0_count / total_cer_greater_than_0 * 100) if total_cer_greater_than_0 else 0, 2)

    return {
        "same_label_cer_0_percentage": same_label_cer_0_percentage,
        "modified_label_cer_0_percentage": modified_label_cer_0_percentage,
        "same_label_cer_greater_than_0_percentage": same_label_cer_greater_than_0_percentage,
        "modified_label_cer_greater_than_0_percentage": modified_label_cer_greater_than_0_percentage
    }


def generate_latex_table(automated_results_dir, results_test_trocr_dir,
                         self_training_file_name_ocr_75, final_test_file_name_ocr_75_25, final_test_file_name_ocr_75,
                         self_training_file_name_mistral_75, final_test_file_name_mistral_75, final_test_file_name_mistral_75_25):
    """
    Generates a LaTeX table with CER values and label change percentages for OCR and Mistral evaluations.

    Args:
        automated_results_dir (str): Directory containing Mistral JSON files.
        results_test_trocr_dir (str): Directory containing OCR JSON files.
        self_training_ocr (str): OCR JSON file name for self-training.
        final_test_ocr_25 (str): OCR JSON file name for 25% final test.
        final_test_ocr_75 (str): OCR JSON file name for 75% final test.
        self_training_mistral (str): Mistral JSON file name for self-training.
        final_test_mistral_75 (str): Mistral JSON file name for 75% final test.
        final_test_mistral_75_25 (str): Mistral JSON file name for 75%+25% final test.

    Returns:
        str: LaTeX code for the table.
    """

    # Assuming the 'calculate_cer_values' and 'calculate_label_change_percentages' functions are defined as before

    # Generate CER values for OCR and Mistral
    cer_values_ocr_self = calculate_cer_values(automated_results_dir, self_training_file_name_mistral_75, results_test_trocr_dir, self_training_file_name_ocr_75)
    cer_values_ocr_final_75 = calculate_cer_values(automated_results_dir, final_test_file_name_mistral_75, results_test_trocr_dir, final_test_file_name_ocr_75)
    cer_values_ocr_final_75_25 = calculate_cer_values(automated_results_dir, final_test_file_name_mistral_75_25, results_test_trocr_dir, final_test_file_name_ocr_75_25)

    # Generate label change percentages for Mistral
    label_changes_self = calculate_label_change_percentages(automated_results_dir, self_training_file_name_mistral_75)
    label_changes_final_75 = calculate_label_change_percentages(automated_results_dir, final_test_file_name_mistral_75)
    label_changes_final_75_25 = calculate_label_change_percentages(automated_results_dir, final_test_file_name_mistral_75_25)

    dict = {
        "Self-training": cer_values_ocr_self,
        "Final test normal": cer_values_ocr_final_75,
        "Final test with Self-training": cer_values_ocr_final_75_25,
        "Labeling changes Self-training": label_changes_self,
        "Labeling changes Final test normal": label_changes_final_75,
        "Labeling changes Final test Self-training": label_changes_final_75_25
    }
    return dict


self_training_file_name_ocr_75 = "evaluation_from_pylaia_75_with_remaining_25_test.json"
final_test_file_name_ocr_75_25 = "evaluation_from_pylaia_75_25_with_final_test_after_mixed.json"
final_test_file_name_ocr_75 = "evaluation_from_pylaia_100_with_final_test.json"

# Mistral Files
self_training_file_name_mistral_75 = "evaluation_from_mistral_75_with_remaining_25_test.json"
final_test_file_name_mistral_75 = "evaluation_from_mistral_100_with_final_test.json"
final_test_file_name_mistral_75_25 = "evaluation_from_mistral_75_with_final_test_75_25_test.json"

# Directory paths
directory_path_mistral = outputs_evaluation_mistral
path_directory_ocr = outputs_evaluation_pylaia

# Generate LaTeX table code
latex_code = generate_latex_table(
    automated_results_dir=directory_path_mistral,
    results_test_trocr_dir=path_directory_ocr,
    self_training_file_name_ocr_75=self_training_file_name_ocr_75,
    final_test_file_name_ocr_75_25=final_test_file_name_ocr_75_25,
    final_test_file_name_ocr_75=final_test_file_name_ocr_75,
    self_training_file_name_mistral_75=self_training_file_name_mistral_75,
    final_test_file_name_mistral_75=final_test_file_name_mistral_75,
    final_test_file_name_mistral_75_25=final_test_file_name_mistral_75_25
)

print(latex_code)