import json
import os

from constants import outputs_evaluation_mistral, outputs_evaluation_pylaia, outputs_path_washington


def tokenize(predicted_label):
    """Converts a predicted label into a space-separated format with <space> tokens."""
    return ' '.join(
        '<space>' if char == ' ' else char for char in predicted_label
    )


def extract_mistral_labels(json_path, output_path):
    """Extracts predicted labels from the MISTRAL section in a JSON file and writes to a text file.

    Args:
        json_path (str): The path to the input JSON file.
        output_path (str): The path to the output text file.
    """
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    with open(output_path, 'w') as out_file:
        for entry in data:
            file_name = entry['file_name'].split('.')[0]  # Get the base file name without extension
            predicted_label = entry['MISTRAL']['predicted_label']
            tokenized_label = tokenize(predicted_label)  # Convert to tokenized format
            out_file.write(f"train/{file_name}.png {tokenized_label}\n")  # Write to the file in the specified format

    print(f"Labels extracted to {output_path}")


# json_file_path = os.path.join(outputs_evaluation_mistral, 'final_3.json')
# output_text_file = os.path.join(outputs_path_washington, 'train_labels_from_mistral_25.txt')
# # Call the function with the file paths
# extract_mistral_labels(json_file_path, output_text_file)
