import json
import os
import re
import random
import math
import shutil
from constants import REPLACEMENTS_WASHINGTON, transcription_washington_path, outputs_path_washington, base_image_path, \
    save_to_json


def tokenize(text):
    # Replace spaces with <space> tokens and tokenize the rest of the characters
    return ' '.join(text.replace(' ', '<space>').replace('', ' ').strip().split())


def tokenize_transcription(transcription):
    """Converts transcription to a spaced format with special token for spaces."""
    return ' '.join(
        '<space>' if char == ' ' else char for char in transcription
    )


class LabelParser:
    """A class for parsing labels from a dataset and splitting the dataset.

    Attributes:
        path (str): The file path to the dataset.
        seed (int): The seed for random number generation to ensure reproducibility.
        pattern (re.Pattern): Compiled regular expression pattern for replacements.
    """

    def __init__(self, path, seed=42):
        """Initializes the LabelParser with a dataset path and a seed for randomization."""
        self.path = path
        self.seed = seed
        self.pattern = re.compile('|'.join(re.escape(key) for key in REPLACEMENTS_WASHINGTON.keys()))

    def parse_label(self, label):
        """Parses a single label from the dataset.
        Args:
            label (str): The label string to parse.
        Returns:
            tuple: A tuple containing the image name and the processed label.
        """
        image_name, rest_of_label = label[:6], label[6:]
        rest_of_label = self.pattern.sub(lambda x: REPLACEMENTS_WASHINGTON[x.group()], rest_of_label)
        rest_of_label = rest_of_label.rstrip('\n')

        # Remove the first space character if it exists
        if rest_of_label.startswith(' '):
            rest_of_label = rest_of_label[1:]

        return image_name, rest_of_label

    def get_subsets(self, training_pct, validation_pct):
        """Splits the dataset into training, validation, and testing subsets.
        Args:
            training_pct (int): The percentage of the dataset to allocate to the training set.
            validation_pct (int): The percentage of the dataset to allocate to the validation set.
        Returns:
            tuple: A tuple of dictionaries for the training, validation, and testing sets.
        """
        with open(self.path, 'r') as file:
            lines = file.readlines()

        random.seed(self.seed)
        random.shuffle(lines)

        total = len(lines)
        training_size = math.ceil(total * training_pct / 100)
        validation_size = math.ceil(total * validation_pct / 100)

        training_lines = lines[:training_size]
        validation_lines = lines[training_size:training_size + validation_size]
        testing_lines = lines[training_size + validation_size:]

        return self._lines_to_dict(training_lines), self._lines_to_dict(validation_lines), self._lines_to_dict(
            testing_lines)

    def get_sequential_subsets(self, training_pct, validation_pct):
        """Sequentially splits the dataset into training, validation, and testing subsets.
        Args:
            training_pct (int): The percentage of the dataset to allocate to the training set.
            validation_pct (int): The percentage of the dataset to allocate to the validation set.
        Returns:
            tuple: A tuple of dictionaries for the training, validation, and testing sets.
        """
        with open(self.path, 'r') as file:
            lines = file.readlines()

        total = len(lines)
        training_size = math.ceil(total * training_pct / 100)
        validation_size = math.ceil(total * validation_pct / 100)

        training_lines = lines[:training_size]
        validation_lines = lines[training_size:training_size + validation_size]
        testing_lines = lines[training_size + validation_size:]

        return self._lines_to_dict(training_lines), self._lines_to_dict(validation_lines), self._lines_to_dict(
            testing_lines)

    def _lines_to_dict(self, lines):
        dict_images_labels = {}
        for line in lines:
            name, label = self.parse_label(line)
            name_ext = name + '.png'
            dict_images_labels[name_ext] = label
        return dict_images_labels

    def manage_images(self, data_dict, source_folder, destination_folder):
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder, exist_ok=True)
        for image_name in data_dict.keys():
            shutil.copy(os.path.join(source_folder, image_name), destination_folder)

    def write_text_files(self, data_dict, subset_name, output_dir):
        """Writes the .txt, _ids.txt, and _eval.txt files for a specific subset."""
        txt_path = os.path.join(output_dir, f"{subset_name}.txt")
        ids_path = os.path.join(output_dir, f"{subset_name}_ids.txt")
        eval_path = os.path.join(output_dir, f"{subset_name}_eval.txt")

        with open(txt_path, 'w') as txt_file, open(ids_path, 'w') as ids_file, open(eval_path, 'w') as eval_file:
            for image_name, transcription in data_dict.items():
                # Write mapping image to transcription in tokens
                # Tokenize and format the transcription appropriately
                formatted_transcription = tokenize_transcription(transcription)
                txt_file.write(f"{subset_name}/{image_name} {formatted_transcription}\n")
                # Write image ids
                ids_file.write(f"{subset_name}/{image_name}\n")
                # Write human-readable eval mapping
                eval_file.write(f"{subset_name}/{image_name} {transcription}\n")


# Initialize LabelParser
label_parser = LabelParser(transcription_washington_path)

# Split dataset and save to JSON files
# training_data, validation_data, testing_data = label_parser.get_subsets(70, 15)
training_data, validation_data, testing_data = label_parser.get_sequential_subsets(80, 10)
# Output paths
output_paths = {
    'train': 'train',
    'valid': 'valid',
    'test': 'test'
}

# Saving JSON and managing images
for subset, data in zip(output_paths.keys(), [training_data, validation_data, testing_data]):
    _path = os.path.join(outputs_path_washington, '')
    save_to_json(data, os.path.join(_path, f"{subset}_seq_data.json"))
    label_parser.write_text_files(data, subset, _path)
    label_parser.manage_images(data, base_image_path, os.path.join(outputs_path_washington, subset))

# save_to_json(training_data, os.path.join(outputs_path_washington, 'train', 'training_seq_data.json'))
# save_to_json(validation_data, os.path.join(outputs_path_washington, 'valid', 'validation_seq_data.json'))
# save_to_json(testing_data, os.path.join(outputs_path_washington, 'test', 'testing_seq_data.json'))
#
#
# # Move images to corresponding directories
# label_parser.manage_images(training_data, base_image_path, os.path.join(outputs_path_washington, 'train'))
# label_parser.manage_images(validation_data, base_image_path, os.path.join(outputs_path_washington, 'valid'))
# label_parser.manage_images(testing_data, base_image_path, os.path.join(outputs_path_washington, 'test'))
