import os

from constants import outputs_path_washington


def concatenate_files(file_path1, file_path2, output_file_path):
    """Concatenates the contents of two text files into a new output file."""
    with open(output_file_path, 'w') as outfile:
        # Read the first file and write its contents to the output file
        with open(file_path2, 'r') as file1:
            outfile.write(file1.read())
        # Ensure there is a newline character between the contents of the two files
        # outfile.write("\n")
        # Read the second file and write its contents to the output file
        with open(file_path1, 'r') as file2:
            outfile.write(file2.read())

# Set your file paths here
# first_file_path = os.path.join(outputs_path_washington, 'train_labels_from_mistral_25.txt')  # Replace with your first file path
# second_file_path = os.path.join(outputs_path_washington, 'training_subset_25_75.txt')  # Replace with your second file path
# combined_output_file_path = os.path.join(outputs_path_washington, 'combined_training_labels_from_mistral_with_75_remaining.txt')  # Replace with your desired output file path
#
# # Call the function with the file paths
# concatenate_files(first_file_path, second_file_path, combined_output_file_path)
# print(f"Files have been concatenated into {combined_output_file_path}")
