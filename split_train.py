import random
import os

from constants import outputs_path_washington


def select_subset_of_train_data(input_file, output_file_prefix, test_file, selection_percentage=25):
    """
    Selects a subset of the input training data file based on the specified percentage and saves it.

    Args:
        input_file (str): Path to the input .txt file containing training data.
        output_file_prefix (str): Prefix for the output files.
        selection_percentage (int): Percentage of data to be selected and saved.
    """
    # Load the data from the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(test_file, 'r') as file:
        lines_test = file.readlines()

    # Calculate selection index
    selection_index = int(len(lines) * (selection_percentage / 100))

    # Select the subset
    selected_data = lines[:selection_index]

    # Save the selected subset
    selected_output_filename = f"{output_file_prefix}_{selection_percentage}.txt"
    with open(selected_output_filename, 'w') as file:
        file.writelines(selected_data)
    print(f"Selected training data saved to {selected_output_filename} with {len(selected_data)} entries.")

    # Select the remaining data to save
    remaining_data = lines[selection_index:]

    # Create the output file name based on the remaining percentage
    remaining_percentage = 100 - selection_percentage
    output_filename = f"{output_file_prefix}_{selection_percentage}_{remaining_percentage}.txt"


    # Save the remaining subset to a new file
    with open(output_filename, 'w') as file:
        file.writelines(remaining_data)
    print(f"Remaining training data saved to {output_filename} with {len(remaining_data)} entries.")

    # Selecct test
    remaining_test = lines_test[selection_index:]
    remaining_percentage_test = 100 - selection_percentage
    output_filename_test = f"test_ids_{selection_percentage}_{remaining_percentage_test}.txt"

    # Save the remaining subset to a new file
    with open(output_filename_test, 'w') as file:
        file.writelines(remaining_test)
    print(f"Remaining training data saved to {output_filename} with {len(remaining_data)} entries.")

    return selected_output_filename, output_filename, output_filename_test


# if __name__ == "__main__":
#
#     input_file_path = os.path.join(outputs_path_washington, 'train.txt')
#     train_percentage = 25  # Modify as needed
#     output_file_prefix = os.path.join(outputs_path_washington, f"training_subset")
#
#     select_subset_of_train_data(input_file_path, output_file_prefix, train_percentage)
