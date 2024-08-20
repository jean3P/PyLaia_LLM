import json
import os
from constants import outputs_evaluation_mistral


def count_cer_zero(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Check if data is a list; if not, print a warning and skip this file
    if not isinstance(data, list):
        print(f"Warning: Data in {file_path} is not a list.")
        return 0, 0

    total_entries = len(data)
    ocr_cer_zero_count = 0
    mistral_cer_zero_count = 0

    for item in data:
        try:
            if item["OCR"]["cer"] == 0:
                ocr_cer_zero_count += 1
            if item["MISTRAL"]["cer"] == 0:
                mistral_cer_zero_count += 1
        except KeyError:
            print(f"Error in data format in {file_path}: keys not found.")

    ocr_cer_zero_percentage = (ocr_cer_zero_count / total_entries) * 100
    mistral_cer_zero_percentage = (mistral_cer_zero_count / total_entries) * 100

    return ocr_cer_zero_percentage, mistral_cer_zero_percentage


# Assuming your JSON files are stored in the directory "json_data"
directory = outputs_evaluation_mistral
results = []

print("{:<50} {:>15} {:>15}".format("File Name", "OCR CER=0%", "MISTRAL CER=0%"))

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        result = count_cer_zero(file_path)
        if result:
            results.append(result)
            print("{:<50} {:>15.2f}% {:>15.2f}%".format(filename, result[0], result[1]))


if results:
    avg_ocr_cer_zero = sum([result[0] for result in results]) / len(results)
    avg_mistral_cer_zero = sum([result[1] for result in results]) / len(results)
    print("\nAverage OCR CER=0%: {:.2f}%".format(avg_ocr_cer_zero))
    print("Average MISTRAL CER=0%: {:.2f}%".format(avg_mistral_cer_zero))
else:
    print("No valid data found to calculate averages.")
