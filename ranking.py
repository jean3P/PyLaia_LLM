import json
import os
import pandas as pd

from constants import outputs_evaluation_mistral, load_from_json, outputs_evaluation_pylaia


def calculate_mean(data, key, field):
    values = [item[key][field] for item in data if key in item and field in item[key]]
    return sum(values) / len(values) if values else None


# Directory containing the JSON files
directory = outputs_evaluation_mistral

# Store the mean CER for each file
mean_cer_data = []

# Iterate through each file in the directory
length = 0
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        length = length+1
        data = load_from_json(file_path)
        mean_cer = calculate_mean(data, 'MISTRAL', 'cer')
        mean_confidence = calculate_mean(data, 'MISTRAL', 'confidence')
        if mean_cer is not None:
            mean_cer_data.append({'File': filename, 'Mean CER': round(mean_cer, 2), 'Mean Confidence': mean_confidence})

# Convert the data to a DataFrame
df = pd.DataFrame(mean_cer_data)

# Sort the DataFrame by Mean CER and get the top 3 files
df_sorted = df.sort_values(by='Mean CER').head(length)

# Display the top 3 files with the lowest mean CER
print("Top 3 files with lowest Mean CER for Mistral:")
print(df_sorted)

# path_ = os.path.join(outputs_evaluation_pylaia, 'evaluation_from_pylaia_25_with_final_test.json')
path_ = os.path.join(outputs_evaluation_pylaia, 'evaluation_from_pylaia_25_with_remaining_75_test.json')
with open(path_, 'r') as file:
    data = json.load(file)

# Calculate the total CER and total confidence
total_cer = sum(item['cer'] for item in data)

# Calculate the number of items
num_items = len(data)

# Calculate the mean CER and mean confidence
mean_cer = total_cer / num_items

print(f"==================")
print(f"Mean CER - Pylaia: {mean_cer}")