import os
import sys
import json


abs_path = sys.path[0]

base_path = os.path.dirname(abs_path)
base_path = os.path.join(base_path, 'PyLaia_LLM')
models_path = os.path.join(base_path, 'models')
base_image_path = os.path.join(base_path, 'datasets', 'washingtondb-v1.0', 'data', 'normalized_size')
transcription_washington_path = os.path.join(base_path, 'datasets', 'washingtondb-v1.0', 'ground_truth', 'transcription.txt')
resources_path = os.path.join(base_path, 'resources')
outputs_path_washington = os.path.join(resources_path, 'datasets', 'washington')
outputs_evaluation_pylaia = os.path.join(resources_path, 'outputs', 'evaluation_pylaia')
outputs_evaluation_mistral = os.path.join(resources_path, 'outputs', 'mistral_evaluation')

REPLACEMENTS_WASHINGTON = {
    's_pt': '.', 's_cm': ',', 's_mi': '-', 's_sq': ";", 's_dash': '-',
    's_sl': '/', 's_bsl': '\\', 's_qm': '?', 's_exc': '!', 's_col': ':',
    's_sc': ';', 's_lp': '(', 's_rp': ')', 's_lb': '[', 's_rb': ']',
    's_lc': '{', 's_rc': '}', 's_dq': '"', 's_ap': '@', 's_hs': '#',
    's_dl': '$', 's_pc': '%', 's_am': '&', 's_ast': '*', 's_pl': '+',
    's_eq': '=', 's_lt': '<', 's_gt': '>', 's_us': '_', 's_crt': '^',
    's_tld': '~', 's_vbar': '|', 's_sp': ' ', 's_s': 's', 's_qt': "'",
    's_GW': 'G.W.', 's_qo': ':', 's_et': 'V', 's_br': ')', 's_bl': '(',
    '|': " ", '-': '', 's_': ''
}



def save_to_json(dict_data, filename):
    """Saves a dictionary to a JSON file.
    Args:
        dict_data (dict): The dictionary to save.
        filename (str): The path to the file where the dictionary will be saved.
    """
    with open(filename, 'w') as file:
        json.dump(dict_data, file, indent=4)


def load_from_json(filename):
    """Loads data from a JSON file into a dictionary.
    Args:
        filename (str): The path to the JSON file to be loaded.
    Returns:
        dict: The dictionary containing the data loaded from the JSON file.
    """
    with open(filename, 'r') as file:
        return json.load(file)

