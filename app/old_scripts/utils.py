import random

import json
from pathlib import Path

def split_dict(data: json, ratios: tuple[float, float, float] = (0.7, 0.1, 0.2)) -> tuple[dict, dict, dict]:
    """Split a dictionary into 10% and 90% for annotation.

    Args:
        dictionary (dict): Preprocessed dictionary with filename as key and value is text.
        train_ratio (float, optional): Ratio with which the dictionary should be split. Defaults to 0.9.

    Returns:
        tuple[dict, dict]: Two dictionaries: one for annotating and one for testing.
    """

    # Shuffle the data. 
    random.shuffle(data)
    # Find index on which the dictionary should be split.
    split1 = int(len(data) * ratios[0])
    split2 = split1 + int(len(data) * ratios[1])
    

    train_dict = data[:split1]
    validation_dict = data[split1:split2]
    test_dict = data[split2:]
    
    return train_dict, validation_dict, test_dict


def prepare_annotation_doccano(filepath: str, destination_folder_path:str) -> None:
    """ Extract indiviual documents from JSON file and save them as individual document in a folder with text and label 
        as values and filename as the documents name (not a key).

    Args:
        filepath (str): Filepath to the JSON file documents should be extracted from.
        destination_folder_path (str): Folder where the individual documents should be saved in. 
    """    
    # Open JSON file. 
    with open(filepath, "r") as f: 
        json_file = json.load(f)
    
    # Iterate over it. 
    for key, value in json_file.items():
        # Create document in given location give it a name (inclusive ending .jsonl). Save it in variable. 
        filename = os.path.join(destination_folder_path, f"{key}.jsonl")
        
        # Create empty, temporary dictionary which is filled with necessary keys (text and label).
        temp_dict = {}
        temp_dict["text"] = value
        temp_dict["label"] = []

        # Open indivdual document and dump temporary dictionary in it. 
        with open(filename, "w", encoding="utf-8") as jsonl_file:
            jsonl_file.write(json.dumps(temp_dict, ensure_ascii=False) + "\n")

# Apply function. 
#prepare_annotation_doccano("annotation_eval.json", "annotation_eval")

path_to_data = Path(Path(__file__).parent, "../data/annotated_data.json")

with open(path_to_data, "r") as f: 
    data = json.load(f)


train, val, test = split_dict(data)

save_path_train = Path(Path(__file__).parent, "../data/train_data.json")
save_path_val = Path(Path(__file__).parent, "../data/val_data.json")
save_path_test = Path(Path(__file__).parent, "../data/test_data.json")

with open(save_path_train, "w", encoding="utf-8") as f: 
    json.dump(train, f, indent=2, ensure_ascii=False)

with open(save_path_val, "w", encoding="utf-8") as f: 
    json.dump(val, f, indent=2, ensure_ascii=False)

with open(save_path_test, "w", encoding="utf-8") as f: 
    json.dump(test, f, indent=2, ensure_ascii=False)

