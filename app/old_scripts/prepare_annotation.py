from pathlib import Path
import json

import random

path = Path(Path("__file__").parent, "../data/new_data_to_annotate.json")
savepath = Path(Path("__file__").parent, "../data/new_data_to_annotate_1000.json")

def split_dict(path: str, savepath: str) -> None: 

    with open(path, "r") as f: 
        data = json.load(f)

    data_1000 = {key: data[key] for key in list(data.keys())[:1000]}

    with open(savepath, "w", encoding="utf-8") as f: 
        json.dump(data_1000, f, indent=2, ensure_ascii=False)

with open(savepath, "r") as f: 
    data = json.load(f)

def replace_newline(data: json, savepath: str) -> None: 


    processed_dict = {}

    for key, value in data.items(): 
        new_value = value.replace("\n", " ")
        processed_dict[key] = new_value
    
    with open(savepath, "w", encoding="utf-8") as f: 
        json.dump(processed_dict, f, indent=2, ensure_ascii=False)

def split_dict(data: json, ratios: tuple[float, float] = (0.8, 0.2)) -> tuple[dict, dict]:
    """Split a dictionary into 20% and 80% for annotation.

    Args:
        dictionary (dict): Preprocessed dictionary with filename as key and value is text.
        train_ratio (float, optional): Ratio with which the dictionary should be split.

    Returns:
        tuple[dict, dict]: Two dictionaries: a bigger one for annotation and a smaller one for testing the accuracy of annotation.
    """

    data_list = list(data.keys())
    # Shuffle the data. 
    random.shuffle(data_list)
    # Find index on which the dictionary should be split.
    split1 = int(len(data_list) * ratios[0])

    annotation = data_list[:split1]
    annotation_val = data_list[split1:]

    annotation_dict = {}
    annotation_val_dict = {}

    for key, value in data.items(): 
        for i in annotation:
            if i == key: 
                annotation_dict[i] = value
    
    for key, value in data.items(): 
        for i in annotation_val: 
            if i == key: 
                annotation_val_dict[i] = value


    return annotation_dict, annotation_val_dict

with open(savepath, "r") as f: 
    data = json.load(f)

#annotation, annotation_val = split_dict(data)

annotation_path = Path(Path("__file__").parent, "../data/new_docanno_annotation.json")
annotation_val_path = Path(Path("__file__").parent, "../data/new_docanno_cross_annoation.json")
"""
with open(annotation_path, "w", encoding="utf-8") as f: 
    json.dump(annotation, f, indent=2, ensure_ascii=False)

with open(annotation_val_path, "w", encoding="utf-8") as f: 
    json.dump(annotation_val, f, indent=2, ensure_ascii=False)"""

with open(annotation_path, "r") as f: 
    annotation = json.load(f)

with open(annotation_val_path, "r") as f: 
    annotation_val = json.load(f)

annotation_directory = Path(Path("__file__").parent, "../data/doccano/new_annotation")
annotation_val_directory = Path(Path("__file__").parent, "../data/doccano/new_annotation_val")

def create_individual_doccano_files(data: json, save_dir: str) -> None: 
    
    for key, value in data.items(): 
        file_path = save_dir / f"{key}.json"

        file_content = {"text": value, "label": []}

        with open(file_path, "w", encoding="utf-8") as f: 
            json.dump(file_content, f, indent=2, ensure_ascii=False)

#create_individual_doccano_files(annotation, annotation_directory)
create_individual_doccano_files(annotation_val, annotation_val_directory)