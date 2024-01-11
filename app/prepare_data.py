import json 
import random

from pathlib import Path

"""# comparison-data
selina_annotation = Path(Path(__file__).parent, "../data/selina_200_annotation.jsonl")
save_selina_annotation = Path(Path(__file__).parent, "../data/selina_tuple_annotation.json")
sorted_selina_annotation = Path(Path(__file__).parent, "../data/selina_sorted_tuple_annotation.json")

teppi_annotation = Path(Path(__file__).parent, "../data/teppi_200_annotation.jsonl")
save_teppi_annotation = Path(Path(__file__).parent, "../data/teppi_tuple_annotation.json")
sorted_teppi_annotation = Path(Path(__file__).parent, "../data/teppi_sorted_tuple_annotation.json")"""



def transform_into_spacy_format(filepath: str, savepath: str) -> None:  
    """ Transforms annotated data into spacy format. 
        The annotated set has the annotated information saved in key labels.  

    Args:
        filepath (str): Path of data being transformed into spacy format. 
        savepath (str): Savepath of transformed data.

    Returns:
        json: Json with correct spaCy format. 
    """    
    # Create empty list where annotated data will be saved in. 
    annotated_data = []

    # Open file of annotated data and save lines in annotated_data.
    with open(filepath, "r") as f: 
        for line in f: 
            data = json.loads(line)
            annotated_data.append(data)

    # Create empty list for formatted transformed_data.
    transformed_data = []

    # Iterate over annotated_data.
    for doc in annotated_data:
        # Create empty list.
        formatted_data = []        
        # Save text of document in variable and append it in to list. 
        text = doc["text"]  
        formatted_data.append(text)

        # Create temporary dictionary. 
        temp_dict = {"entities": []}
        # Iterate over "labels" in the document.
        for label in doc["label"]: 
            # Save necessary labels in variable. 
            start = label[0]
            end = label[1]
            label = label[2]

            # Create tuple. Entity "label" needs to be upper case.
            entities = (start, end, label.upper())
            
            # Save tuple in temporary dictionary.
            temp_dict["entities"].append(entities)
        
        # Add temporary dictionary to list. 
        formatted_data.append(temp_dict)

        # Append newly formatted document in training_data.
        transformed_data.append(formatted_data)
    
    # Save training_data in a new file. 
    with open(savepath, "w", encoding="utf-8") as f: 
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)


def sort_dataset(file_path: str, save_path: str) -> None:
    """ Sort a dataset for a possible comparison.

    Args:
        file_path (str): Path to the dataset which should be sorted.
        save_path (str): Save path for the sorted dataset. 
    """    
    # Open file. 
    with open(file_path, "r") as f: 
        data = json.load(f)
    
    # Sort data and save it. 
    sorted_data = sorted(data, key=lambda item: item["text"])

    # Save sorted data in new file. 
    with open(save_path, "w", encoding="utf-8") as f: 
        json.dump(sorted_data, f, indent=2, ensure_ascii=False)


def split_dict(data: json, ratios: tuple[float, float, float] = (0.7, 0.1, 0.2)) -> tuple[dict, dict, dict]:
    """Split a dictionary randomly into training, evaluation and test data.

    Args:
        dictionary (dict): Preprocessed dictionary with filename as key and value is text.
        train_ratio (float, optional): Ratio with which the dictionary should be split. Defaults to 0.7, 0.1, 0.2.

    Returns:
        tuple[dict, dict]: Three dictionaries: training data, evaluation data and test data.
    """

    # Shuffle the data. 
    random.shuffle(data)
    # Find index on which the dictionary should be split.
    split1 = int(len(data) * ratios[0])
    split2 = split1 + int(len(data) * ratios[1])
    
    # Split data into three parts. 
    train_data = data[:split1]
    evaluation_data = data[split1:split2]
    test_data = data[split2:]

    return train_data, evaluation_data, test_data


def save_file(data: list, save_path: str) -> None: 
    """ Save data to a json-file.

    Args:
        data (list): Data to save. 
        save_path (str): Save path where data should be saved at. 
    """    
    with open(save_path, "w", encoding="utf-8") as f: 
        json.dump(data, f, indent=2, ensure_ascii=False)
    

"""# used 06.01.24
transform_into_spacy_format(selina_annotation, save_selina_annotation)
transform_into_spacy_format(teppi_annotation, save_teppi_annotation)
        
sort_dataset(save_selina_annotation, sorted_selina_annotation)
sort_dataset(save_teppi_annotation, sorted_teppi_annotation)"""

# used 10.01.2024

annotated_data = Path(Path(__file__).parent, "../data/1000_annotation.jsonl")
transformed_annotated_data = Path(Path(__file__).parent, "../data/transformed_1000_annotation.json")

transform_into_spacy_format(annotated_data, transformed_annotated_data)

with open(transformed_annotated_data, "r") as f: 
    data = json.load(f)

train_data, evaluation_data, test_data = split_dict(data)

savepath_train = Path(Path(__file__).parent, "../data/train_data.json")
savepath_eval = Path(Path(__file__).parent, "../data/eval_data.json")
savepath_test = Path(Path(__file__).parent, "../data/test_data.json")

save_file(train_data, savepath_train)
save_file(evaluation_data, savepath_eval)
save_file(test_data, savepath_test)

