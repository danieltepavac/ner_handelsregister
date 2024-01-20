import json
import warnings

from pathlib import Path

import spacy
from spacy.training import offsets_to_biluo_tags

model_directory = Path(Path(__file__).parent, "../models/experiment1_detailed_annotation/ner1_without_eval")

test_path = Path(Path(__file__).parent, "../data/experiment1_detailed_annotation/test_data.json")
train_path = Path(Path(__file__).parent, "../data/experiment1_detailed_annotation/train_data.json")
eval_path = Path(Path(__file__).parent, "../data/experiment1_detailed_annotation/eval_data.json")


def count_alignment_error(path: str, model_dir: str) -> int:
    """ Count the occurances of the alignment error. 

    Args:
        path (str): Path to affected file. 
        model_dir (str): Directory where the model is found. 

    Returns:
        int: Occurances of alignment error. 
    """    

    # Initialize counter.
    warning_count = 0

    # New function for a warning handler. #TODO: How to fill out docstring? 
    def alignment_warning_handler(message, category, filename, lineno, file=None, line=None):
        """_summary_

        Args:
            message (_type_): _description_
            category (_type_): _description_
            filename (_type_): _description_
            lineno (_type_): _description_
            file (_type_, optional): _description_. Defaults to None.
            line (_type_, optional): _description_. Defaults to None.
        """ 
        # Declare warning count as global variable.    
        nonlocal warning_count

        # Extract the filename from the Path object.
        file_path = Path(filename)
        # Stem provides the filename without the extension. 
        file_name = file_path.stem  

        print(f"Processing file: {file_name}")

        # Check if the warning message contains the specific text you want to catch.
        if "Some entities could not be aligned" in str(message):
            # Increment the warning count.
            warning_count += 1
            # Handle the warning as you see fit.
            print(f"Custom warning handler: Ignoring misaligned entities warning. Count: {warning_count}")
        else:
            # Handle other warnings by printing them.
            warnings.showwarning(message, category, filename, lineno, file, line)

    # Register the custom warning handler which invokes alignment_warning_handler instead of the default one. 
    warnings.showwarning = alignment_warning_handler


    def BIO_tags(path: str, model_dir: str) -> None: 
        """ Show BIO-tags of annotated data to catch alignment errors. 

        Args:
            path (str): Path of affected file. 
            model_dir (str): Directory where the model can be found. 
        """        

        # Open file. 
        with open(path, "r") as f: 
            DATA = json.load(f)
        
        # Load model.
        nlp = spacy.load(model_dir)
        
        # Iterate over the data and save tags. 
        for text, entities in DATA:
            tags = offsets_to_biluo_tags(nlp.make_doc(text), entities.get("entities"))

    # Execute function. 
    BIO_tags(path, model_dir)

    # Return warning_count. 
    return warning_count 

def count_annotated_entities(path: str) -> int: 
    """ Count annotated entities. 

    Args:
        path (str): Path to concerned file. 

    Returns:
        int: Count of all entities. 
    """

    # Open file. 
    with open(path, "r") as f: 
        DATA = json.load(f)
    
    # Initialize occurances.
    occurances = 0

    # Iterate over data and store each occurance in occurances. 
    for _, entities in DATA: 
        length = len(entities.get("entities"))
        occurances += length
    
    return occurances

def percentage(warning_count: int, entities_count: int) -> float: 
    """ Calculate the percentage of entities throwing an alignment error. 

    Args:
        warning_count (int): Count of occurance of alignment error. 
        entities_count (int): Count of all entities. 

    Returns:
        float: Percentage of how many entities overall are affected.
    """
    # Calculate percentage. 
    return warning_count/entities_count * 100

# Count alignment warnings. 
warning_count_train = count_alignment_error(train_path, model_directory)
warning_count_test = count_alignment_error(test_path, model_directory)
warning_count_eval = count_alignment_error(eval_path, model_directory)

# Count entities.
entities_count_train = count_annotated_entities(train_path)
entities_count_test = count_annotated_entities(test_path)
entities_count_eval = count_annotated_entities(eval_path)

# Calculate percentage.
percentage_train = percentage(warning_count_train, entities_count_train)
percentage_test = percentage(warning_count_test, entities_count_test)
percentage_eval = percentage(warning_count_eval, entities_count_eval)

# Save calculated results. 
data = {
    "count_train": {
        "warning": warning_count_train,
        "entities": entities_count_train,
        "error_percentage": percentage_train
    },
    "count_test": {
        "warning": warning_count_test,
        "entities": entities_count_test,
        "error_percentage": percentage_test
    },
    "count_eval": {
        "warning": warning_count_eval,
        "entities": entities_count_eval,
        "error_percentage": percentage_eval
    }
}

# Save path where calculated results should be saved in. 
save_path = Path(Path(__file__).parent, "../results/error_analysis.json")

# Save calculated results in json-file. 
with open(save_path, "w", encoding="utf-8") as f: 
    json.dump(data, f, indent=2, ensure_ascii=False)


