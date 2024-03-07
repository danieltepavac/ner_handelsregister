# Import necessary libraries.
import json
import logging
from pathlib import Path

import spacy
from spacy.language import Language
from spacy.scorer import Scorer
from spacy.training.example import Example


def configure_logger():
    """ Configuration of a looger. 
    """ 
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler and set the logging level
    logging_file = Path(Path(__file__).parent, "../data/training_log.txt")
    file_handler = logging.FileHandler(logging_file)
    file_handler.setLevel(logging.INFO)

    # Create a console handler and set the logging level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and associate it with the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Initialize logger.
configure_logger()

def open_json(path: str) -> json:
    """Open and read a json file.

    Args:
        path (str): Path to json file.

    Returns:
        json: Read in json file. 
    """ 
    with open(path, "r") as f:
        data = json.load(f)
    return data


def evaluate(test_path: str, model_dir: str, save_path_result: str):
    """Evalute a model with test data. Input are paths to the respective objects. 

    Args:
        test_path (str): Path of train data.
        model_dir (str): Directory where the model is saved.
        save_path_result (str): Path where result should be saved. 
    """    
    # Open and read test data.
    logging.info("Loading test data...")
    TEST_DATA = open_json(test_path)

    # Load trained model.
    logging.info("Loading model from '%s'...", model_dir)
    nlp = spacy.load(model_dir)

    # Create Example-objects for evaluation.
    TEST_DATA = [Example.from_dict(nlp(text), annotations) for text, annotations in TEST_DATA]
    
    # Initialize scorer.
    scorer = Scorer()

    # Apply the scorer on list of examples.
    scores = scorer.score(examples=TEST_DATA)

    # Save results in a dictionary. 
    results = {
        "Precision": scores["ents_p"],
        "Recall": scores["ents_r"],
        "F1 score": scores["ents_f"],
        "Entities per type": scores["ents_per_type"]
    }

    # Log results. 
    logging.info("Evaluation results: Precision: %s, Recall: %s, F1 score: %s", scores["ents_p"], scores["ents_r"], scores["ents_f"])
    logging.info("Entities per type: %s", scores["ents_per_type"])

    # Save results. 
    save_result(save_path_result, results)

def evaluate_trf(data: json, nlp: Language): 
    """Evaluate a model with test data. 

    Args:
        data (json): Test data. 
        nlp (Language): Model which should be evaluated. 
    """    
    # Load trained model. 
    logging.info("Loading model from '%s'...", nlp)
    nlp = spacy.load(nlp)

    # Create Example-objects for evaluation.
    TEST_DATA = [Example.from_dict(nlp(text), annotations) for text, annotations in data]

    # Initialize scorer.
    scorer = Scorer()

    # Apply the scorer on list of examples.
    scores = scorer.score(examples=TEST_DATA)
                          
    # Save results in a dictionary. 
    results = {
        "Precision": scores["ents_p"],
        "Recall": scores["ents_r"],
        "F1 score": scores["ents_f"],
        "Entities per type": scores["ents_per_type"]
    }

    # Log results. 
    logging.info("Evaluation results: Precision: %s, Recall: %s, F1 score: %s", scores["ents_p"], scores["ents_r"], scores["ents_f"])
    logging.info("Entities per type: %s", scores["ents_per_type"])

    # Return results.
    return results


def save_result(save_path: str, obj_to_save: object):
    """ Open save_path and save object in it. 


    Args:
        save_path (str): Path where the object should be saved at. 
        obj_to_save (_type_): Object to be saved. 
    """

    with open(save_path, "w", encoding="utf-8") as f: 
        json.dump(obj_to_save, f, indent=2, ensure_ascii=False)

# Test data for each annotation sets. 
specific_test_path = Path(Path(__file__).parent, "../data/experiment1_specific_annotation/test_data.json")
general_test_path = Path(Path(__file__).parent, "../data/experiment2_general_annotation/test_data.json")

def main_blank_specific(): 
    specific_blank_model = Path(Path(__file__).parent, "../models/specific/blank_val")
    
    save_path_result = Path(Path(__file__).parent, "../results/exp1/specific_blank.json")

    evaluate(specific_test_path, specific_blank_model, save_path_result)

def main_ft_specific(): 
    specific_finetuned_model = Path(Path(__file__).parent, "../models/specific/lm_val")

    save_path_result = Path(Path(__file__).parent, "../results/exp1/specific_ft.json")

    evaluate(specific_test_path, specific_finetuned_model, save_path_result)

def main_trf_specific():
    # Transformer models need an separate function because the test data needs to be split so it can be processed with the machine at hand. 
    specific_transformer_model = Path(Path(__file__).parent, "../models/specific/transformers")

    TEST_DATA = open_json(specific_test_path)
    
    half_length = len(TEST_DATA) // 2
    first_half = TEST_DATA[:half_length]
    second_half = TEST_DATA[half_length:]

    first_half_result = evaluate_trf(first_half, specific_transformer_model)
    second_half_result = evaluate_trf(second_half, specific_transformer_model)


    combined_dict = {
    "Precision": (first_half_result["Precision"] + second_half_result["Precision"]) / 2,
    "Recall": (first_half_result["Recall"] + second_half_result["Recall"]) / 2,
    "F1 score": (first_half_result["F1 score"] + second_half_result["F1 score"]) / 2,
    "Entities per type": {}
    }

    for key, value in first_half_result["Entities per type"].items():
    # Check if the key exists in the existing dictionary
        if key in combined_dict["Entities per type"]:
            # Update the new dictionary by averaging the values
            combined_dict["Entities per type"][key] = {
                "p": (first_half_result["Entities per type"][key]["p"] + value["p"]) / 2,
                "r": (first_half_result["Entities per type"][key]["r"] + value["r"]) / 2,
                "f": (first_half_result["Entities per type"][key]["f"] + value["f"]) / 2
            }
        else:
            # If the key doesn"t exist, simply add it to the new dictionary
            combined_dict["Entities per type"][key] = value
    
    with open(Path(Path(__file__).parent, "../results/exp1/specific_trf.json"), "w", encoding="utf-8") as f: 
        json.dump(combined_dict, f, indent=2, ensure_ascii=False)

def main_blank_general(): 
    general_blank_model = Path(Path(__file__).parent, "../models/general/blank_val")

    save_path_result = Path(Path(__file__).parent, "../results/exp2/general_blank.json")

    evaluate(general_test_path, general_blank_model, save_path_result)

def main_ft_general(): 
    general_finetuned_model = Path(Path(__file__).parent, "../models/general/lm_val")

    save_path_result = Path(Path(__file__).parent, "../results/exp2/general_ft.json") 

    evaluate(general_test_path, general_finetuned_model, save_path_result)

def main_trf_general():
    # Transformer models need an separate function because the test data needs to be split so it can be processed with the machine at hand. 
    general_transformer_model = Path(Path(__file__).parent, "../models/general/transformers")

    TEST_DATA = open_json(general_test_path)
    
    half_length = len(TEST_DATA) // 2
    first_half = TEST_DATA[:half_length]
    second_half = TEST_DATA[half_length:]

    first_half_result = evaluate_trf(first_half, general_transformer_model)
    second_half_result = evaluate_trf(second_half, general_transformer_model)


    combined_dict = {
    "Precision": (first_half_result["Precision"] + second_half_result["Precision"]) / 2,
    "Recall": (first_half_result["Recall"] + second_half_result["Recall"]) / 2,
    "F1 score": (first_half_result["F1 score"] + second_half_result["F1 score"]) / 2,
    "Entities per type": {}
    }

    for key, value in first_half_result["Entities per type"].items():
    # Check if the key exists in the existing dictionary
        if key in combined_dict["Entities per type"]:
            # Update the new dictionary by averaging the values
            combined_dict["Entities per type"][key] = {
                "p": (first_half_result["Entities per type"][key]["p"] + value["p"]) / 2,
                "r": (first_half_result["Entities per type"][key]["r"] + value["r"]) / 2,
                "f": (first_half_result["Entities per type"][key]["f"] + value["f"]) / 2
            }
        else:
            # If the key doesn"t exist, simply add it to the new dictionary
            combined_dict["Entities per type"][key] = value
    
    with open(Path(Path(__file__).parent, "../results/exp2/general_trf.json"), "w", encoding="utf-8") as f: 
        json.dump(combined_dict, f, indent=2, ensure_ascii=False)


if __name__ == "__main__": 
    main_trf_general()





