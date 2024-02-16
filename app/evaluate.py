# Import necessary libraries.
import json

from pathlib import Path

import spacy
from spacy.training.example import Example
from spacy.scorer import Scorer

def open_json(path: str) -> json:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def evaluate(test_path:str, model_dir: str): 

    TEST_DATA = open_json(test_path)

    nlp = spacy.load(model_dir)
    # Create empty list of examples. 
    examples = []

    # Iterate over EVALUATION_DATA_TUPLES.
    for text, annotations in TEST_DATA: 
        # Apply custom model on every text. 
        doc = nlp(text)
        # Save predictions and gold standard in example. 
        example = Example.from_dict(doc, annotations)
        # Save individual example in list of examples. 
        examples.append(example)

    # Save the scorer. 
    scorer = Scorer()

    # Apply the scorer on list of examples. 
    scores = scorer.score(examples=examples)

    results = {
        'Precision': scores["ents_p"],
        'Recall': scores["ents_r"],
        'F1 score': scores["ents_f"]
    }

    return results

    # Print evaluation metrics. 
    print("Precision:", scores["ents_p"])
    print("Recall:", scores["ents_r"])
    print("F1 score:", scores["ents_f"])
    print("Ents per type:", scores["ents_per_type"])

def save(save_path:str, obj_to_save):

    with open(save_path, "w", encoding="utf-8") as f: 
        json.dump(obj_to_save, f, indent=2, ensure_ascii=False)


