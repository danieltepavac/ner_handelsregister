# Import necessary libraries.
import json

from pathlib import Path

import spacy
from spacy.training.example import Example
from spacy.scorer import Scorer


# Save directory model is saved at. 
model_directory = Path(Path(__file__).parent, "ner1_finetuned_broad")

# Load the trained model from the output directory.
nlp = spacy.load(model_directory)

# Path to evaluation data. 
test_path = Path(Path(__file__).parent, "../data/broader_test_data.json")

# Open EVALUATION_DATA.
with open(test_path, "r") as f:
    TEST_DATA = json.load(f)

# Transform EVALUATION_DATA into list of tuples. Necessary step for creating Example-Objects.
#EVALUATION_DATA_TUPLES = [tuple(i) for i in EVALUATION_DATA]

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

# Print evaluation metrics. 
print("Precision:", scores["ents_p"])
print("Recall:", scores["ents_r"])
print("F1 score:", scores["ents_f"])





