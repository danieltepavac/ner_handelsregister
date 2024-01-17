import json
from pathlib import Path
import spacy
from spacy.training.example import Example
from spacy.scorer import Scorer

# Save directory model is saved at.
model_directory = Path(Path(__file__).parent, "output_plain_de/model-best")

# Load the trained model from the output directory.
nlp = spacy.load(model_directory)

# Path to evaluation data.
test_path = Path(Path(__file__).parent, "../data/test_data.json")

# Open EVALUATION_DATA.
with open(test_path, "r") as f:
    TEST_DATA = json.load(f)

# Create empty list of examples.
examples = []

# Iterate over TEST_DATA.
for text, annotations in TEST_DATA:
    # Apply the custom model on every text.
    doc = nlp(text)
    # Save predictions and gold standard in example.
    example = Example.from_dict(doc, annotations)
    # Save individual example in the list of examples.
    examples.append(example)

# Save the scorer.
scorer = Scorer()

# Apply the scorer on the list of examples.
scores = scorer.score(examples)

# Print evaluation metrics.
print("Precision:", scores["ents_p"])
print("Recall:", scores["ents_r"])
print("F1 score:", scores["ents_f"])


