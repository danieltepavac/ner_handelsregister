# Source: https://blog.futuresmart.ai/building-a-custom-ner-model-with-spacy-a-step-by-step-guide
# Training of own model with train and val_data. 

# Necessary packages to load JSON file.
from pathlib import Path
import json

# Package for training a language model. 
import spacy
from spacy.training.example import Example

# Necessary packages for training.
import random
from tqdm import tqdm

# Save relative path of target JSON file. 
train_path = Path(Path(__file__).parents[1], "../data/experiment1_detailed_annotation/train_data.json")
val_path = Path(Path(__file__).parent[1], "../data/experiment1_detailed/annotation/eval_data.json")

# Read train_path and save it as TRAIN_DATA.
with open(train_path, "r") as f: 
    TRAIN_DATA = json.load(f)

# Read val_path and save it as VAL_DATA
with open(val_path, "r") as f: 
    VAL_DATA = json.load(f)

# Initialize empty model. 
model = None
# Create Path-Objekt to save NER model in.
output_dir=Path(Path(__file__).parents[1], "../models/experiment1_detailed_annotation/ner2_finetuned/")
# Set training iterations to 100.
n_iter=100

# Load model. Check if it is None, if so, then create an empty German Language Model. If not, then load the model. 
if model is not None:
    nlp = spacy.load(model)  
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.load("de_core_news_lg") 
    print("Loaded 'de_core_news_lg' model")

# Set up pipeline by adding "ner" to pipeline if it is not present. Otherwise, just get the "ner"-pipeline.
if "ner" not in nlp.pipe_names:
    # "last=True" adds pipeline at the last possible position.
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Iterate through TRAIN_DATA by focusing on "annotations". _ means other parts are denoted because their irrelevant for the loop.
for _, annotations in TRAIN_DATA:
    # Get "entities" of annotation and add them to the ner model. 
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])
# Create empty list for example. 
example = []

# Assure that only ner is present as pipeline by saving them in "other_pipes"..
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
# ..and disabling them.
with nlp.disable_pipes(*other_pipes):
    # Start training process and saving the result in an optimizer.  
    optimizer = nlp.create_optimizer()
    # Iterate over the amount of iterations already set. "itn" is an individual iteration.
    for itn in range(n_iter):
        # Shuffle TRAIN_DATA.
        random.shuffle(TRAIN_DATA)
        # Create empty dictionary to save losses in. 
        losses = {}
        # Iterate over TRAIN_DATA with implemented progress bar. 
        for text, annotations in tqdm(TRAIN_DATA):
            # Take input text and create a DOC-object. 
            doc = nlp.make_doc(text)
            # Create spacy EXAMPLE-object based on the created DOC-object and annotations.
            # EXAMPLE-object: container holding a processed document/text as well as their annotations. 
            example = Example.from_dict(doc, annotations)
            # Update the ner model and its parameters with the use of the training example.
            nlp.update(
                # Take a single example object. 
                [example], 
                # Introduce a dropout of 0.5. Probability of dropping a neuron in the neural network. 
                drop=0.5,
                # Choose as optimizer Stochastic Gradient Descent. 
                sgd=optimizer,
                # Accumulate losses. 
                losses=losses)
        
        # Initialize total_val_loss starting by 0 and adding 1 for each iteration.
        total_val_loss = 0
        # Iterate over VAL_DATA.
        for text, annotations in VAL_DATA: 
            # Create Doc- as well es Example-Object out of VAL_DATA.
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            # Save evaluation in a dict. 
            eval_metrics = nlp.evaluate([example])
            # Comparing the model's predictions with correct annotations and computing loss based on "ents_f". 
            val_loss = eval_metrics.get("ents_f", 0.0)
            # Add current loss to total. 
            total_val_loss += val_loss
        
        # Calculate average validation loss. 
        avg_val_loss = total_val_loss / len(VAL_DATA)

        print(f"Interation {itn + 1}: Training Loss: {losses['ner']}, Validation Loss: {avg_val_loss}")


# Save spacy model. Check if "output_dir" exists. If not, then create it. 
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    # Save language model to disk in target output directory. 
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)