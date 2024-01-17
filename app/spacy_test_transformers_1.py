import spacy
from spacy.training.example import Example
from spacy.tokens import DocBin, Doc, Span

import json
from pathlib import Path

# Load spaCy English model
nlp = spacy.load("de_dep_news_trf")

path_train = Path(Path(__file__).parent, "../data/train_data.json")
path_dev = Path(Path(__file__).parent, "../data/eval_data.json")

with open(path_train, "r") as f: 
    TRAIN_DATA = json.load(f)

with open(path_dev, "r") as f: 
    DEV_DATA = json.load(f)



# Initialize a list to store Doc objects
docs_train = []



for text, annotations in TRAIN_DATA:
    # Process the text with spaCy
    doc = nlp(text)

    # Create a list of tuples (start, end, label)
    entities = [(start, end, label) for start, end, label in annotations['entities']]

    spans = []

    for entity in entities: 
        start = entity[0]
        end = entity[1]
        label = entity[2]

        span = doc.char_span(start, end, label)
        spans.append(span)

    filtered_spans = [span for span in spans if span is not None]

    doc.ents = filtered_spans
    
    # Append the modified Doc to the list
    docs_train.append(doc)

# Initialize a list to store Doc objects
docs_dev = []

for text, annotations in DEV_DATA:
    # Process the text with spaCy
    doc = nlp(text)
    
    # Create a list of tuples (start, end, label)
    entities = [(start, end, label) for start, end, label in annotations['entities']]
    
    spans = []

    for entity in entities: 
        start = entity[0]
        end = entity[1]
        label = entity[2]

        span = doc.char_span(start, end, label)
        spans.append(span)


    filtered_spans = [span for span in spans if span is not None]

    doc.ents = filtered_spans
    
    # Append the modified Doc to the list
    docs_dev.append(doc)



# Create and save a collection of training docs
train_docbin = DocBin(docs=docs_train)
train_docbin.to_disk("./train.spacy")


test_docbin = DocBin(docs=docs_dev)
test_docbin.to_disk("./dev.spacy")