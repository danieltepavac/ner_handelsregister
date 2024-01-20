import json
from pathlib import Path

import spacy

from spacy.tokens import DocBin


# Load German Transformer.
nlp = spacy.load("de_dep_news_trf")

# Path to files. 
detailed_path_train = Path(Path(__file__).parent, "../data/experiment1_detailed_annotation/train_data.json")
detailed_path_dev = Path(Path(__file__).parent, "../data/experiment1_detailed_annotation/eval_data.json")

broader_path_train = Path(Path(__file__).parent, "../data/experiment2_broader_annotation/train_data.json")
broader_path_dev = Path(Path(__file__).parent, "../data/experiment2_broader_annotation/eval_data.json")

# Save path files. 
detailed_save_path_train = Path(Path(__file__).parent, "../data/experiment1_detailed_annotation/train.spacy")
detailed_save_path_dev = Path(Path(__file__).parent, "../data/experiment1_detailed_annotation/dev.spacy")

broader_save_path_train = Path(Path(__file__).parent, "../data/experiment2_broader_annotation/train.spacy")
broader_save_path_dev = Path(Path(__file__).parent, "../data/experiment2_broader_annotation/dev.spacy")


def create_spacy_files(path_to_data: str, save_path: str) -> None:
    """Create .spacy files for training a custom model with transformers for ner. 

    Args:
        path_to_data (str): Path to the file being transformed. 
        save_path (str): Save path where the transformed data should be stored in. 
    """    

    with open(path_to_data, "r") as f: 
        data = json.load(f)

    # Initialize a list to store Doc objects.
    docs = []

    # Iterate over data. 
    for text, annotations in data: 
        # Process text with spaCy. 
        doc = nlp(text)
        
        # Save entities in a list after iteration. 
        entities = [(start, end, label) for start, end, label in annotations["entities"]]

        # Initialize empty span-list. 
        spans = []

        # Iterate over entities. 
        for entity in entities: 
            # Save start, end, label in respective variable. 
            start = entity[0]
            end = entity[1]
            label = entity[2]

            # Get the spans of the characters of labeled tokens. 
            span = doc.char_span(start, end, label)
            # Append it to the spans-list. 
            spans.append(span)
        
        # Filter out non-None elements and save it in new list. 
        filtered_spans = [span for span in spans if span is not None]

        # Add filtered spans of token to doc.ents.
        doc.ents = filtered_spans

        # Append modified Doc to docs-list. 
        docs.append(doc)

    # Create and save a collection of training docs. 
    docbin = DocBin(docs=docs)
    docbin.to_disk(save_path)
        

create_spacy_files(detailed_path_train, detailed_save_path_train)
create_spacy_files(detailed_path_dev, detailed_save_path_dev)
create_spacy_files(broader_path_train, broader_save_path_train)
create_spacy_files(broader_path_dev, broader_save_path_dev)