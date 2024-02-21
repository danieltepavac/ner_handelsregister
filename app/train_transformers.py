import json
import spacy
from spacy_transformers import Transformer
from spacy.training.example import Example
from spacy.scorer import Scorer

from pathlib import Path
import random
import logging 

from tqdm import tqdm

def configure_logger():
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
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

configure_logger()

def open_json(path: str) -> json:
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_model(output_dir: str, nlp): 
    # Save spacy model. Check if "output_dir" exists. If not, then create it. 
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        # Save language model to disk in target output directory. 
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


def train_transformer(train_path: str, val_path: str, output_dir: str,  n_iter: int=100, early_stopping_rounds: int=5):

    logging.info("Loading training and validation data...")
    TRAIN_DATA = open_json(train_path)
    VAL_DATA = open_json(val_path)

    best_val_loss = 0
    rounds_without_improvement = 0  # Counter for consecutive rounds without improvement
    
    logging.info("Initializing spaCy pipeline...")
    nlp = spacy.blank("de")
    
    # Define transformer configuration
    config = {
        "model": {
            "@architectures": "spacy-transformers.TransformerModel.v3",
            "name": "bert-base-german-cased",
            "tokenizer_config": {"use_fast": True},
            "transformer_config": {"output_attentions": True},
            "mixed_precision": True,
            "grad_scaler_config": {"init_scale": 32768}
        }
    }
    
    nlp.add_pipe("transformer", config=config)
    nlp.add_pipe("ner")
    
    # Retrieve the NER component
    ner = nlp.get_pipe("ner")

    # Iterate over TRAIN_DATA to add entity labels to the NER component
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable other pipeline components during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        rounds_without_improvement = 0
        previous_val_loss = float('inf')
        for itn in range(n_iter):
            # Shuffle training data for each iteration
            random.shuffle(TRAIN_DATA)
            losses = {}
            # Create Example objects and update the model
            for text, annotations in tqdm(TRAIN_DATA):
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, losses=losses)

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

            logging.info("Iteration %s: Training Loss: %s, Validation Loss: %s", itn + 1, losses['ner'], avg_val_loss)

            if avg_val_loss > best_val_loss:
                best_val_loss = avg_val_loss
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
                logging.info("%s rounds without improvement", rounds_without_improvement)

            if rounds_without_improvement >= early_stopping_rounds:
                logging.info("Validation loss hasn't improved for %s rounds. Stopping training.", early_stopping_rounds)
                break  # Early stopping

    
    save_model(output_dir, nlp)

detailed_train_path = Path(Path(__file__).parent, "../data/experiment1_detailed_annotation/train_data.json")
detailed_val_path = Path(Path(__file__).parent, "../data/experiment1_detailed_annotation/val_data.json")

broader_train_path = Path(Path(__file__).parent, "../data/experiment2_broader_annotation/train_data.json")
broader_val_path = Path(Path(__file__).parent, "../data/experiment2_broader_annotation/val_data.json")
broader_test_path = Path(Path(__file__).parent, "../data/experiment2_broader_annotation/test_data.json")

save_transformers = Path(Path(__file__).parent, "../new_models/broader/transformers")
save_path_result = Path(Path(__file__).parent, "../new_results/exp2/broader_transformer.json")

train_transformer(broader_train_path, broader_val_path, save_transformers)