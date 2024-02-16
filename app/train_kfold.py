import logging

from pathlib import Path
import json

# Package for training a language model. 
import spacy
from spacy.training.example import Example

from spacy_transformers import Transformer
from spacy_transformers.pipeline_component import DEFAULT_CONFIG

# Necessary packages for training.
from spacy.scorer import Scorer
from tqdm import tqdm

from sklearn.model_selection import KFold

def configure_logger():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler and set the logging level
    file_handler = logging.FileHandler('training_log.txt')
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

def train_blank(data: json, output_dir: str, n_iter=50, patience=5, n_folds=5):

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []


    for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
        logging.info("Training on Fold %d", fold)

        model = None
        best_loss = float('inf')

        # Load model. Check if it is None, if so, then create an empty German Language Model. If not, then load the model. 
        if model is not None:
            nlp = spacy.load(model)  
            print("Loaded model '%s'" % model)
        else:
            nlp = spacy.blank("de")  
            print("Created blank 'de' model")

        # Set up pipeline by adding "ner" to pipeline if it is not present. Otherwise, just get the "ner"-pipeline.
        if 'ner' not in nlp.pipe_names:
            # "last=True" adds pipeline at the last possible position.
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")

        for _, annotations in data:
        # Get "entities" of annotation and add them to the ner model. 
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # Assure that only ner is present as pipeline by saving them in "other_pipes"..
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        # ..and disabling them.
        with nlp.disable_pipes(*other_pipes):
            # Start training process and saving the result in an optimizer.  
            optimizer = nlp.begin_training()

            train_data = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in [data[i] for i in train_idx]]
            
            last_avg_loss = float("inf")
            patience_ = patience

            for iteration in range(n_iter):
                losses = {}
                for example in tqdm(train_data, desc=f"Training (Fold {fold}), Iteration {iteration + 1}"):
                    loss = nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)

                # Print the loss after each iteration
                avg_loss = losses["ner"] / len(train_data)
                logging.info("Fold %d, Iteration %d - Average Loss: %.6f", fold, iteration + 1, avg_loss)

                # Check if the current model has the best loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_path = f"{output_dir}/best_model_fold_{fold}.spacy"
                    nlp.to_disk(best_model_path)
                    logging.info("Best model saved for Fold %d with average loss: %.6f", fold, best_loss)
                
                if avg_loss < last_avg_loss:
                    last_avg_loss = avg_loss
                    patience_ = 0
                else:
                    patience_ += 1   
                    print(f"patience count increased to {patience_}")                
                                            
                if patience_ >= patience:
                    break
            
            # Save the last model for each fold
            last_model_path = f"{output_dir}/last_model_fold_{fold}.spacy"
            nlp.to_disk(last_model_path)
            logging.info("Last model saved for Fold %d", fold)

            best_nlp = spacy.load(best_model_path)
            results.append(evaluate(data, best_nlp, test_idx))

    logging.info("Training complete.")
    return results

def train_german_lm(data: json, output_dir: str, n_iter=50, patience=5, n_folds=5):

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []


    for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
        logging.info("Training on Fold %d", fold)

        model = None
        best_loss = float('inf')

        if model is not None:
            nlp = spacy.load(model)  
            print("Loaded model '%s'" % model)
        else:
            nlp = spacy.load('de_core_news_lg')  
            print("Loaded 'de_core_news_lg' model")

        if "transformer" not in nlp.pipe_names: 
            nlp.add_pipe("transformers", config=DEFAULT_CONFIG, first=True)
        else: 
            nlp.add_pipe("transformers", config=DEFAULT_CONFIG, first=True)

        # Set up pipeline by adding "ner" to pipeline if it is not present. Otherwise, just get the "ner"-pipeline.
        if 'ner' not in nlp.pipe_names:
            # "last=True" adds pipeline at the last possible position.
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")

        for _, annotations in data:
        # Get "entities" of annotation and add them to the ner model. 
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # Assure that only ner is present as pipeline by saving them in "other_pipes"..
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        # ..and disabling them.
        with nlp.disable_pipes(*other_pipes):
            # Start training process and saving the result in an optimizer.  
            optimizer = nlp.create_optimizer()

            train_data = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in [data[i] for i in train_idx]]
            
            last_avg_loss = float("inf")
            patience_ = patience

            for iteration in range(n_iter):
                losses = {}
                for example in tqdm(train_data, desc=f"Training (Fold {fold}), Iteration {iteration + 1}"):
                    loss = nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)

                # Print the loss after each iteration
                avg_loss = losses["ner"] / len(train_data)
                logging.info("Fold %d, Iteration %d - Average Loss: %.6f", fold, iteration + 1, avg_loss)

                # Check if the current model has the best loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_path = f"{output_dir}/best_model_fold_{fold}.spacy"
                    nlp.to_disk(best_model_path)
                    logging.info("Best model saved for Fold %d with average loss: %.6f", fold, best_loss)
                
                if avg_loss < last_avg_loss:
                    last_avg_loss = avg_loss
                    patience_ = 0
                else:
                    patience_ += 1   
                    print(f"patience count increased to {patience_}")                
                                            
                if patience_ >= patience:
                    break
            
            # Save the last model for each fold
            last_model_path = f"{output_dir}/last_model_fold_{fold}.spacy"
            nlp.to_disk(last_model_path)
            logging.info("Last model saved for Fold %d", fold)

            best_nlp = spacy.load(best_model_path)
            results.append(evaluate(data, best_nlp, test_idx))

    logging.info("Training complete.")
    return results

def evaluate(data, nlp, test_idx): 

    test_data = [Example.from_dict(nlp(text), annotations) for text, annotations in [data[i] for i in test_idx]]

    scorer = Scorer()

    scores = scorer.score(examples=test_data)
                          
    logging.info("Precision: %f", scores["ents_p"])
    logging.info("Recall: %f", scores["ents_r"])
    logging.info("F1 score: %f", scores["ents_f"]) 

    results = {
    'Precision': scores["ents_p"],
    'Recall': scores["ents_r"],
    'F1 score': scores["ents_f"]
    }

    return results

def main():
    # Specify the paths and parameters
    data_path = Path(Path(__file__).parent, "../data/cross_validation/all_data_broad.json")
    output_dir = Path(Path(__file__).parent, "../models/cross_validation/b3")
    n_iter = 1000
    patience = 5
    n_folds = 5

    # Load your data
    data = open_json(data_path)

    # Train the model
    x = train_blank(data, output_dir, n_iter, patience, n_folds)

if __name__ == "__main__":
    main()