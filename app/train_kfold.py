import json
import logging

from pathlib import Path

from numpy import ndarray

# Necessary packages for training.
from sklearn.model_selection import KFold
from tqdm import tqdm

import spacy
from spacy.scorer import Scorer
from spacy.training.example import Example


def configure_logger():
    """ Configuration of a looger. 
    """   

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler and set the logging level.
    file_handler = logging.FileHandler("training_log.txt")
    file_handler.setLevel(logging.INFO)

    # Create a console handler and set the logging level.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and associate it with the handlers.
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger.
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

def cross_validation_blank(data: json, output_dir: str, save_path_results: str, n_iter:int =100, patience:int =5, n_folds: int=5):
    """Cross-Validiation of the blank German Language Model. 

    Args:
        data (json): Data the model should be trained on. 
        output_dir (str): Directory where the model should be saved at.
        save_path_results (str): Path of the file with the results. 
        n_iter (int, optional): Training iteration. Defaults to 100.
        patience (int, optional): Number of iteration after the model should stop training if no improvement. Defaults to 5.
        n_folds (int, optional): Number of folds of Cross-Validation. Defaults to 5.
    """   

    # Initialize the folds for Cross-Validation.
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Create empty list for results. 
    results = []

    # Iterate over each fold. 
    for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
        logging.info("Training on Fold %d", fold)

        # Initialize empty model.
        model = None
        # Intialize loss for early stopping. 
        best_loss = float("inf")

        # Load model. Check if it is None, if so, then create an empty German Language Model. If not, then load the model. 
        if model is not None:
            nlp = spacy.load(model)  
            print("Loaded model '%s'" % model)
        else:
            nlp = spacy.blank("de")  
            print("Created blank 'de' model")

        # Set up pipeline by adding "ner" to pipeline if it is not present. Otherwise, just get the "ner"-pipeline.
        if "ner" not in nlp.pipe_names:
            # "last=True" adds pipeline at the last possible position.
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")
        
        # Iterate through TRAIN_DATA by focusing on "annotations". _ means other parts are denoted because their irrelevant for the loop.
        for _, annotations in data:
        # Get "entities" of annotation and add them to the ner model. 
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # Assure that only ner is present as pipeline by saving them in "other_pipes"..
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        # ..and disabling them.
        with nlp.disable_pipes(*other_pipes):
            # Start training process and saving the result in an optimizer.  
            optimizer = nlp.begin_training()

            # Create Example-object of this fold's train data. 
            TRAIN_DATA = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in [data[i] for i in train_idx]]
            
            # Initialize last average loss and patience. 
            last_avg_loss = float("inf")
            patience_ = patience

            # Iterate over number of iteration. 
            for iteration in range(n_iter):
                # Create empty list for losses. 
                losses = {}
                # Iterate over train data and update the loss. 
                for example in tqdm(TRAIN_DATA, desc=f"Training (Fold {fold}), Iteration {iteration + 1}"):
                    loss = nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)

                # Print the loss after each iteration and log it. 
                avg_loss = losses["ner"] / len(TRAIN_DATA)
                logging.info("Fold %d, Iteration %d - Average Loss: %.6f", fold, iteration + 1, avg_loss)

                # Check if the current model has the best loss. If so, save it. 
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_path = f"{output_dir}/best_model_fold_{fold}.spacy"
                    nlp.to_disk(best_model_path)
                    logging.info("Best model saved for Fold %d with average loss: %.6f", fold, best_loss)
                
                # Check if the loss has improved.
                if avg_loss < last_avg_loss:
                    last_avg_loss = avg_loss
                    patience_ = 0
                # If not, increase counter by 1.
                else:
                    patience_ += 1   
                    print(f"patience count increased to {patience_}")                

                # Check for early stopping.                       
                if patience_ >= patience:
                    break
            
            # Save the last model for each fold. 
            last_model_path = f"{output_dir}/last_model_fold_{fold}.spacy"
            nlp.to_disk(last_model_path)
            logging.info("Last model saved for Fold %d", fold)

            # Load the current best model. 
            best_nlp = spacy.load(best_model_path)
            # Evaluate after each fold. 
            evaluation_result = evaluate(data, best_nlp, test_idx)
            # Append result. 
            results.append(evaluation_result)
            logging.info("Evaluation result for Fold %d: %s", fold, str(evaluation_result))

    # Save results. 
    logging.info("Training complete.")
    with open(save_path_results, "w", encoding="utf-8") as f: 
                json.dump(results, f, indent=2, ensure_ascii=False)


def cross_validation_german_lm(data: json, output_dir: str, save_path_results: str, n_iter: int=100, patience: int=5, n_folds: int=5):
    """Cross-Validiation of the German Language Model de_core_news_lg.

    Args:
        data (json): Data the model should be trained on. 
        output_dir (str): Directory where the model should be saved at.
        save_path_results (str): Path of the file with the results. 
        n_iter (int, optional): Training iteration. Defaults to 100.
        patience (int, optional): Number of iteration after the model should stop training if no improvement. Defaults to 5.
        n_folds (int, optional): Number of folds of Cross-Validation. Defaults to 5.
    """ 

    # Initialize the folds for Cross-Validation.
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    # Create empty list for results.
    results = []

    # Iterate over each fold.
    for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
        logging.info("Training on Fold %d", fold)

        # Initialize empty model.
        model = None
        # Intialize loss for early stopping. 
        best_loss = float("inf")

        # Load model. Check if it is None, if so, then load the German Language Model. If not, then load the model.
        if model is not None:
            nlp = spacy.load(model)  
            print("Loaded model '%s'" % model)
        else:
            nlp = spacy.load("de_core_news_lg")  
            print("Loaded "de_core_news_lg" model")

        # Set up pipeline by adding "ner" to pipeline if it is not present. Otherwise, just get the "ner"-pipeline.
        if "ner" not in nlp.pipe_names:
            # "last=True" adds pipeline at the last possible position.
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")

        # Iterate through TRAIN_DATA by focusing on "annotations". _ means other parts are denoted because their irrelevant for the loop.
        for _, annotations in data:
        # Get "entities" of annotation and add them to the ner model. 
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # Assure that only ner is present as pipeline by saving them in "other_pipes"..
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        # ..and disabling them.
        with nlp.disable_pipes(*other_pipes):
            # Start training process and saving the result in an optimizer.  
            optimizer = nlp.create_optimizer()

            # Create Example-object of this fold's train data.
            TRAIN_DATA = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in [data[i] for i in train_idx]]
            
            # Initialize last average loss and patience.
            last_avg_loss = float("inf")
            patience_ = patience
            
            # Iterate over number of iteration.
            for iteration in range(n_iter):
                # Create empty list for losses.
                losses = {}
                # Iterate over train data and update the loss.
                for example in tqdm(TRAIN_DATA, desc=f"Training (Fold {fold}), Iteration {iteration + 1}"):
                    loss = nlp.update([example], drop=0.5, losses=losses, sgd=optimizer)

                # Print the loss after each iteration.
                avg_loss = losses["ner"] / len(TRAIN_DATA)
                logging.info("Fold %d, Iteration %d - Average Loss: %.6f", fold, iteration + 1, avg_loss)

                # Check if the current model has the best loss.
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_path = f"{output_dir}/best_model_fold_{fold}.spacy"
                    nlp.to_disk(best_model_path)
                    logging.info("Best model saved for Fold %d with average loss: %.6f", fold, best_loss)
                
                # Check if the loss has improved.
                if avg_loss < last_avg_loss:
                    last_avg_loss = avg_loss
                    patience_ = 0
                # If not, increase counter by 1.
                else:
                    patience_ += 1   
                    print(f"patience count increased to {patience_}")                

                # Check for early stopping.                    
                if patience_ >= patience:
                    break
            
            # Save the last model for each fold.
            last_model_path = f"{output_dir}/last_model_fold_{fold}.spacy"
            nlp.to_disk(last_model_path)
            logging.info("Last model saved for Fold %d", fold)

            # Load the current best model.
            best_nlp = spacy.load(best_model_path)
            # Evaluate after each fold and append it to result. 
            results.append(evaluate(data, best_nlp, test_idx))

    # Save results. 
    logging.info("Training complete.")
    with open(save_path_results, "w", encoding="utf-8") as f: 
        json.dump(results, f, indent=2, ensure_ascii=False)

def cross_validation_transformers(data: json, output_dir: str, save_path_results: str, n_iter: int=100, patience: int=5, n_folds:int =5): 
    """Cross-Validiation of the German Language Model bert-base-german-cased.

    Args:
        data (json): Data the model should be trained on. 
        output_dir (str): Directory where the model should be saved at.
        save_path_results (str): Path of the file with the results. 
        n_iter (int, optional): Training iteration. Defaults to 100.
        patience (int, optional): Number of iteration after the model should stop training if no improvement. Defaults to 5.
        n_folds (int, optional): Number of folds of Cross-Validation. Defaults to 5.
    """

    # Initialize the folds for Cross-Validation.
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    # Create empty list for results.
    results = []

    # Iterate over each fold.
    for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
        logging.info("Training on Fold %d", fold)

        # Intialize loss for early stopping.
        best_loss = float("inf")
        
        # Initialize blank German Language Model.
        logging.info("Initializing spaCy pipeline...")
        nlp = spacy.blank("de")

        # Define transformer configuration.
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

        # Add transformer component inclusive configuration to pipeline.
        nlp.add_pipe("transformer", config=config)
        # Add ner component to pipeline.
        nlp.add_pipe("ner")
        
        # Initialize pipeline.
        nlp.initialize()
        
        # Retrieve the NER component.
        ner = nlp.get_pipe("ner")

        # Iterate through TRAIN_DATA by focusing on "annotations". _ means other parts are denoted because their irrelevant for the loop.
        for _, annotations in data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # Assure that only ner is present as pipeline by saving them in "other_pipes"..
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        # ..and disabling them.
        with nlp.disable_pipes(*other_pipes):
            # Start training process and saving the result in an optimizer.  
            optimizer = nlp.begin_training()
            
            # Create Example-object of this fold's train data.
            TRAIN_DATA = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in [data[i] for i in train_idx]]
            
            # Initialize last average loss and patience.
            last_avg_loss = float("inf")
            patience_ = patience

            # Iterate over number of iteration.
            for iteration in range(n_iter):
                # Create empty list for losses.
                losses = {}
                # Iterate over train data and update the loss.
                for example in tqdm(TRAIN_DATA, desc=f"Training (Fold {fold}), Iteration {iteration + 1}"):
                    loss = nlp.update([example], drop=0.5, losses=losses)

                # Print the loss after each iteration.
                avg_loss = losses["ner"] / len(TRAIN_DATA)
                logging.info("Fold %d, Iteration %d - Average Loss: %.6f", fold, iteration + 1, avg_loss)

                # Check if the current model has the best loss.
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_path = f"{output_dir}/best_model_fold_{fold}.spacy"
                    nlp.to_disk(best_model_path)
                    logging.info("Best model saved for Fold %d with average loss: %.6f", fold, best_loss)
                
                # Check if the loss has improved.
                if avg_loss < last_avg_loss:
                    last_avg_loss = avg_loss
                    patience_ = 0
                # If not, increase counter by 1.
                else:
                    patience_ += 1   
                    print(f"patience count increased to {patience_}")                

                # Check for early stopping.                     
                if patience_ >= patience:
                    break
            
            # Save the last model for each fold.
            last_model_path = f"{output_dir}/last_model_fold_{fold}.spacy"
            nlp.to_disk(last_model_path)
            logging.info("Last model saved for Fold %d", fold)

             # Load the current best model.
            best_nlp = spacy.load(best_model_path)
            # Evaluate after each fold and append it to result.
            results.append(evaluate(data, best_nlp, test_idx))
        
    # Save results. 
    logging.info("Training complete.")
    with open(save_path_results, "w", encoding="utf-8") as f: 
        json.dump(results, f, indent=2, ensure_ascii=False)


def evaluate(data: json, nlp: spacy.language, test_idx: ndarray) -> dict[str, float]:
    """_summary_

    Args:
        data (json): Data the model should be trained on.
        nlp (spacy.language): Model which should be evaluated. 
        test_idx (ndarray): Indices of the data serving as TEST_DATA

    Returns:
        dict[str, float]: Dict with the results for precision, recall and f1 score. 
    """    
    
    # Create Example-object of this fold's train data.
    TEST_DATA = [Example.from_dict(nlp(text), annotations) for text, annotations in [data[i] for i in test_idx]]

    # Initialze scorer.
    scorer = Scorer()

    # Evaluate the model with the help of the scorer.
    scores = scorer.score(examples=TEST_DATA)

    # Log results
    logging.info("Precision: %f", scores["ents_p"])
    logging.info("Recall: %f", scores["ents_r"])
    logging.info("F1 score: %f", scores["ents_f"]) 

    # Save resulst in dictionary.
    results = {
    "Precision": scores["ents_p"],
    "Recall": scores["ents_r"],
    "F1 score": scores["ents_f"]
    }

    # Return results. 
    return results

def main_blank_specific():
    # Specify the paths and parameters.
    data_path = Path(Path(__file__).parent, "../data/cross_validation/all_data.json")
    output_dir = Path(Path(__file__).parent, "../models/cross_validation/specific_blank")
    save_path_result = Path(Path(__file__).parent, "../results/cross_validation/specific_blank.json")
    n_iter = 1000
    patience = 5
    n_folds = 5

    # Load data.
    data = open_json(data_path)

    cross_validation_blank(data, output_dir, save_path_result, n_iter, patience, n_folds)


def main_blank_general():
    # Specify the paths and parameters.
    data_path = Path(Path(__file__).parent, "../data/cross_validation/all_data_general.json")
    output_dir = Path(Path(__file__).parent, "../models/cross_validation/general_blank")
    save_path_result = Path(Path(__file__).parent, "../results/cross_validation/general_blank.json")
    n_iter = 1000
    patience = 5
    n_folds = 5

    # Load data.
    data = open_json(data_path)

    cross_validation_blank(data, output_dir, save_path_result, n_iter, patience, n_folds)


def main_lm_specific():
    # Specify the paths and parameters.
    data_path = Path(Path(__file__).parent, "../data/cross_validation/all_data.json")
    output_dir = Path(Path(__file__).parent, "../models/cross_validation/specific_lm")
    save_path_result = Path(Path(__file__).parent, "../results/cross_validation/specific_lm.json")
    n_iter = 1000
    patience = 5
    n_folds = 5

    # Load data.
    data = open_json(data_path)

    cross_validation_german_lm(data, output_dir, save_path_result, n_iter, patience, n_folds)


def main_lm_general(): 
    # Specify the paths and parameters.
    data_path = Path(Path(__file__).parent, "../data/cross_validation/all_data_general.json")
    output_dir = Path(Path(__file__).parent, "../models/cross_validation/general_lm")
    save_path_result = Path(Path(__file__).parent, "../results/cross_validation/general_lm.json")
    n_iter = 1000
    patience = 5
    n_folds = 5

    # Load data.
    data = open_json(data_path)

    cross_validation_german_lm(data, output_dir, save_path_result, n_iter, patience, n_folds)

def main_trf_specific(): 
    # Specify the paths and parameters.
    data_path = Path(Path(__file__).parent, "../data/cross_validation/all_data.json")
    output_dir = Path(Path(__file__).parent, "../models/cross_validation/specific_trf")
    save_path_result = Path(Path(__file__).parent, "../results/cross_validation/specific_trf.json")
    n_iter = 1000
    patience = 5
    n_folds = 5

    # Load data.
    data = open_json(data_path)

    cross_validation_transformers(data, output_dir, save_path_result, n_iter, patience, n_folds)

def main_trf_general():
    # Specify the paths and parameters.
    data_path = Path(Path(__file__).parent, "../data/cross_validation/all_data_general.json")
    output_dir = Path(Path(__file__).parent, "../models/cross_validation/general_trf")
    save_path_result = Path(Path(__file__).parent, "../results/cross_validation/general_trf.json")
    n_iter = 1000
    patience = 5
    n_folds = 5

    # Load data.
    data = open_json(data_path)

    # Train the model.
    cross_validation_transformers(data, output_dir, save_path_result, n_iter, patience, n_folds)

if __name__ == "__main__":
    main_trf_general()