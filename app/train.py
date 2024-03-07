# Source: https://blog.futuresmart.ai/building-a-custom-ner-model-with-spacy-a-step-by-step-guide
# Training of own model with just train_data.


import json
import logging

from pathlib import Path

# Necessary packages for training.
import random
from tqdm import tqdm

# Package for training a language model.
import spacy
from spacy.training.example import Example

def configure_logger():
    """ Configuration of a looger. 
    """    

    # Create a logger.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler and set the logging level.
    logging_file = Path(Path(__file__).parent, "../data/training_log.txt")
    file_handler = logging.FileHandler(logging_file)
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

def save_model(output_dir: str, nlp: spacy.language):
    """ Save a model after training.

    Args:
        output_dir (str): Directory where the model should be saved at. 
        nlp (spacy.language): Model to save. 
    """    

    # Save spacy model. Check if "output_dir" exists. If not, then create it. 
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        # Save language model to disk in target output directory. 
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

def train_blank(train_path: str, output_dir: str, n_iter: int=100, early_stopping_iters: int=5):
    """ Train a model based on a blank German Language Model.

    Args:
        train_path (str): Path of train data.
        output_dir (str): Directory where the model should be saved at.
        n_iter (int, optional): Training iteration. Defaults to 100.
        early_stopping_iters (int, optional): Number of iteration after the model should stop training if no improvement. 
                                            Defaults to 5.
    """    

    # Open train data.
    logging.info("Loading training data...")
    TRAIN_DATA = open_json(train_path)

    # Initialize empty model. 
    model = None
    # Initialize variables for early stopping.
    total_iterations = 0
    best_loss = float("inf")
    early_stop_count = 0

    # Load model. Check if it is None, if so, then create an empty German Language Model. If not, then load the model. 
    if model is not None:
        nlp = spacy.load(model)  
        logging.info("Loaded model '%s'", model)
    else:
        nlp = spacy.blank("de")  
        logging.info("Created blank "de" model")

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
        optimizer = nlp.begin_training()
        # Iterate over the amount of iterations already set. "itn" is an individual iteration.
        for itn in range(n_iter):
            total_iterations += 1
            # Shuffle TRAIN_DATA.
            random.shuffle(TRAIN_DATA)
            # Create empty dictionary to save losses in. 
            losses = {}
            # Iterate over TRAIN_DATA with implemented progress bar.
            for text, annotations in tqdm(TRAIN_DATA):
                # Take input text and create a DOC-object. 
                doc = nlp.make_doc(text)
                # Create spacy EXAMPLE-object based on the created DOC-object and annotations. 
                example = Example.from_dict(doc, annotations)
                # Update the ner model and its parameters with the use of the training example.
                loss = nlp.update(
                    # Take a single example object. 
                    [example], 
                    # Introduce a dropout of 0.5. Probability of dropping a neuron in the neural network. 
                    drop=0.5,
                    # Choose as optimizer Stochastic Gradient Descent. 
                    sgd=optimizer,
                    # Accumulate losses. 
                    losses=losses)
            
            # Log losses and total iterations.
            logging.info("Losses: %s, Total Iterations: %s", losses, total_iterations)
            
            # Check if the loss has improved.
            if losses["ner"] < best_loss:
                best_loss = losses["ner"]
                early_stop_count = 0
            # If not, increase counter by 1. 
            else:
                early_stop_count += 1
                logging.info("Early stop count: %s", early_stop_count)

            # Check for early stopping.
            if early_stop_count >= early_stopping_iters:
                logging.info("Early stopping triggered after %s iterations without improvement.", early_stopping_iters)
                break

    # Save model. 
    save_model(output_dir, nlp)
    logging.info("Last model was saved.")

def train_german_lm(train_path: str, output_dir: str, n_iter: int=100, early_stopping_iters: int=5): 

    TRAIN_DATA = open_json(train_path)

    # Initialize empty model. 
    model = None
    total_iterations = 0
    best_loss = float("inf")
    early_stop_count = 0

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
            total_iterations += 1
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
                loss = nlp.update(
                    # Take a single example object. 
                    [example], 
                    # Introduce a dropout of 0.5. Probability of dropping a neuron in the neural network. 
                    drop=0.5,
                    # Choose as optimizer Stochastic Gradient Descent. 
                    sgd=optimizer,
                    # Accumulate losses. 
                    losses=losses)
            
            # Print losses and total iterations
            print(losses, total_iterations)
            
            # Check if the loss has improved
            if losses["ner"] < best_loss:
                best_loss = losses["ner"]
                early_stop_count = 0
            else:
                early_stop_count += 1
                print(early_stop_count)

            # Check for early stopping
            if early_stop_count >= early_stopping_iters:
                print("Early stopping triggered after {} iterations without improvement.".format(early_stopping_iters))
                break
    
    save_model(output_dir, nlp)

def train_blank_val(train_path: str, val_path: str, output_dir: str, n_iter: int=100, early_stopping_rounds: int = 5): 
    """ Train a model based on a blank German Language Model.

    Args:
        train_path (str): Path of train data.
        val_path (str): Path of train data. 
        output_dir (str): Directory where the model should be saved at.
        n_iter (int, optional): Training iteration. Defaults to 100.
        early_stopping_iters (int, optional): Number of iteration after the model should stop training if no improvement. 
                                            Defaults to 5.
    """    

    # Load training and validation data.
    logging.info("Loading training and validation data...")
    TRAIN_DATA = open_json(train_path)
    VAL_DATA = open_json(val_path)

    # Initialize empty model.
    model = None
    # Initialize variables for early stopping. 
    best_val_loss = 0
    rounds_without_improvement = 0  

    # Load model. Check if it is None, if so, then create an empty German Language Model. If not, then load the model.
    if model is not None:
        nlp = spacy.load(model)
        logging.info("Loaded model '%s'", model)
    else:
        nlp = spacy.blank("de")
        logging.info("Created blank 'de' model")

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
        optimizer = nlp.begin_training()
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

            # Log information about current iteration.
            logging.info("Iteration %s: Training Loss: %s, Validation Loss: %s", itn + 1, losses["ner"], avg_val_loss)

            # Check if the loss has improved. 
            if avg_val_loss > best_val_loss:
                best_val_loss = avg_val_loss
                rounds_without_improvement = 0
            # If not, increase counter by 1.
            else:
                rounds_without_improvement += 1
                logging.info("%s rounds without improvement", rounds_without_improvement)
            
            # Check for early stopping.
            if rounds_without_improvement >= early_stopping_rounds:
                logging.info("Validation loss hasn't improved for %s rounds. Stopping training.", early_stopping_rounds)
                break  
    
    # Save model.
    save_model(output_dir, nlp)
    logging.info("Model was saved.")

def train_german_lm_val(train_path: str, val_path: str, output_dir: str, n_iter: int=100, early_stopping_rounds: int = 5): 
    """Train a model based on the German Language Model de_core_news_lg.

    Args:
        train_path (str): Path of train data.
        val_path (str): Path of train data. 
        output_dir (str): Directory where the model should be saved at.
        n_iter (int, optional): Training iteration. Defaults to 100.
        early_stopping_iters (int, optional): Number of iteration after the model should stop training if no improvement. 
                                            Defaults to 5.
    """    

    # Load training and validation data. 
    logging.info("Loading training and validation data...")
    TRAIN_DATA = open_json(train_path)
    VAL_DATA = open_json(val_path)

    # Initialize empty model. 
    model = None
    # Initialize variables for early stopping.
    best_val_loss = 0
    rounds_without_improvement = 0 

    # Load model. Check if it is None, if so, then load the German Language Model. If not, then load the model. 
    if model is not None:
        nlp = spacy.load(model)  
        logging.info("Loaded model '%s'", model)
    else:
        nlp = spacy.load("de_core_news_lg") 
        logging.info("Loaded 'de_core_news_lg' model")
    
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

            # Log information about current iteration.
            logging.info("Iteration %s: Training Loss: %s, Validation Loss: %s", itn + 1, losses["ner"], avg_val_loss)

            # Check if the loss has improved.
            if avg_val_loss > best_val_loss:
                best_val_loss = avg_val_loss
                rounds_without_improvement = 0
            # If not, increase counter by 1.
            else:
                rounds_without_improvement += 1
                logging.info("%s rounds without improvement", rounds_without_improvement)
            
            # Check for early stopping.
            if rounds_without_improvement >= early_stopping_rounds:
                logging.info("Validation loss hasn't improved for %s rounds. Stopping training.", early_stopping_rounds)
                break  

    # Save model.             
    save_model(output_dir, nlp)
    logging.info("Model was saved.")

def train_transformer(train_path: str, val_path: str, output_dir: str,  n_iter: int=100, early_stopping_rounds: int=5):
    """Train a model based on the Transformers Language Model bert-base-german-cased.

    Args:
        train_path (str): Path of train data.
        val_path (str): Path of train data. 
        output_dir (str): Directory where the model should be saved at.
        n_iter (int, optional): Training iteration. Defaults to 100.
        early_stopping_iters (int, optional): Number of iteration after the model should stop training if no improvement. 
                                            Defaults to 5.
    """ 

    # Load training and validation data.
    logging.info("Loading training and validation data...")
    TRAIN_DATA = open_json(train_path)
    VAL_DATA = open_json(val_path)

    # Initialize variables for early stopping.
    best_val_loss = 0
    rounds_without_improvement = 0  
    
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
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Assure that only ner is present as pipeline by saving them in "other_pipes"..
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    # ..and disabling them.
    with nlp.disable_pipes(*other_pipes):
        # Start training process and saving the result in an optimizer. 
        optimizer = nlp.begin_training()
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
                example = Example.from_dict(doc, annotations)
                # Update the ner model and its parameters with the use of the training example.
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

            # Log information about current iteration.
            logging.info("Iteration %s: Training Loss: %s, Validation Loss: %s", itn + 1, losses["ner"], avg_val_loss)

            # Check if the loss has improved.
            if avg_val_loss > best_val_loss:
                best_val_loss = avg_val_loss
                rounds_without_improvement = 0
            # If not, increase counter by 1.
            else:
                rounds_without_improvement += 1
                logging.info("%s rounds without improvement", rounds_without_improvement)

            # Check for early stopping.
            if rounds_without_improvement >= early_stopping_rounds:
                logging.info("Validation loss hasn't improved for %s rounds. Stopping training.", early_stopping_rounds)
                break 

    
    # Save model.             
    save_model(output_dir, nlp)
    logging.info("Model was saved.")


specific_train_path = Path(Path(__file__).parent, "../data/experiment1_specific_annotation/train_data.json")
specific_val_path = Path(Path(__file__).parent, "../data/experiment1_specific_annotation/val_data.json")

general_train_path = Path(Path(__file__).parent, "../data/experiment2_general_annotation/train_data.json")
general_val_path = Path(Path(__file__).parent, "../data/experiment2_general_annotation/val_data.json")

def main_blank_specific(): 
    save_path = Path(Path(__file__).parent, "../models/specific/blank_val")

    train_blank_val(specific_train_path, specific_val_path, save_path)

def main_blank_general(): 
    save_path = Path(Path(__file__).parent, "../models/general/blank_val")

    train_blank_val(general_train_path, general_val_path, save_path)

def main_lm_specific():
    save_path = Path(Path(__file__).parent, "../models/specific/lm_val")

    train_german_lm_val(specific_train_path, specific_val_path, save_path)

def main_lm_general():
    save_path = Path(Path(__file__).parent, "../models/general/lm_val")

    train_blank_val(general_train_path, general_val_path, save_path)

def main_trf_specific(): 
    save_path = Path(Path(__file__).parent, "../models/specific/trf")

    train_transformer(specific_train_path, specific_val_path, save_path)

def main_trf_general(): 
    save_path = Path(Path(__file__).parent, "../models/general/trf")

    train_transformer(general_train_path, general_val_path, save_path)

