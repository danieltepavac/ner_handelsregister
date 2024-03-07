import json
import logging
from pathlib import Path

from scipy.stats import wilcoxon
import spacy
from spacy.scorer import Scorer
from spacy.training.example import Example


def configure_logger():
    """ Configuration of a looger. 
    """ 
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


def evaluate_f1_scores(test_path: str, model_dir: str) -> list[float]:
    """Evaluate f1 scores fore each document.

    Args:
        test_path (str): Path of train data.
        model_dir (str): Directory where the model is saved.

    Returns:
        list[float]: List with f1 scores for each documents. 
    """       
    # Open and read test data.
    logging.info("Loading test data...")
    TEST_DATA = open_json(test_path)

    # Load trained model.
    logging.info("Loading model from '%s'...", model_dir)
    nlp = spacy.load(model_dir)

    # Create Example-objects for evaluation.
    TEST_DATA = [Example.from_dict(nlp(text), annotations) for text, annotations in TEST_DATA]
    

    # Initialize scorer.
    scorer = Scorer()

    # Initialize an empty list to store F1 scores for each document.
    f1_scores = []

    # Apply the scorer on each document individually and extract F1 score.
    for example in TEST_DATA:
        score = scorer.score([example])
        f1 = score["ents_f"]
        f1_scores.append(f1)

    # Return the list of f1_scores. 
    return f1_scores

def evaluate_f1_score_trf(test_data: str, model: str) -> list[float]:
    """Evaluate the f1_score. Concrete data is loaded, not paths. 
    Args:
        test_data (str): Test_data to be evaluated on. 
        model (str): Used model.

    Returns:
        list[float]: List of f1 scores. 
    """    

    # Initialize scorer. 
    scorer = Scorer()

    # Initialize empty f1 scores list. 
    f1_scores = []  
    
    # Iterate over the test data. 
    for text, annotations in test_data:
        # Create for each document an example object.
        example = Example.from_dict(model(text), annotations)
        # Evaluate the example object. 
        score = scorer.score([example])
        f1 = score["ents_f"]
        # Append f1 scores of current document to list. 
        f1_scores.append(f1)
    
    # Return list of f1_scores. 
    return f1_scores

def wilcoxon_signed_rank_test(f1_scores_model1, f1_scores_model2, alpha=0.05) -> tuple[float, float, bool]:
    """
    Perform Wilcoxon signed-rank test on two sets of F1 scores.
    
    Parameters:
        f1_scores_model1 (list): List of F1 scores from model 1.
        f1_scores_model2 (list): List of F1 scores from model 2.
        alpha (float): Significance level (default is 0.05).
        
    Returns:
        (float, float, bool): Tuple containing test statistic, p-value, and a boolean indicating
                              whether the null hypothesis is rejected.
    """
    
    # Calculate Wilcoxon statistic and p-value for both models. 
    statistic, p_value = wilcoxon(f1_scores_model1, f1_scores_model2)
    # Estimate the rejection of the null hypothesis. 
    reject_null_hypothesis = p_value < alpha

    # Return each value. 
    return statistic, p_value, reject_null_hypothesis

def main(): 
    
    # Paths for data and models. 
    specific_test_path = Path(Path(__file__).parent, "../data/experiment1_specific_annotation/test_data.json")
    general_test_path = Path(Path(__file__).parent, "../data/experiment2_general_annotation/test_data.json")

    specific_blank_model = Path(Path(__file__).parent, "../models/specific/blank_val")
    specific_finetuned_model = Path(Path(__file__).parent, "../models/specific/lm_val")

    general_blank_model = Path(Path(__file__).parent, "../models/general/blank_val")
    general_finetuned_model = Path(Path(__file__).parent, "../models/general/lm_val")

    # Evaluate and save f1 scores for blank. 
    f1_score_blank_specific = evaluate_f1_scores(specific_test_path, specific_blank_model)
    f1_score_blank_general = evaluate_f1_scores(general_test_path, general_blank_model)

    # Compute wilcoxon for blank.
    statistic_blank, p_value_blank, reject_null_hypothesis_blank = wilcoxon_signed_rank_test(f1_score_blank_specific, f1_score_blank_general)

    # Evaluate and save f1 scores for ft. 
    f1_score_ft_specific = evaluate_f1_scores(specific_test_path, specific_finetuned_model)
    f1_score_ft_general = evaluate_f1_scores(general_test_path, general_finetuned_model)

    # Compute wilcoxon for ft. 
    statistic_ft, p_value_ft, reject_null_hypothesis_ft = wilcoxon_signed_rank_test(f1_score_ft_specific, f1_score_ft_general)

    # Create results dictionaries and save it in json. 
    results_blank= {
        "Wilcoxon statistic": statistic_blank, 
        "p_value": p_value_blank, 
        "reject_null_hypothesis": str(reject_null_hypothesis_blank)
    }
    results_ft = {    
        "Wilcoxon statistic": statistic_ft, 
        "p_value": p_value_ft, 
        "reject_null_hypothesis": str(reject_null_hypothesis_ft)
        }

    with open(Path(Path(__file__).parent, "../result/wilcoxon/blank.json"), "w", encoding="utf-8") as f: 
        json.dump(results_blank, f, indent=2, ensure_ascii=False)
    
    with open(Path(Path(__file__).parent, "../result/wilcoxon/ft.json"), "w", encoding="utf-8") as f: 
        json.dump(results_ft, f, indent=2, ensure_ascii=False)


def main_trf(): 

    # Paths for data. 
    specific_test_path = Path(Path(__file__).parent, "../data/experiment1_specific_annotation/test_data.json")
    general_test_path = Path(Path(__file__).parent, "../data/experiment2_general_annotation/test_data.json")

    # Load TEST_DATA for both annotations. 
    specific_TEST_DATA = open_json(specific_test_path)
    general_TEST_DATA = open_json(general_test_path)

    # Save Transformer models. 
    specific_trf_model = Path(Path(__file__).parent, "../models/specific/transformers")
    general_trf_model = Path(Path(__file__).parent, "../models/general/transformers")

    # Load respective models. 
    nlp1 = spacy.load(specific_trf_model)
    nlp2 = spacy.load(general_trf_model)



    # Split TEST_DATA for Transformer Model because otherwise the computational power is not enough. 
    half_length = len(specific_TEST_DATA) // 2
    first_half = specific_TEST_DATA[:half_length]
    second_half = specific_TEST_DATA[half_length:]

    # Evaluate and save f1 scores. 
    f1_score_trf_specific_1 = evaluate_f1_score_trf(first_half, nlp1)
    f1_score_trf_specific_2 = evaluate_f1_score_trf(second_half, nlp1)

    # Again, split TEST_DATA for Transformer Model. 
    half_length_g = len(general_TEST_DATA) // 2
    first_half_g = general_TEST_DATA[:half_length_g]
    second_half_g = general_TEST_DATA[half_length_g:]

    # Evaluate and save f1 scores. 
    f1_score_trf_general_1 = evaluate_f1_score_trf(first_half_g, nlp2)
    f1_score_trf_general_2 = evaluate_f1_score_trf(second_half_g, nlp2)

    # Conncatenated both parts again. 
    f1_score_trf_specific = f1_score_trf_specific_1 + f1_score_trf_specific_2
    f1_score_trf_general = f1_score_trf_general_1 + f1_score_trf_general_2

    # Compute wilcoxon. 
    statistic, p_value, reject_null_hypothesis = wilcoxon_signed_rank_test(f1_score_trf_specific, f1_score_trf_general)

    # Create result dictionary and save it in Json. 
    results = {
        "Wilcoxon statistic": statistic, 
        "p_value": p_value,
        "reject_null_hypothesis": str(reject_null_hypothesis)
    }

    with open(Path(Path(__file__).parent, "../result/wilcoxon/trf.json"), "w", encoding="utf-8") as f: 
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main_trf()  


