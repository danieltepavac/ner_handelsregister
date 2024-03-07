from pathlib import Path
import json

def open_json(path: str) -> dict:
    """Open and read a json file.

    Args:
        path (str): Path to json file.

    Returns:
        dict: Read in json file.
    """ 
    with open(path, "r") as f:
        data = json.load(f)
    return data

def average_f1_each_entity(result1_path: json, result2_path: json, result3_path: json, num_dict: int, save_path: str): 
    """Compute the average f1 score for each available entity. 

    Args:
        result1_path (json): Results of first model.
        result2_path (json): Results of second model.
        result3_path (json): Results of third model.
        num_dict (int): Number of result dictionaries.
        save_path (str): Path where averages should be saved. 
    """   
    # Initialize empty dict.  
    sum_dict = {}

    # Open each result file. 
    result1 = open_json(result1_path)
    result2 = open_json(result2_path)
    result3 = open_json(result3_path) 

    # Iterate over the results. 
    for dict in[result1, result2, result3]: 
        # Iterate over each dict. 
        for key, value in dict.get("Entities per type", {}).items():
            # If entity is not in sum_dict, append it.  
            if key not in sum_dict: 
                sum_dict[key] = value["f"]
            # Otherwise sum it up. 
            else: 
                sum_dict[key] += value["f"]
    
    # Create a dict with each average f1 entity score, respectively. 
    avg_dict = {key: value/ num_dict for key, value in sum_dict.items()}

    with open(save_path, "w", encoding="utf-8") as f: 
        json.dump(avg_dict, f, indent=2, ensure_ascii=False)


def main_specific(): 
    specific_blank_path = Path(Path(__file__).parent, "../results/exp1/specific_blank.json")
    specific_ft_path = Path(Path(__file__).parent, "../results/exp1/specific_ft.json")
    specific_trf_path = Path(Path(__file__).parent, "../results/exp1/specific_trf.json") 

    save_path = Path(Path(__file__).parent, "../results/exp1/average_f1_scores_over_entities.json")
    average_f1_each_entity(specific_blank_path, specific_ft_path, specific_trf_path, 3, save_path)

def main_general(): 
    general_blank_path = Path(Path(__file__).parent, "../results/exp2/general_blank.json")
    general_ft_path = Path(Path(__file__).parent, "../results/exp2/general_ft.json")
    general_trf_path = Path(Path(__file__).parent, "../results/exp2/general_trf.json")

    save_path = Path(Path(__file__).parent, "../results/exp2/average_f1_scores_over_entities.json")
    average_f1_each_entity(general_blank_path, general_ft_path, general_trf_path, 3, save_path)

if __name__ == "__main__": 
    main_general()