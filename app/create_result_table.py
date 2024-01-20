import pandas as pd

from pathlib import Path

import json

# Define a function to read evaluation results from a JSON file.
def read_evaluation_results(file_path: str) -> json:
    """ Read in jsons-file with the results. 

    Args:
        file_path (str): File path to the json-file. 

    Returns:
        json: Returns json with the content: results. 
    """    

    with open(file_path, "r") as json_file:
        results = json.load(json_file)

    return results


def collecting_results(directory: str) -> pd.DataFrame:
    """ Create DataFrame with the results of each experiment. 

    Args:
        directory (str): Path to directory where the json files are stored in. 

    Returns:
        pd.DataFrame: DataFrame with the results collected.
    """    

    # Save directory path in Variable. 
    results_directory = Path(directory)

    # List all JSON files in the directory.
    json_files = [file for file in results_directory.glob("*.json")]

    # Initialize an empty list to store the results.
    all_results = []

    # Iterate through each json file and read the results.
    for json_file in json_files:
        file_path = str(json_file)
        
        results = read_evaluation_results(file_path)
        # Add a column for the file name. 
        results["File"] = json_file.name
        # Append results to overall list. 
        all_results.append(results)

    # Convert the list of dictionaries into a pandas DataFrame.
    df = pd.DataFrame(all_results)

    return df

# Specify the directory containing your JSON files
results_experiment1_dir = Path(Path(__file__).parent, "../results/experiment1_detailed_annotation")
results_experiment2_dir = Path(Path(__file__).parent, "../results/experiment2_broader_annotation/")

experiment1 = collecting_results(results_experiment1_dir)
experiment2 = collecting_results(results_experiment2_dir)

# Save newly created files as csv-file. 
save_path_experiment1 = Path(Path(__file__).parent, "../results/experiment1_detailed_annotation/all_results.csv")
save_path_experiment2 = Path(Path(__file__).parent, "../results/experiment2_broader_annotation/all_results.csv")

experiment1.to_csv(save_path_experiment1, index=False)
experiment2.to_csv(save_path_experiment2, index=False)