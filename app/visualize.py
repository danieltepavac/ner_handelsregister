import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd


import json

# detailed.
path_detailed_with_eval = Path(Path(__file__).parent, "../results/experiment1_detailed_annotation/2_with_eval_ents.json")
path_detailed_finetuned = Path(Path(__file__).parent, "../results/experiment1_detailed_annotation/2_finetuned_ents.json")
path_detailed_transformers = Path(Path(__file__).parent, "../results/experiment1_detailed_annotation/transformers_ents.json")

# broader.
path_broader_with_eval = Path(Path(__file__).parent, "../results/experiment2_broader_annotation/2_with_eval_ents.json")
path_broader_finetuned = Path(Path(__file__).parent, "../results/experiment2_broader_annotation/2_finetuned_ents.json")
path_broader_transformers = Path(Path(__file__).parent, "../results/experiment2_broader_annotation/transformers_ents.json")

# detailed save.
save_path_detailed_with_eval = Path(Path(__file__).parent, "../results/graphs/experiment1_detailed_annotation/2_with_eval_ents")
save_path_detailed_finetuned = Path(Path(__file__).parent, "../results/graphs/experiment1_detailed_annotation/2_finetuned")
save_path_detailed_transformers = Path(Path(__file__).parent, "../results/graphs/experiment1_detailed_annotation/transformers")

# broader.
save_path_broader_with_eval = Path(Path(__file__).parent, "../results/graphs/experiment2_broader_annotation/2_with_eval_ents")
save_path_broader_finetuned = Path(Path(__file__).parent, "../results/graphs/experiment2_broader_annotation/2_finetuned")
save_path_broader_transformers = Path(Path(__file__).parent, "../results/graphs/experiment2_broader_annotation/transformers")




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

def open_data(path: str) -> json: 

    with open(path, "r") as f: 
        data = json.load(f)

    return data



def create_bar_graph_entities(data: json, save_path: str) -> None: 

    for entity, scores in data.items():
        # Replace "/" with "_"
        entity_filename = entity.replace("/", "_")

        fig, ax = plt.subplots(figsize=(10,5))

        bar_width = 0.2
        index = range(3)
        labels = ["Precision", "Recall", "F1 Score"]

        
        scores_list = [scores["p"], scores["r"], scores["f"]]
        colors = ["blue", "green", "orange"]

        bars = ax.bar(index, scores_list, bar_width, color=colors, label=entity)

        for i, v in enumerate(scores_list): 
            ax.text(i, v/2, str(round(v, 3)), ha="center", va="center", color="white")

            ax.set_xlabel("Metrics.")
            ax.set_ylabel("Scores.")
            ax.set_title(f"{entity} - Precision, Recall, and F1 Score.")
            ax.set_xticks([i for i in index])
            ax.set_xticklabels(labels)
        
        save_path.mkdir(parents=True, exist_ok=True)

        plt.savefig(f"{save_path}/{entity_filename}.png")


# created 24.01.2024
"""detailed_with_eval = open_data(path_detailed_with_eval)
create_bar_graph_entities(detailed_with_eval, save_path_detailed_with_eval)

detailed_finetuned = open_data(path_detailed_finetuned)
create_bar_graph_entities(detailed_finetuned, save_path_detailed_finetuned)

detailed_transformers = open_data(path_detailed_transformers)
create_bar_graph_entities(detailed_transformers, save_path_detailed_transformers)

broader_with_eval = open_data(path_broader_with_eval)
create_bar_graph_entities(broader_with_eval, save_path_broader_with_eval)

broader_finetuned = open_data(path_broader_finetuned)
create_bar_graph_entities(broader_finetuned, save_path_broader_finetuned)

broader_transformers = open_data(path_broader_transformers)
create_bar_graph_entities(broader_transformers, save_path_broader_transformers)"""



# Directory containing your data files
exp1_with_eval = Path(Path(__file__).parent, "../results/experiment1_detailed_annotation/2_with_eval_ents.json")
exp1_finetuned = Path(Path(__file__).parent, "../results/experiment1_detailed_annotation/2_finetuned_ents.json")
exp1_transformers = Path(Path(__file__).parent, "../results/experiment1_detailed_annotation/transformers_ents.json")

exp2_with_eval = Path(Path(__file__).parent, "../results/experiment2_broader_annotation/2_with_eval_ents.json")
exp2_finetuned = Path(Path(__file__).parent, "../results/experiment2_broader_annotation/2_finetuned_ents.json")
exp2_transformers = Path(Path(__file__).parent, "../results/experiment2_broader_annotation/transformers_ents.json")

save_path_1_with_eval_fscore = Path(Path(__file__).parent, "../results/graphs/exp1_with_eval_fscore.png")
save_path_1_finetuned_fscore = Path(Path(__file__).parent, "../results/graphs/exp1_finetuned_fscore.png")
save_path_1_transformers_fscore = Path(Path(__file__).parent, "../results/graphs/exp1_transformers_fscore.png")

save_path_2_with_eval_fscore = Path(Path(__file__).parent, "../results/graphs/exp2_with_eval_fscore.png")
save_path_2_finetuned_fscore = Path(Path(__file__).parent, "../results/graphs/exp2_finetuned_fscore.png")
save_path_2_transformers_fscore = Path(Path(__file__).parent, "../results/graphs/exp2_transformers_fscore.png")

def create_bar_graph_over_all_files(dir: str, save_path: str, color: str) -> None: 

    with open(dir, "r") as f: 
        data = json.load(f)
                      
    f1_scores = []
    precision_scores = []
    recall_scores = []
    entity_names = []


    for entity, scores in data.items():
        entity_names.append(entity) 
        f1_scores.append(scores["f"])
        precision_scores.append(scores["p"])
        recall_scores.append(scores["r"])


    # Create a subplot for F1 scores
    fig, ax1 = plt.subplots(figsize=(10, 18))
    bars = ax1.bar(entity_names, f1_scores, color=color)
    ax1.set_xlabel("Entity Names")
    ax1.set_ylabel("F1 Scores", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title("F1 Scores for Each Entity")


    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    for bar, value in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.01, f'{value:.2f}', ha='center', color='black')

    # Save the figure as an image
    plt.tight_layout()
    plt.savefig(save_path)




create_bar_graph_over_all_files(exp1_with_eval, save_path_1_with_eval_fscore, color="blue")
create_bar_graph_over_all_files(exp1_finetuned, save_path_1_finetuned_fscore, color="green")
create_bar_graph_over_all_files(exp1_transformers, save_path_1_transformers_fscore, color="orange")

create_bar_graph_over_all_files(exp2_with_eval, save_path_2_with_eval_fscore, color="blue")
create_bar_graph_over_all_files(exp2_finetuned, save_path_2_finetuned_fscore, color="green")
create_bar_graph_over_all_files(exp2_transformers, save_path_2_transformers_fscore, color="orange")
            

