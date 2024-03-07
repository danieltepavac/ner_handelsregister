import matplotlib.pyplot as plt

from pathlib import Path

import json

import numpy as np

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

def extract_f1_scores(data: json) -> dict[str, float]:
    """ Extract f1 scores of a result file. 

    Args:
        data (json): Data which the f1 scores should be extracted. 

    Returns:
        dict[str, float]: Dictionary with entity as key and f1 score as value. 
    """  
    entity_f1_scores = {}
    # Iterate over all entities and save the f1 scores. 
    for entity, scores in data["Entities per type"].items():
        entity_f1_scores[entity] = scores["f"]
    return entity_f1_scores

def create_examples_in_confusion_matrix_form(labels: list[str], colors: list[str], save_path:str): 
    """Create an example of an confusion matrix with a concrete word for each case

    Args:
        labels (list[str]): Labels serving as example. 
        colors (list[str]): Colors of each corner. 
        save_path (str): Path where plot should be saved. 
    """    
    # Plotting.
    plt.figure(figsize=(6, 6))

    # Add rectangles representing each outcome with the specified colors.
    for label, color, x, y in zip(labels, colors, [0.5, 1.5, 1.5, 0.5], [0.5, 0.5, -0.5, -0.5]):
        plt.fill([x-0.5, x+0.5, x+0.5, x-0.5], [y-0.5, y-0.5, y+0.5, y+0.5], color=color, alpha=0.5)

        # Adding labels to each corner.
        plt.text(x, y, label, ha="center", va="center", fontsize=13, color="black", fontweight="bold")

    # Remove axis.
    plt.axis("off")

    # Ensures tight layout without dividing the image.
    plt.tight_layout()  

    plt.show()

    # Save figure. 
    plt.savefig(save_path)

def bar_entities(result1_path: json, result2_path: json, result3_path: json, save_path: str) -> None: 
    """Create a bar plot for each entity performance in each model. 

    Args:
        result1_path (json): Results of the first model.
        result2_path (json): Results of the second model.
        result3_path (json): Results of the third model.
        save_path (str): Path where plot should be saved. 
    """    
    # Open each result file. 
    result1 = open_json(result1_path)
    result2 = open_json(result2_path)
    result3 = open_json(result3_path)

    # Extract the f1 scores of each entity. 
    f1_score1 = extract_f1_scores(result1)
    f1_score2 = extract_f1_scores(result2)
    f1_score3 = extract_f1_scores(result3)

    # Combine f1 scores for all models into one dictionary.
    all_entities = sorted(set().union(f1_score1.keys(), f1_score2.keys(), f1_score3.keys()))

    # Plotting.
    plt.figure(figsize=(14, 8))  

    # Define bar properties.
    bar_width = 0.25
    opacity = 0.8  

    # Plot bars.
    r1 = np.arange(len(all_entities))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    color_blank = "#FFA500"  # orange
    color_lm = "#8ED1FC"  # blue
    color_trf = "#2E8B57"  # green

    plt.bar(r1, [f1_score1.get(entity, 0) for entity in all_entities], color=color_blank, width=bar_width, edgecolor="grey", label="blank LM model", alpha=opacity)
    plt.bar(r2, [f1_score2.get(entity, 0) for entity in all_entities], color=color_lm, width=bar_width, edgecolor="grey", label="spaCy LM model", alpha=opacity)
    plt.bar(r3, [f1_score3.get(entity, 0) for entity in all_entities], color=color_trf, width=bar_width, edgecolor="grey", label="Transformers LM model", alpha=opacity)

    plt.xlabel("Entity Type", fontweight="bold", fontsize=14)
    plt.ylabel("F1-score", fontweight="bold", fontsize=14)
    plt.title("F1-score for Different Entity Types", fontweight="bold", fontsize=16)
    plt.xticks([r + bar_width for r in range(len(all_entities))], all_entities, rotation=90, ha="center", fontsize=10)
    plt.yticks(fontsize=10)

    # Add grid lines.
    plt.grid(axis="y", linestyle="--")

    # Adjust y-axis limit.
    plt.ylim(0, 1) 

    plt.legend(fontsize=12)

    # Save the plot as a PNG file with higher quality (dpi).
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)  
    plt.show()

def bar_annotation_comparison(data1: json, data2: json, save_path: str):
    
    # Save keys sorted.
    keys = sorted(data1.keys())
    # Save the values.
    values1 = [data1[key] for key in keys]
    values2 = [data2[key] for key in keys]

    # Plotting.
    bar_width = 0.35
    index = range(len(keys))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)  

    bars1 = plt.bar(index, values1, bar_width, label="t set", color="burlywood")  
    bars2 = plt.bar([i + bar_width for i in index], values2, bar_width, label="s set", color="crimson")  

    plt.xlabel("Entities", fontsize=12)  
    plt.ylabel("Frequencies", fontsize=12)  
    plt.title("Comparison of Annotation for set S and set T", fontsize=14)  
    plt.xticks([i + bar_width/2 for i in index], keys, rotation=90, fontsize=10, ha="right")  
    plt.yticks(fontsize=10)  
    plt.grid(axis="y", linestyle="--", alpha=0.7)  

    plt.legend()  

    plt.tight_layout()  
    plt.show()

    plt.savefig(save_path)

def bar_plot_one_data(data: json, save_path: str):
    keys = sorted(data.keys())
    values = [data[key] for key in keys]

    bar_width = 0.35
    index = range(len(keys))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)  

    bars = plt.bar(index, values, bar_width, label="Frequencies", color='burlywood')  

    # Add text labels on top of each bar
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Entities', fontsize=14)  
    plt.ylabel('Frequency', fontsize=14)  
    plt.title('Frequency for each entity after annotation', fontsize=16)  
    plt.xticks(index, keys, rotation=90, fontsize=11, ha='right')  
    plt.yticks(fontsize=10)  
    plt.grid(axis='y', linestyle='--', alpha=0.7)  

    plt.ylim(0, max(values) * 1.1)  

    plt.legend()  

    plt.tight_layout()  
    plt.show()

    plt.savefig(save_path)

def main_bar_specific():
    specific_blank_path = Path(Path(__file__).parent, "../results/exp1/specific_blank.json")
    specific_ft_path = Path(Path(__file__).parent, "../results/exp1/specific_ft.json")
    specific_trf_path = Path(Path(__file__).parent, "../results/exp1/specific_trf.json") 

    save_path = Path(Path(__file__).parent, "../results/figures/bar_specific_entities.png")
    bar_entities(specific_blank_path, specific_ft_path, specific_trf_path, save_path)

def main_bar_general(): 
    general_blank_path = Path(Path(__file__).parent, "../results/exp2/general_blank.json")
    general_ft_path = Path(Path(__file__).parent, "../results/exp2/general_ft.json")
    general_trf_path = Path(Path(__file__).parent, "../results/exp2/general_trf.json")

    save_path = Path(Path(__file__).parent, "../results/figures/bar_general_entities.png")
    bar_entities(general_blank_path, general_ft_path, general_trf_path, save_path)

def main_bar_annotation_comparsion(): 
    path = Path(Path(__file__).parent, "../results/annotation_comparison/entities.json")

    with open(path, "r") as f: 
        data = json.load(f)
    
    t_overall_count = data["t_overall_count"]
    s_overall_count = data["s_overall_count"]

    save_path = Path(Path(__file__).parent, "../results/annotation_comparison/comparison.png")
    bar_annotation_comparison(t_overall_count, s_overall_count, save_path)

def main_bar_annotation_single():
    path = Path(Path(__file__).parent, "../data/cross_validation/all_data.json")

    with open(path, "r") as f: 
        data = json.load(f)
    
    t_overall_count = data["t_overall_count"]

    save_path = Path(Path(__file__).parent, "../results/annotation_comparison/comparison_single.png")
    bar_plot_one_data(t_overall_count, save_path)


def main_ex_confusion_matrix(): 

    # Define the labels for each corner.
    labels = ["True Positive:\nHannover", "True Negative:\nBundespersonalausweis", "False Positive:\nAmtsgericht Hildesheim", "False Negative:\nHRB 22640"]

    # Define the colors for each corner.
    colors = ["green", "green", "red", "red"]

    create_examples_in_confusion_matrix_form(labels, colors, Path(Path(__file__).parent, "../results/figures/example_confusion_matrix.png"))


if __name__ == "__main__": 
    main_bar_annotation_single()




"""
bar_plot(t_overall_count, s_overall_count)

"""