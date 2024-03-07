import json
from pathlib import Path

import matplotlib.pyplot as plt


def jaccard_similarity(path1: str, path2: str) -> float:
    """ Compute Jaccard Similarity for two annoated datasets to see how similar annotated they are.
        Jaccard Similarity: measure of similarity for two datasets. 

    Args:
        path1 (str): Path to first annotated dataset.
        path2 (str): Path to second annotated dataset. 

    Returns:
        float: Jaccard Similarity. 
    """    
    # Open both datasets. 
    with open(path1, "r") as f: 
        data1 = json.load(f)

    with open(path2, "r") as f: 
        data2 = json.load(f)

    # Create empty sets. 
    entities_set1 = set()
    entities_set2 = set()

    # Extract entities from the first list.
    for item in data1:
        # Check if item is of type "dict" and if "entities" is said "dict".
        if isinstance(item, dict) and "entities" in item:
            # Update the created set with each entity. frozenset is used because the set should be immutable. 
            entities_set1.update(frozenset(entity) for entity in item["entities"])

    # Extract entities from the second list.
    for item in data2:
        # Check if item is of type "dict" and if "entities" is said "dict".
        if isinstance(item, dict) and "entities" in item:
            # Update the created set with each entity. frozenset is used because the set should be immutable.
            entities_set2.update(frozenset(entity) for entity in item["entities"])


    # Calculate Jaccard similarity.
    # Intersection: returns a set containing the common elements between the two sets.
    intersection = len(entities_set1.intersection(entities_set2))
    # Union: size of the union between both sets. 
    union = len(entities_set1) + len(entities_set2) - intersection

    # Jaccard similarity is calculated as the size of the intersection divided by the size of the union.
    similarity = intersection / union if union != 0 else 0
    return similarity


def count_individual_entities(path: str) -> dict[str, int]: 
    """Count the occurances of each respective entity.

    Args:
        path (str): Path to the file of which entities should be counted.

    Returns:
        dict [str, int]: Dictionary with entity as key and the number of occurances as value.  
    """    
    # Open data. 
    with open(path, "r") as f: 
        data = json.load(f)
    
    # Create empty dict
    overall_entities_count = {}

    # Iterate over each document. 
    for doc in data: 
        # Create empty temp dict. 
        entity_count = {}

        # Take the entities..
        entities = doc.get("entities")

        # .. and iterate over them. 
        for ent in entities: 
            # Count each instance of each entity. 
            for i in ent: 
                if isinstance(i, str): 
                    if i in entity_count: 
                        entity_count[i] += 1
                    else: 
                        entity_count[i] = 1

        # Append each entity count to the overall dict. 
        for entity, count in entity_count.items(): 
            overall_entities_count[entity] = overall_entities_count.get(entity, 0) + count

    # Return the overall entity count. 
    return overall_entities_count

def main(): 

    s_sorted_annotation = Path(Path(__file__).parent, "../data/annotation_comparison/s_sorted_tuple_annotation.json")
    t_sorted_annotation = Path(Path(__file__).parent, "../data/annotation_comparison/t_sorted_tuple_annotation.json")

    similarity = jaccard_similarity(s_sorted_annotation, t_sorted_annotation)

    t_overall_count = count_individual_entities(t_sorted_annotation)
    s_overall_count = count_individual_entities(s_sorted_annotation)

    result = {
        "t_overall_count": t_overall_count,
        "s_overall_count": s_overall_count,
        "jaccard_similarity": similarity
    }

    with open(Path(Path(__file__).parent, "../results/annotation_comparison/entities.json"), "w", encoding="utf-8") as f: 
        json.dump(result, f, indent=2, ensure_ascii=False)

def main_single():
    path = Path(Path(__file__).parent, "../data/1000_annotation.json")

    overall_count = count_individual_entities(path)

    with open(Path(Path(__file__).parent, "../results/annotation_comparison/entities_single.json"), "w", encoding="utf-8") as f:
        json.dump(overall_count, f, indent=2, ensure_ascii=False) 



if __name__ == "__main__": 
    main_single()





