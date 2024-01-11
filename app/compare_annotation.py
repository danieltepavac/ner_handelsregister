import json
from pathlib import Path


s_sorted_annotation = Path(Path(__file__).parent, "../data/selina_sorted_tuple_annotation.json")
t_sorted_annotation = Path(Path(__file__).parent, "../data/teppi_sorted_tuple_annotation.json")

#TODO: necessary function? 
def get_entities(file_path: str) -> list:
    """Get entities out of a labeled dataset. 

    Args:
        file_path (str): Path to the annotated file. 

    Returns:
        list: List of all entities. 
    """    
    with open(file_path, "r") as f: 
        data = json.load(f)
    
    all_entities = []
    
    for doc in data: 
        entities = doc.get("entities")
        all_entities.append(entities)
    
    return all_entities

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
        list1 = json.load(f)

    with open(path2, "r") as f: 
        list2 = json.load(f)

    # Create empty sets. 
    entities_set1 = set()
    entities_set2 = set()

    # Extract entities from the first list.
    for item in list1:
        # Check if item is of type "dict" and if "entities" is said "dict".
        if isinstance(item, dict) and "entities" in item:
            # Update the created set with each entity. frozenset is used because the set should be immutable. 
            entities_set1.update(frozenset(entity) for entity in item["entities"])

    # Extract entities from the second list.
    for item in list2:
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


similarity = jaccard_similarity(s_sorted_annotation, t_sorted_annotation)
print(f"Jaccard Similarity: {similarity}")






