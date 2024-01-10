import json
from pathlib import Path

import spacy
from spacy.scorer import Scorer



sorted_selina_annotation = Path(Path(__file__).parent, "../data/selina_sorted_tuple_annotation.json")
sorted_teppi_annotation = Path(Path(__file__).parent, "../data/teppi_sorted_tuple_annotation.json")

def get_entities(file_path: str) -> list: 

    with open(file_path, "r") as f: 
        data = json.load(f)
    
    all_entities = []
    
    for doc in data: 
        entities = doc.get("entities")
        all_entities.append(entities)
    
    return all_entities

selina_entities = get_entities(sorted_selina_annotation)
teppi_entities = get_entities(sorted_teppi_annotation)

def jaccard_similarity(path1, path2):

    with open(path1, "r") as f: 
        list1 = json.load(f)

    with open(path2, "r") as f: 
        list2 = json.load(f)

    entities_set1 = set()
    entities_set2 = set()

# Extract entities from the first list
    for item in list1:
        if isinstance(item, dict) and "entities" in item:
            entities_set1.update(frozenset(entity) for entity in item["entities"])

    # Extract entities from the second list
    for item in list2:
        if isinstance(item, dict) and "entities" in item:
            entities_set2.update(frozenset(entity) for entity in item["entities"])


    # Calculate Jaccard similarity
    intersection = len(entities_set1.intersection(entities_set2))
    union = len(entities_set1) + len(entities_set2) - intersection

    similarity = intersection / union if union != 0 else 0
    return similarity



similarity = jaccard_similarity(sorted_selina_annotation, sorted_teppi_annotation)
print(f"Jaccard Similarity: {similarity}")






