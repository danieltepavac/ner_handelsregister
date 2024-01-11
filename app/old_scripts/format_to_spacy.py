import json 

from pathlib import Path

def transform_into_spacy_format_with_entity(filepath: str, savepath: str) -> json:  
    """ Transforms annotated data into spacy format, so it can be used as training data. 
        The annotated set has the annotated information saved in key entities.  

    Args:
        filepath (str): Path of data being transformed into spacy format. 
        savepath (str): Savepath of transformed data.

    Returns:
        json: Json with correct spaCy format. 
    """    
     
    # Create empty list where annotated data will be saved in. 
    annotated_data = []

    # Open file of annotated data and save lines in annotated_data.
    with open(filepath, "r") as f: 
        for line in f: 
            data = json.loads(line)
            annotated_data.append(data)

    # Create empty list for formatted training_data.
    training_data = []

    # Iterate over annotated_data.
    for doc in annotated_data:
        # First, create a mask of the wished for format. 
        format = {"text": "", "entities": []}
        
        # Save text of document in variable and save it in the mask. 
        text = doc["text"]  
        format["text"] = text

        # Iterate over "entities" in the document.
        for entity in doc["entities"]: 
            # Save necessary entities in variable. 
            start = entity["start_offset"]
            end = entity["end_offset"]
            label = entity["label"]

            # Create tuple. Entity "label" needs to be upper case.
            entities = (start, end, label.upper())

            # Save tuple in right place of mask.
            format["entities"].append(entities)

            # Append newly formatted document in training_data.
            training_data.append(format)
    
    # Save training_data in a new file. 
    with open(savepath, "w", encoding="utf-8") as f: 
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    return training_data

def transform_into_spacy_format_with_label(filepath: str, savepath: str) -> json:  
    """ Transforms annotated data into spacy format, so it can be used as training data. 
        The annotated set has the annotated information saved in key labels.  

    Args:
        filepath (str): Path of data being transformed into spacy format. 
        savepath (str): Savepath of transformed data.

    Returns:
        json: Json with correct spaCy format. 
    """    
     
    # Create empty list where annotated data will be saved in. 
    annotated_data = []

    # Open file of annotated data and save lines in annotated_data.
    with open(filepath, "r") as f: 
        for line in f: 
            data = json.loads(line)
            annotated_data.append(data)

    # Create empty list for formatted training_data.
    training_data = []

    # Iterate over annotated_data.
    for doc in annotated_data:
        # First, create a mask of the wished for format. 
        format = {"text": "", "entities": []}
        
        # Save text of document in variable and save it in the mask. 
        text = doc["text"]  
        format["text"] = text

        # Iterate over "labels" in the document.
        for label in doc["label"]: 
            # Save necessary labels in variable. 
            start = label[0]
            end = label[1]
            label = label[2]

            # Create tuple. Entity "label" needs to be upper case.
            entities = (start, end, label.upper())

            # Save tuple in right place of mask.
            format["entities"].append(entities)

            # Append newly formatted document in training_data.
            training_data.append(format)
    
    # Save training_data in a new file. 
    with open(savepath, "w", encoding="utf-8") as f: 
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    return training_data

def transform_jsonl_into_spacy_format_as_tuples(filepath: str, savepath: str) -> json:  
    """ Transforms annotated data into spacy format, so it can be used as training data. 
        The annotated set has the annotated information saved in key entities.
        It should be saved as list of tuples.   

    Args:
        filepath (str): Path of data being transformed into spacy format. 
        savepath (str): Savepath of transformed data.

    Returns:
        json: Json with correct spaCy format. 
    """    
     
    # Create empty list where annotated data will be saved in. 
    annotated_data = []

    # Open file of annotated data and save lines in annotated_data.
    with open(filepath, "r") as f:  
        for line in f: 
            data = json.loads(line)
            annotated_data.append(data)

    # Create empty list for formatted training_data.
    training_data = []

    # Iterate over annotated_data.
    for doc in annotated_data:
        # Save text of document in variable and create empty_list. 
        text = doc["text"]
        entity_list = []

        # Iterate over "entities" in the document.
        for entity in doc.get("entities", []):
            # Save necessary entities in variables.
            start = entity.get("start_offset")
            end = entity.get("end_offset")
            label = entity.get("label")

            # Check if all required fields are present.
            if all((start, end, label)):
                # Create a tuple with the entity label in uppercase.
                entity_tuple = (start, end, label.upper())

                # Save the tuple in the "entities" list of the format dictionary.
                entity_list.append(entity_tuple)

        # Create tuple format with necessary information. 
        format = (text, {"entities": entity_list})  

        # Append newly formatted document in training_data.
        training_data.append(format)
  
    # Save training_data in a new file. 
    with open(savepath, "w", encoding="utf-8") as f: 
        json.dump(training_data, f, indent=2, ensure_ascii=False)


def transform_into_spacy_format_as_tuples(filepath: str, savepath: str) -> json:  
    """ Transforms annotated data into spacy format, so it can be used as training data. 
        The annotated set has the annotated information saved in key entities.
        It should be saved as list of tuples.   

    Args:
        filepath (str): Path of data being transformed into spacy format. 
        savepath (str): Savepath of transformed data.

    Returns:
        json: Json with correct spaCy format. 
    """    
     
    # Open file of annotated data and save lines in annotated_data.
    with open(filepath, "r") as f:  
        annotated_data = json.load(f)

    # Create empty list for formatted training_data.
    training_data = []

    # Iterate over annotated_data.
    for doc in annotated_data:
        # Save text of document in variable and create empty_list. 
        text = doc.get("text")
        entities = doc.get("entities")
        entity_list = []

        # Iterate over "entities" in the document.
        for entity in entities:
            # Save necessary entities in variables.
            start = entity.get("start_offset")
            end = entity.get("end_offset")
            label = entity.get("label")
            
            # Check if all required fields are present.
            if all((start, end, label)):
                # Create a tuple with the entity label in uppercase.
                entity_tuple = (start, end, label.upper())

                # Save the tuple in the "entities" list of the format dictionary.
                entity_list.append(entity_tuple)

        # Create tuple format with necessary information. 
        format = (text, {"entities": entity_list})  

        # Append newly formatted document in training_data.
        training_data.append(format)
  
    # Save training_data in a new file. 
    with open(savepath, "w", encoding="utf-8") as f: 
        json.dump(training_data, f, indent=2, ensure_ascii=False)


path_to_train = Path(Path(__file__).parent, "../data/train_data.json")
path_to_val = Path(Path(__file__).parent, "../data/val_data.json")
path_to_test = Path(Path(__file__).parent, "../data/test_data.json")

save_path_train = Path(Path(__file__).parent, "../data/tuple_train_data.json")
save_path_val = Path(Path(__file__).parent, "../data/tuple_val_data.json")
save_path_test = Path(Path(__file__).parent, "../data/tuple_test_data.json")

# annotation from 06.01.2023
selina_annotation = Path(Path(__file__).parent, "../data/selina_200_annotation.jsonl")

save_selina_annotation = Path(Path(__file__).parent, "../data/selina_tuple_annotation.json")

transform_jsonl_into_spacy_format_as_tuples(selina_annotation, save_selina_annotation)

#transform_into_spacy_format_as_tuples(path_to_train, save_path_train)
#transform_into_spacy_format_as_tuples(path_to_val, save_path_val)
#transform_into_spacy_format_as_tuples(path_to_test, save_path_test)