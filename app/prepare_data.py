import json 

from pathlib import Path

#data
selina_annotation = Path(Path(__file__).parent, "../data/selina_200_annotation.jsonl")
save_selina_annotation = Path(Path(__file__).parent, "../data/selina_tuple_annotation.json")
sorted_selina_annotation = Path(Path(__file__).parent, "../data/selina_sorted_tuple_annotation.json")

teppi_annotation = Path(Path(__file__).parent, "../data/teppi_200_annotation.jsonl")
save_teppi_annotation = Path(Path(__file__).parent, "../data/teppi_tuple_annotation.json")
sorted_teppi_annotation = Path(Path(__file__).parent, "../data/teppi_sorted_tuple_annotation.json")



def transform_into_spacy_format(filepath: str, savepath: str) -> None:  
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
        formatted_data = {"text": "", "entities": []}
        
        # Save text of document in variable and save it in the mask. 
        text = doc["text"]  
        formatted_data["text"] = text

        # Iterate over "labels" in the document.
        for label in doc["label"]: 
            # Save necessary labels in variable. 
            start = label[0]
            end = label[1]
            label = label[2]

            # Create tuple. Entity "label" needs to be upper case.
            entities = (start, end, label.upper())

            # Save tuple in right place of mask.
            formatted_data["entities"].append(entities)

        # Append newly formatted document in training_data.
        training_data.append(formatted_data)
    
    print(len(training_data))
    
    # Save training_data in a new file. 
    with open(savepath, "w", encoding="utf-8") as f: 
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    return training_data

def sort_dataset(file_path: str, save_path: str) -> None:

    with open(file_path, "r") as f: 
        data = json.load(f)
    
    sorted_data = sorted(data, key=lambda item: item["text"])

    print(len(sorted_data))

    with open(save_path, "w", encoding="utf-8") as f: 
        json.dump(sorted_data, f, indent=2, ensure_ascii=False)
    

# used 06.01.24
transform_into_spacy_format(selina_annotation, save_selina_annotation)
transform_into_spacy_format(teppi_annotation, save_teppi_annotation)
        
sort_dataset(save_selina_annotation, sorted_selina_annotation)
sort_dataset(save_teppi_annotation, sorted_teppi_annotation)

