from pathlib import Path
import json

train_path = Path(Path(__file__).parent, "../data/experiment1_detailed_annotation/train_data.json")
train_save_path = Path(Path(__file__).parent, "../data/experiment2_broader_annotation/train_data.json")

test_path = Path(Path(__file__).parent, "../data/experiment1_detailed_annotation/test_data.json")
test_save_path = Path(Path(__file__).parent, "../data/experiment2_broader_annotation/test_data.json")

eval_path = Path(Path(__file__).parent, "../data/experiment1_detailed_annotation/eval_data.json")
eval_save_path = Path(Path(__file__).parent, "../data/experiment2_broader_annotation/eval_data.json")

def change_labels(path_file: str, save_path: str) -> None: 
    """ Change labels to a broader approach.

    Args:
        path_file (str): File path to data.
        save_path (str): Save path where transformed data should be saved in. 
    """    

    # Open data.
    with open(path_file, "r") as f: 
        DATA = json.load(f)
    
    # Iterate over data. 
    for doc in DATA: 
        
        # Iterate over entities.
        for _, entities in doc[1].items():
            
            # Get each entity and change respective label into a broader category. 
            for entity in entities: 
                if entity[2].endswith("_DATE"):
                    entity[2] = "DATE"
                if entity[2].endswith("_LOCATION"):
                    entity[2] = "LOCATION"
                if entity[2].endswith("_STREET"):
                    entity[2] = "STREET"
                if entity[2].endswith("_POSTAL_CODE"): 
                    entity[2] = "POSTAL_CODE"
    
    # Save transformed data. 
    with open(save_path, "w", encoding="utf-8") as f: 
        json.dump(DATA, f, indent=2, ensure_ascii=False)

change_labels(train_path, train_save_path)
change_labels(test_path, test_save_path)
change_labels(eval_path, eval_save_path)
    