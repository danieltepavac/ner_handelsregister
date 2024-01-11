import json

from pathlib import Path

eval_path = Path(Path("__file__").parent, "../data/annotated_eval.jsonl")

val_save_path_with_content = Path(Path("__file__").parent, "../data/formatted_annotated_val_with_content.json")

val_save_path = Path(Path("__file__").parent, "../data/formatted_annotated_val.json")

train_path = Path(Path("__file__").parent, "../data/annotated_train.jsonl")

train_save_path_with_content = Path(Path("__file__").parent, "../data/formatted_annotated_train_with_content.json")

train_save_path = Path(Path("__file__").parent, "../data/formatted_annotated_train.json")

concat_save_path = Path(Path("__file__").parent, "../data/annotated_data.json")


def transform_dataset(path_of_dataset: str, save_path: str) -> None: 
    """Transform evaluation dataset into validation dataset in desired format: 
        entities: [{"id": _, "label": _, "start_offset": _, "end_offset": _}, ...]

    Args:
        path_of_dataset (str): Path to file being transformed. 
        save_path (str): Save path where transformed file should be saved in. 
    """    
    
    # Create empty data list and save content of jsonl-file in it. 
    data = []

    with open(path_of_dataset, "r") as f:
        for line in f: 
            data_line = json.loads(line)
            data.append(data_line)

    # Create new empty list where transformed data should be saved in. 
    new_data = []

    # Initialize counter to create an id for each annotated entity. 
    counter = 2000
    
    # Iterate over data. 
    for doc in data:
        print(doc) 
        # Save all information in respective variables. 
        outer_id = doc.get("id")
        text = doc.get("text")
        entities = doc.get("label")
        comments = doc.get("Comments")

        # Creat new empty list where transformed entities are saved in. 
        new_entities = []


        # Iterate over entities. 
        for i in entities: 
            # Save all information in respective variables. 
            inner_id = counter
            label = i[2]
            start_offset = i[0]
            end_offset = i[1]

            # Create temporary dict entry with desired format. 
            temp_dict = {"id": inner_id, "label": label, "start_offset": start_offset, "end_offset": end_offset}

            # Append entry to new_entities. 
            new_entities.append(temp_dict)

            # Increase counter. 
            counter += 1 
        

        # Create temporary overall entry in desired format. 
        new_entry = {"id": outer_id, "text": text, "entities": new_entities, "Comments": comments}

        # Append it to new_data. 
        new_data.append(new_entry)

    # Save it in new json. 
    with open(save_path, "w", encoding="utf-8") as f: 
        json.dump(new_data, f, indent=2, ensure_ascii=False)



def delete_key_from_dict(path_of_dataset: str, save_path: str, key_to_delete: str) -> None: 

    data = []

    with open(path_of_dataset, "r") as f: 
        for line in f: 
            data_line = json.loads(line)
            data.append(data_line)
    
    for doc in data: 

        del doc[key_to_delete]
    
    with open(save_path, "w", encoding="utf-8") as f: 
        json.dump(data, f, indent=2, ensure_ascii=False)


def change_key_name(path_of_dataset: str, save_path: str, key_to_change_name: str, new_key: str) -> None: 

    with open(path_of_dataset, "r") as f: 
        data = json.load(f)

    
    for doc in data: 
        entities = doc.get("entities")
        for i in entities:
            label = i.get("label")
            if label == key_to_change_name: 
                i["label"] = new_key            
    
    with open(save_path, "w", encoding="utf-8") as f: 
        json.dump(data, f, indent=2, ensure_ascii=False)


def remove_label(path_of_dataset: str, save_path: str, label: str) -> None: 

    with open(path_of_dataset, "r") as f: 
        data = json.load(f)

    transformed_dict = []

    for doc in data: 
        id = doc.get("id")
        text = doc.get("text")
        entities = doc.get("entities")

        current_label = []

        for i in entities: 
            if i.get("label") != label: 
                current_label.append(i)

        temp_dict = {"id": id, "text": text, "entities": current_label}

        transformed_dict.append(temp_dict)

    with open(save_path, "w", encoding="utf-8") as f: 
        json.dump(transformed_dict, f, indent=2, ensure_ascii=False)
    
def concat_jsons(path_to_first_json: str, path_to_second_json: str, save_path: str) -> None:

    with open(path_to_first_json, "r") as f: 
        data_1 = json.load(f) 
    
    with open(path_to_second_json, "r") as f: 
        data_2 = json.load(f)
    
    all_data = data_1 + data_2

    with open(save_path, "w", encoding="utf-8") as f: 
        json.dump(all_data, f, indent=2, ensure_ascii=False)

#transform_dataset(eval_path, val_save_path_with_content)
#delete_key_from_dict(train_path, train_save_path_with_content, "relations")
#change_key_name(train_save_path_with_content, train_save_path_with_content, "location_company", "company_location")
#change_key_name(train_save_path_with_content, train_save_path_with_content, "location_name", "name_location")
#change_key_name(train_save_path_with_content, train_save_path_with_content, "location_notary", "notary_location")
#remove_label(train_save_path_with_content, train_save_path, "content")
#remove_label(val_save_path_with_content, val_save_path, "content")
concat_jsons(train_save_path, val_save_path, concat_save_path)
