import json

from tqdm import tqdm

from pathlib import Path


def read_ndjson(file_path: str) -> dict:
    """Reads ndjson file and returns dictionary with required information.

    Args:
        file_path (str): File path to ndjson file. 

    Returns:
        dict: Dictionary with file name as key and description as value. 
    """

    # Create empty dictionary. 
    data = {}

    # Open ndjson file.
    with open(file_path, "r") as file:
        
        # Read in all lines. Mainly done for progress bar.
        lines = file.readlines()
        
        # Iterate through lines. Progress bar is here implemented. 
        for line in tqdm(lines, total=len(lines), desc="Processing lines"): 
            
            # Load every line.
            json_data = json.loads(line)

            # Save file name in variable. Is the key of dictionary.
            file_name = json_data.get("file_name")
            
            # Save description as value in dictionary. Test hierarchically, if all values exist to avoid errors. 
            content = json_data.get("content", [])
            if content:
                responses = content[0].get("responses", [])
                if responses:
                    text_annotations = responses[0].get("textAnnotations", [])
                    if text_annotations:
                        description = text_annotations[0].get("description")
                        if description:
                            data[file_name] = description
    
    # Return the dictionary. 
    return data

path_to_file = Path(Path(__file__).parent, "../data/1000_sample.ndjson")

save_path_file = Path(Path(__file__).parent, "../data/1000_read_sample.json")

data = read_ndjson(path_to_file)

cleaned_dict = {key: value.replace('\n', '') for key, value in data.items()}

with open(save_path_file, "w", encoding="utf-8") as f: 
    json.dump(cleaned_dict, f, indent=2, ensure_ascii=False)




