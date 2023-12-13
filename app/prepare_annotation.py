from pathlib import Path
import json 

path = Path(Path("__file__").parent, "../data/new_data_to_annotate.jsonl")

with open(path, "r") as f: 
    data = json.load(f)

print(len(data))