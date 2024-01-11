import json
from pathlib import Path

# Input and output directories
input_dir = Path(Path("__file__").parent, "../data/doccano/new_annotation_val")
output_dir = Path(Path("__file__").parent, "../data/doccano/new_annotation_val_jsonl")

output_dir.mkdir(parents=True, exist_ok=True)


# Iterate over all JSON files in the input directory
for input_file in input_dir.glob('*.json'):
    # Read the JSON file
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    # Convert to JSONL format (single line)
    jsonl_data = json.dumps(data, ensure_ascii=False)

    # Save to the output directory with the same file name and a ".jsonl" extension
    output_file = output_dir / input_file.name.replace('.json', '.jsonl')
    with open(output_file, 'w', encoding="utf-8") as jsonl_file:
        jsonl_file.write(jsonl_data + '\n')