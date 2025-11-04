import os
import json

# Input and output directory
input_folder = 'annotation/saves'  # Replace with your input folder path
output_folder = 'annotation/saves_txt'  # Replace with your output folder path

"""
Convert annotation data files from json format used in Segment Anything Annotator Master to txt used for YOLO

Segment Anything Annotator Master:
https://github.com/haochenheheda/segment-anything-annotator
"""
 

# Function to convert annotations
def convert_annotations(json_file):
    # Read JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Get base file name without extension
    base_filename = os.path.basename(json_file).split('.')[0]
    
    # Prepare output file path
    output_file = os.path.join(output_folder, base_filename + '.txt')

    # Open the output file to write
    with open(output_file, 'w') as out_file:
        for shape in data['shapes']:
            # Extract points and format them
            points = shape['points']
            # Flatten the points into the desired format and prepend the class index (0)
            points_str = ' '.join([f"{x/640.0} {y/480.0}" for x, y in points])
            out_file.write(f"0 {points_str}\n")

# Process all JSON files in the input folder
def process_json_folder(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            json_file_path = os.path.join(input_folder, filename)
            convert_annotations(json_file_path)

# Run the script
process_json_folder(input_folder)
