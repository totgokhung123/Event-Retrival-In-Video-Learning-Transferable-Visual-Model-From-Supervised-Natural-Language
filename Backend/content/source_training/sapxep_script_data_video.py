# Re-running due to environment reset
from pathlib import Path
import re
import json

# Reload the uploaded JSON file
input_path = "Filtered_Frames_Script.json"
with open(input_path, "r") as f:
    video_data = json.load(f)

# Function to extract sortable key
def sort_key(item):
    path = item[0]
    category = "0" if "/Violence/" in path else "1"
    filename = Path(path).stem
    match = re.search(r'(\d+)', filename)
    number = int(match.group(1)) if match else 0
    return (category, number)

# Sort and reformat
sorted_video_data = dict(sorted(video_data.items(), key=sort_key))

# Save to a new JSON file
output_path = "mapping_captions_frame_sorted.json"
with open(output_path, "w") as f:
    json.dump(sorted_video_data, f, indent=2)

output_path


########################### sắp xếp lại frame theo video



# # Re-import necessary modules after environment reset
# from pathlib import Path
# import re
# import json

# # Reload the uploaded JSON file
# script_path = "/mnt/data/Filtered_Frames_Script.json"
# with open(script_path, "r", encoding="utf-8") as f:
#     frame_data = json.load(f)

# # Define the sort key function using video number
# def extract_video_number(item):
#     path = item[0]
#     match = re.search(r'\\V_(\d+)', path)
#     number = int(match.group(1)) if match else float('inf')
#     return number

# # Sort based on extracted video number
# sorted_frame_data = dict(sorted(frame_data.items(), key=extract_video_number))

# # Save the sorted output
# sorted_script_path = "/mnt/data/Filtered_Frames_Script_sorted.json"
# with open(sorted_script_path, "w", encoding="utf-8") as f:
#     json.dump(sorted_frame_data, f, indent=2, ensure_ascii=False)

# sorted_script_path
