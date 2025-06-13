import os
import random
import shutil
import csv
from pathlib import Path

# Define source and output directories
sensitive_dir = r"E:\Đồ án chuyên ngành\dataset\NSFW\out\test\NSFW"
violence_dir = r"E:\Đồ án chuyên ngành\dataset\testseg\datasosanh\datatest_PhanLoai\Violence-Image-Dataset\violence_images"
nonviolence_dir = r"E:\Đồ án chuyên ngành\dataset\dataNSFW_neutral\out\test\NonViolence"
output_dir = r"E:\Đồ án chuyên ngành\dataset\datatest_classification"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of image files from both directories
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files

# Get image files from all three categories
sensitive_images = get_image_files(sensitive_dir)
violence_images = get_image_files(violence_dir)
nonviolence_images = get_image_files(nonviolence_dir)

# Check if we have enough images
if len(sensitive_images) < 500:
    print(f"Warning: Only {len(sensitive_images)} Sensitive content images found.")
if len(violence_images) < 500:
    print(f"Warning: Only {len(violence_images)} Violence images found.")
if len(nonviolence_images) < 500:
    print(f"Warning: Only {len(nonviolence_images)} NonViolence images found.")

# Select 500 random images from each category
selected_sensitive = random.sample(sensitive_images, min(500, len(sensitive_images)))
selected_violence = random.sample(violence_images, min(500, len(violence_images)))
selected_nonviolence = random.sample(nonviolence_images, min(500, len(nonviolence_images)))

# Prepare mapping data
mapping_data = []

# Copy and rename sensitive images
for i, img_path in enumerate(selected_sensitive):
    ext = os.path.splitext(img_path)[1]
    new_filename = f"sensitive_{i+1:03d}{ext}"
    new_path = os.path.join(output_dir, new_filename)
    shutil.copy2(img_path, new_path)
    mapping_data.append([new_filename, "Sensitive content"])
    print(f"Copied {img_path} to {new_path}")

# Copy and rename violence images
for i, img_path in enumerate(selected_violence):
    ext = os.path.splitext(img_path)[1]
    new_filename = f"violence_{i+1:03d}{ext}"
    new_path = os.path.join(output_dir, new_filename)
    shutil.copy2(img_path, new_path)
    mapping_data.append([new_filename, "Violence"])
    print(f"Copied {img_path} to {new_path}")

# Copy and rename nonviolence images
for i, img_path in enumerate(selected_nonviolence):
    ext = os.path.splitext(img_path)[1]
    new_filename = f"nonviolence_{i+1:03d}{ext}"
    new_path = os.path.join(output_dir, new_filename)
    shutil.copy2(img_path, new_path)
    mapping_data.append([new_filename, "NonViolence"])
    print(f"Copied {img_path} to {new_path}")

# Sort the mapping data by file path
mapping_data.sort(key=lambda x: x[0])

# Create CSV mapping file
csv_path = os.path.join(output_dir, "image_mapping.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image Path", "Category"])
    writer.writerows(mapping_data)

print(f"\nProcess completed.")
print(f"Total images copied: {len(mapping_data)}")
print(f"CSV mapping file created at: {csv_path}") 