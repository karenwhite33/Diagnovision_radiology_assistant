import os
import pandas as pd

#  Load the full dataset CSV with image paths
full_df_path = "D:/AI Bootcamp Github/Proyecto FInal/CheXpert/full_dataset.csv" 
full_df = pd.read_csv(full_df_path)

# Get the image paths from the CSV
image_paths_in_csv = full_df["path_to_image"].tolist()

#  Define the main folder containing the subfolders (train, valid, etc.)
image_folder_path = "D:/AI Bootcamp Github/Proyecto FInal/CheXpert/CheXpert-v1.0-small"  # Folder containing train/valid subfolders

#  Walk through all the subfolders (train/, valid/) and check for the images
missing_images = []

# Walk through the folder structure (train, valid, etc.)
for root, dirs, files in os.walk(image_folder_path):
    # Check for image files and match them with the paths in the CSV
    for file in files:
        if file.endswith(".jpg"):
            # Construct the relative file path in the same format as the CSV
            relative_path = os.path.relpath(os.path.join(root, file), image_folder_path).replace("\\", "/")
            
            # Check if the image path exists in the CSV
            if relative_path in image_paths_in_csv:
                image_paths_in_csv.remove(relative_path)  

# Step 4: Print out the results
if image_paths_in_csv:
    print(f"❌ Missing images: {len(image_paths_in_csv)}")
    print(f"Missing images: {image_paths_in_csv[:5]}") 
else:
    print("✅ All images in the dataset are present in the folder.")
