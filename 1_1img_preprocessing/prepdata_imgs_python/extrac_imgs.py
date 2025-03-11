import zipfile
import os
import pandas as pd
from tqdm import tqdm
import shutil

#Load the CSV file with image paths (make sure the path to your CSV is correct)
full_df_path = "D:/AI Bootcamp Github/Proyecto FInal/CheXpert/full_dataset.csv"  # Update with the actual path to your CSV
full_df = pd.read_csv(full_df_path)

# Get the image paths from the CSV
image_paths_to_extract = full_df["path_to_image"].tolist()

#  Define the ZIP file path where the images will be saved
zip_file_path = "D:/AI Bootcamp Github/Proyecto FInal/CheXpert/CheXpert-v1.0-small.zip" 
output_folder = "D:/AI Bootcamp Github/Proyecto FInal/CheXpert/dataset_extracted" 

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the ZIP file and extract only the images that are in the full_df
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Get all files in the ZIP
    all_files_in_zip = zip_ref.namelist()

    # Filter files based on the image paths in the full_df
    images_to_extract = [file for file in all_files_in_zip if file.endswith('.jpg') and file in image_paths_to_extract]

    #  Extract the filtered images and preserve the folder structure
    for file in tqdm(images_to_extract, desc="Extracting images", unit="image"):
        # Extract the file to its correct subfolder in the output folder
        extracted_file_path = os.path.join(output_folder, file)
        
        # Create any necessary subfolders in the output folder
        os.makedirs(os.path.dirname(extracted_file_path), exist_ok=True)
        
        # Extract the image file into the correct folder
        zip_ref.extract(file, output_folder)

print(f"✅ All relevant images have been extracted to {output_folder}, preserving folder structure.")
