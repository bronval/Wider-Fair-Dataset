import pandas as pd
import os
import shutil

def create_subset_images(df_filtered):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    wider_merged_path = os.path.join(project_root, 'preProcessingDataset', 'Wider_merged')
    subset_folder = os.path.join(project_root, "subset_images") 

    os.makedirs(subset_folder, exist_ok=True)

    all_filenames = set(df_filtered["filename"].unique())

    for root, dirs, files in os.walk(wider_merged_path):
        relative_root = os.path.relpath(root, wider_merged_path)
        if relative_root != ".":
            dest_dir = os.path.join(subset_folder, relative_root)
            os.makedirs(dest_dir, exist_ok=True)

    for filename in all_filenames:
        found = False

        full_image_path = os.path.join(wider_merged_path, filename)
        relative_dir = os.path.dirname(filename)
        dest_dir = os.path.join(subset_folder, relative_dir)

        if os.path.exists(full_image_path):
            os.makedirs(dest_dir, exist_ok=True)
            try:
                shutil.copy(full_image_path, os.path.join(dest_dir, os.path.basename(filename)))
                found = True
            except Exception as e:
                print(f"Error copying {filename}: {e}")

        if not found:
            print(f"Warning: {filename} not found in Wider_merged folder!")




