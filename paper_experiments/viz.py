import subprocess
import os
import shutil

def collect_summary_folders(source_root='Results', dest_root='viz_thesis\Experimental\Summary_table'):
    """
    Copies all folders named 'summary' from the source directory tree
    into the destination root
    """
    for root, dirs, files in os.walk(source_root):
        if os.path.basename(root) == 'summary':
            relative_path = os.path.relpath(root, source_root)
            dest_path = os.path.join(dest_root, relative_path)

            os.makedirs(dest_path, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_path, file)
                shutil.copy2(src_file, dst_file)

    print(f"All 'summary' folders copied from '{source_root}' to '{dest_root}' successfully.")


def create_dataset_description():
    result = subprocess.run(
        ["python", "table_viz.py"],  
        cwd="./Train_test_dataset",                      
    )
    
if __name__ == "__main__":
    create_dataset_description()
    collect_summary_folders()