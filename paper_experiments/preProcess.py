import subprocess

def run_building_dataset():
    result = subprocess.run(
        ["python", "building_dataset.py"],  
        cwd="./Pre_processing_dataset",                      
    )
def run_comparison_dataset():
    result = subprocess.run(
        ["python", "comparison_filtering.py"],  
        cwd="./Pre_processing_dataset",                      
    )
if __name__ == "__main__":
    run_building_dataset()
    run_comparison_dataset()
