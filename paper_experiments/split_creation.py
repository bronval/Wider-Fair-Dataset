import subprocess

def run_splits_creations():
    result = subprocess.run(
        ["python", "main_train_test_split.py"],  
        cwd="./Binary_linear_programming",                      
    )

if __name__ == "__main__":
    run_splits_creations()