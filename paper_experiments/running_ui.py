import subprocess

def run_ui():
    result = subprocess.run(
        ["python", "ui.py"],  
        cwd="./Annotator",                      
    )
if __name__ == "__main__":
    
    run_ui()