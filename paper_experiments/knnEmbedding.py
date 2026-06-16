import subprocess

def run_embeddings():
    result = subprocess.run(
        ["python", "embedding.py"],  
        cwd="./Embedding_space",                      
    )

if __name__ == "__main__":
    run_embeddings()