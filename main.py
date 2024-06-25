import subprocess
import sys

def install_libraries():
    print("Installing required libraries...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytube", "--quiet"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "git+https://github.com/huggingface/transformers.git", "accelerate", "datasets[audio]", "--quiet"])

def mkt():
    install_libraries()
    print("Libraries installed successfully.")
