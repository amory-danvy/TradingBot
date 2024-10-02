import os
import subprocess
import sys

def run_streamlit():
    streamlit_script = """
import streamlit as st
from interface import run_app

if __name__ == '__main__':
    run_app()
    """
    temp_script = "temp_streamlit_script.py"
    with open(temp_script, "w") as f:
        f.write(streamlit_script)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", temp_script])
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)

if __name__ == "__main__":
    run_streamlit()