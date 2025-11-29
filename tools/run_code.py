from google import genai
import subprocess
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
from google.genai import types
load_dotenv()
google_client = genai.Client()

def clean_code_formatting(source_code: str) -> str:
    source_code = source_code.strip()
    # Remove ```python ... ``` or ``` ... ```
    if source_code.startswith("```"):
        # remove first line (```python or ```)
        source_code = source_code.split("\n", 1)[1]
    if source_code.endswith("```"):
        source_code = source_code.rsplit("\n", 1)[0]
    return source_code.strip()

@tool
def run_code(source_code: str) -> dict:
    """
    Executes a Python code 
    This tool:
      1. Takes in python code as input
      3. Writes code into a temporary .py file
      4. Executes the file
      5. Returns its output

    Parameters
    ----------
    source_code : str
        Python source code to execute.

    Returns
    -------
    dict
        {
            "stdout": <program output>,
            "stderr": <errors if any>,
            "return_code": <exit code>
        }
    """
    try: 
        script_filename = "runner.py"
        os.makedirs("LLMFiles", exist_ok=True)
        with open(os.path.join("LLMFiles", script_filename), "w") as script_file:
            script_file.write(source_code)

        execution_process = subprocess.Popen(
            ["uv", "run", script_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="LLMFiles"
        )
        std_output, std_error = execution_process.communicate()
        if len(std_output) >= 10000:
            return std_output[:10000] + "...truncated due to large size"
        if len(std_error) >= 10000:
            return std_error[:10000] + "...truncated due to large size"
        # --- Step 4: Return everything ---
        return {
            "stdout": std_output,
            "stderr": std_error,
            "return_code": execution_process.returncode
        }
    except Exception as execution_error:
        return {
            "stdout": "",
            "stderr": str(execution_error),
            "return_code": -1
        }