from langchain_core.tools import tool
import requests
import os

@tool
def download_file(source_url: str, target_filename: str) -> str:
    """
    Download a file from a URL and save it with the given filename
    in the current working directory.

    Args:
        source_url (str): Direct URL to the file.
        target_filename (str): The filename to save the downloaded content as.

    Returns:
        str: Full path to the saved file.
    """
    try:
        http_response = requests.get(source_url, stream=True)
        http_response.raise_for_status()
        storage_directory = "LLMFiles"
        os.makedirs(storage_directory, exist_ok=True)
        file_path = os.path.join(storage_directory, target_filename)
        with open(file_path, "wb") as file_handle:
            for data_chunk in http_response.iter_content(chunk_size=8192):
                if data_chunk:
                    file_handle.write(data_chunk)

        return target_filename
    except Exception as download_error:
        return f"Error downloading file: {str(download_error)}"