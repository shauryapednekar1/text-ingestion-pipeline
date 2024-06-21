import glob
import json
import os
import zipfile
from typing import List

from langchain_core.documents import Document
from smart_open import open

from .enhanced_document import EnhancedDocument


def unzip_recursively(source_zip, target_dir, max_depth=100):
    # TODO(STP): Make this check cleaner.
    dataset_name = os.path.basename(source_zip)
    assert dataset_name[-4:] == ".zip"
    dataset_name = dataset_name[:-4]
    unzipped_loc = os.path.join(target_dir, dataset_name)

    # HACK(STP): Setting the max depth is a workaround to prevent quines
    # (self-reproducing programs) from leading to unintended behavior.
    # See https://research.swtch.com/zip for more.
    if max_depth < 0:
        print("Max recursion depth reached.")
        return

    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Open the ZIP file
    with zipfile.ZipFile(source_zip, "r") as zip_ref:
        # Extract all the contents into the target directory
        zip_ref.extractall(target_dir)
        print(f"Extracted {source_zip} into {target_dir}")

    # Loop over the directory
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            # Check if the file is a ZIP file
            if file.endswith(".zip"):
                file_path = os.path.join(root, file)
                # Generate new target directory based on ZIP file name
                new_target_dir = os.path.join(root, file[:-4])
                # Recurse into the function with decremented depth
                unzip_recursively(file_path, new_target_dir, max_depth - 1)
                # Optionally delete the ZIP file after extraction
                os.remove(file_path)
                print(f"Deleted {file_path}")

    return unzipped_loc


def save_docs_to_file(
    docs: List[EnhancedDocument],
    original_file_path: str,
    output_dir: str,
):
    """Saves Documents to file on disk."""
    os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.basename(original_file_path)
    output_path = original_file_path
    if output_path.split(".")[-1] != "jsonl":
        output_path += ".jsonl"
    output_path = os.path.join(output_dir, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for doc in docs:
            f.write(doc.json() + "\n")


def get_files_from_dir(root_dir: str):
    """Generator function to iterate over file paths recursively."""
    for path in glob.iglob(os.path.join(root_dir, "**"), recursive=True):
        if os.path.isfile(path):
            yield path


def load_docs_from_jsonl(file_path) -> List[EnhancedDocument]:
    res = []
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = EnhancedDocument(**data)
            res.append(obj)
    return res
