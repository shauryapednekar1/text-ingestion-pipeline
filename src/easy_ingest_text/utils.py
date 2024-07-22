"""Utility functions used by package."""

import glob
import json
import os
import zipfile
from typing import Generator, List

from .enhanced_document import EnhancedDocument

# NOTE(STP): We might want to use smart_open when dealing with datasets
# that are stored in the cloud.
# from smart_open import open


def unzip_recursively(
    source_zip: str, target_dir: str, max_depth: int = 100
) -> str:
    """
    Recursively unzips a ZIP file into a target directory, including any nested
    ZIP files.

    Args:
        source_zip: The file path of the source ZIP file.
        target_dir: The directory where the ZIP file contents should be
            extracted.
        max_depth: The maximum depth for recursion to avoid infinite loops.

    Returns:
        The path to the directory where the ZIP file was ultimately extracted.

    Raises:
        AssertionError: If the source_zip does not end with '.zip'.
    """
    # TODO(STP): Make this check cleaner.
    if source_zip[-4:] != ".zip":
        raise ValueError(
            "Zipped dataset name must end in '.zip', not %s. "
            "Received dataset name: %s",
            source_zip[-4:],
            source_zip,
        )

    # HACK(STP): Setting the max depth is a workaround to prevent quines
    # (self-reproducing programs) from leading to unintended behavior.
    # See https://research.swtch.com/zip for more.
    if max_depth < 0:
        print("Max recursion depth reached.")
        return

    os.makedirs(target_dir, exist_ok=True)

    with zipfile.ZipFile(source_zip, "r") as zip_ref:
        zip_ref.extractall(target_dir)
        print(f"Extracted {source_zip} into {target_dir}")

    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".zip"):
                file_path = os.path.join(root, file)
                new_target_dir = os.path.join(root, file[:-4])
                unzip_recursively(file_path, new_target_dir, max_depth - 1)
                os.remove(file_path)
                print(f"Deleted {file_path}")

    return target_dir


def save_docs_to_file(
    docs: List[EnhancedDocument],
    original_file_path: str,
    output_dir: str,
) -> None:
    """Saves a list of documents to a JSONL file in the specified directory.

    Args:
        docs: A list of EnhancedDocument objects to be saved.
        original_file_path: The path to the original file for naming the output.
        output_dir: The directory where the output JSONL file will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = (
        original_file_path
        if original_file_path.split(".")[-1] == "jsonl"
        else original_file_path + ".jsonl"
    )
    output_path = os.path.join(output_dir, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for doc in docs:
            f.write(doc.json() + "\n")


def get_files_from_dir(root_dir: str) -> Generator[str, None, None]:
    """Yields file paths recursively from a given directory.

    Args:
        root_dir: The root directory from which to start the search.

    Yields:
        The path to each file found.
    """
    for path in glob.iglob(os.path.join(root_dir, "**"), recursive=True):
        if os.path.isfile(path):
            yield path


def load_docs_from_jsonl(file_path: str) -> List[EnhancedDocument]:
    """
    Loads documents from a JSONL file and returns them as a list of
    EnhancedDocument objects.

    Args:
        file_path: The file path to the JSONL file containing the documents.

    Returns:
        A list of EnhancedDocument objects parsed from the JSONL file.
    """
    res = []
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = EnhancedDocument(**data)
            res.append(obj)
    return res
