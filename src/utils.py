import os
import zipfile


def unzip_recursively(source_zip, target_dir, max_depth=100):
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


# Usage example
# unzip_recursively('data/financial_dataset.zip', 'data/financial_dataset')
