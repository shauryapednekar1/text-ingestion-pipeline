import glob
import logging
import multiprocessing
import os
import tempfile
import zipfile
from functools import partial
from typing import List

from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    UnstructuredFileLoader,
)
from langchain_core.documents import Document
from smart_open import open
from tqdm import tqdm

from utils import unzip_recursively

default_autoloader_config = {
    "JSONLoader": {
        "required": {
            "jq_schema": None,  # "",
        },
        "optional": {
            "content_key": None,
            "is_content_key_jq_parsable": False,
            "metadata_func": None,
            "text_content": True,
            "json_lines": False,
        },
    },
    "CSVLoader": {
        "required": {},
        "optional": {
            "source_column": None,
            "metadata_columns": (),
            "csv_args": None,
            "encoding": None,
            "autodetect_encoding": False,
        },
    },
}


class Loader:

    def __init__(
        self,
        dataset_directory,
        is_zipped,
        unzip=True,
        autoloader_config=default_autoloader_config,
        num_workers=10,
        save_docs=False,
        output_location=None,
    ) -> None:
        self.dataset_directory = dataset_directory
        self.output_location = output_location
        self.autoloader_config = autoloader_config
        self.is_zipped = is_zipped
        self.unzip = unzip
        self.autoloaders = self._get_valid_autoloaders()
        self.num_workers = 10
        self.save_docs = save_docs

        if self.save_docs and self.output_location is None:
            raise ValueError(
                "Must provide an output location when saving documents."
            )

    def load_dataset(self, directory=None) -> List[Document]:
        """

        Takes in the location to a dataset and stores them as standardized
        Document objects.
        """
        if directory is None:
            directory = self.dataset_directory

        logging.debug("Loading dataset from %s", directory)
        if self.is_zipped and not self.unzip:
            warning_message = """
            Dataset is compressed but will not be unzipped.
            This is really slow and hasn't been optimized yet, so use with 
            caution. Alternatively, pass unzip=True when instantiating the
            Loader instance, which results in significantly faster processing
            speeds but takes up more disk space.
            """
            logging.warning(warning_message)

            with zipfile.ZipFile(directory, "r") as z:
                # NOTE(STP): Converting to a list here in order to provide
                # a progress bar. However, this is not memory efficient and
                # might not be worth the tradeoff, especially for massive
                # datasets.
                infolist = list(z.infolist())
                # NOTE(STP): len(infolist) isn't strictly accurate since it
                # includes objects that aren't files. However, it's a
                # reasonable approximation.
                num_files = len(infolist)
                partial_func = partial(self.load_zipped_file, directory)

                with tqdm(total=num_files) as pbar:
                    with multiprocessing.Pool(self.num_workers) as pool:
                        for _ in pool.imap_unordered(partial_func, infolist):
                            pbar.update(1)
        else:
            if self.is_zipped:
                # TODO(STP): Remove magic string.
                unzipped_directory = "unzipped_data"
                os.makedirs(unzipped_directory, exist_ok=True)
                # TODO(STP): Fix filenames here. Right now, the locations for
                # the unzipped data and standardized data are fairly
                # inflexible.
                directory = os.path.join(
                    unzipped_directory,
                    os.path.basename(self.dataset_directory),
                )
                unzip_recursively(self.dataset_directory, directory)

            all_files = list(glob.iglob(f"{directory}/**", recursive=True))
            with tqdm(total=len(all_files)) as pbar:
                with multiprocessing.Pool(self.num_workers) as pool:
                    for _ in pool.imap_unordered(
                        self.load_regular_file, all_files
                    ):
                        pbar.update(1)

    def load_zipped_file(
        self, directory: str, zipped_file_info: zipfile.ZipInfo
    ) -> None:
        logging.debug(f"Loading {zipped_file_info}")
        with zipfile.ZipFile(directory, "r") as z:
            if not zipped_file_info.is_dir():
                with tempfile.TemporaryDirectory() as temp_dir:
                    z.extract(zipped_file_info, path=temp_dir)
                    extracted_file_path = os.path.join(
                        temp_dir, zipped_file_info.filename
                    )
                    docs = self.file_to_docs(extracted_file_path)
                    if self.save_docs:
                        self.save_file(docs, extracted_file_path)
        logging.debug(f"{zipped_file_info} loaded")

    def load_regular_file(self, file_path: str) -> None:
        logging.debug(f"Loading {file_path}")
        if os.path.isfile(file_path):
            docs = self.file_to_docs(file_path)
            if self.save_docs:
                self.save_file(docs, file_path)
        logging.debug(f"{file_path} loaded")

    def file_to_docs(self, file_path) -> List[Document]:
        # NOTE(STP): Switching to unstructured's file-type detection in the
        # future might be worthwhile (although their check for whether a file
        # is a JSON file is whether or not json.load() succeeds, which might
        # not be performant?).
        # See https://github.com/Unstructured-IO/unstructured/blob/main/unstructured/file_utils/filetype.py # noqa: E501
        file_extension = file_path.split(".")[-1]
        if file_extension == "json" and "JSONLoader" in self.autoloaders:
            config = self.autoloader_config["JSONLoader"]
            kwargs = {**config["required"], **config["JSONLoader"]["optional"]}
            try:
                loader = JSONLoader(file_path, **kwargs)
                docs = loader.load()
                return docs
            except Exception as e:
                logging.debug(
                    "Filepath %s failed to load using JSONLoader: %s\n "
                    "Falling back to generic loader.",
                    file_path,
                    e,
                )
                return self.fallback_loader(file_path)

        elif file_extension == "csv" and "CSVLoader" in self.autoloaders:
            config = self.autoloader_config["CSVLoader"]
            kwargs = {**config["required"], **config["CSVLoader"]["optional"]}
            try:
                loader = CSVLoader(file_path, **kwargs)
                docs = loader.load()
                return docs

            except Exception as e:
                logging.debug(
                    "Filepath %s failed to load using CSVLoader: %s\n",
                    "Falling back to generic loader.",
                    file_path,
                    e,
                )
                return self.fallback_loader(file_path)

        else:
            # Fallback to unstructured loader.
            return self.fallback_loader(file_path)

    def save_file(self, docs: List[Document], original_file_path: str):
        """Saves a document to disk."""
        os.makedirs(self.output_location, exist_ok=True)
        output_path = original_file_path + ".jsonl"
        output_path = os.path.join(self.output_location, output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            for doc in docs:
                f.write(doc.json() + "\n")

    def fallback_loader(self, file_path) -> List[Document]:
        logging.info("Using fallback loader for %s.", file_path)
        loader = UnstructuredFileLoader(
            file_path,
            mode="elements",
            strategy="fast",
        )
        docs = loader.load()
        return docs

    def _get_valid_autoloaders(self):
        """Returns the set of valid autoloaders.

        An autoloader is considered valid if the required arguments for
        the autoloader exist in the loader_config.

        This function will only be called once per dataset.
        """
        self.autoloaders = set()
        for autoloader, config in self.autoloader_config.items():
            usable = True
            required_config = config["required"]
            for required_arg, val in required_config.items():
                if val is None:
                    usable = False
                    break
            if usable:
                self.autoloaders.add(autoloader)


"""
class CustomLoader:

    def load_file(self, file_path):
        # TODO(STP): Convert pseudocode.
        file_extension = file_path.split(".")[-1]
        if file_extension == "json":
            with open(file_path) as fin:
                data = json.load(fin)
                text = data["text"]
                del data["text"]
                metadata = data
                return Document(page_content=text, metadata=metadata)
        else:
            super().load_file(file_path)
"""
