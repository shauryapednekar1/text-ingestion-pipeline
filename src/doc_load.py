import logging
import multiprocessing
import os
import tempfile
import zipfile
from functools import partial
from typing import List, Optional, Set

from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    UnstructuredFileLoader,
)
from langchain_core.documents import Document
from tqdm import tqdm

from defaults import DEFAULT_AUTOLOADER_CONFIG
from utils import get_files_from_dir, save_docs_to_file, unzip_recursively


class Loader:

    def __init__(
        self,
        autoloader_config: dict = DEFAULT_AUTOLOADER_CONFIG,
    ) -> None:
        self.autoloader_config: dict = autoloader_config
        self.autoloaders: Set[str] = self._get_valid_autoloaders()

    def load_dataset(
        self,
        input_dir: str,
        is_zipped: bool = False,
        unzip_dir="unzipped",
        save_docs=False,
        output_dir=None,
        detailed_progress=False,
        num_workers=10,
        max_files=None,
    ) -> None:
        """

        Takes in the location to a dataset and stores them as standardized
        Document objects.
        """
        if save_docs and output_dir is None:
            raise ValueError(
                "Must provide an output directory when saving documents."
            )

        logging.debug("Loading dataset from %s", input_dir)

        if is_zipped:
            directory = self.unzip_dataset(input_dir, unzip_dir)
        else:
            directory = input_dir

        num_files = None
        if detailed_progress:
            num_files = len(list(get_files_from_dir(directory)))

        partial_func = partial(self.load_file, save_docs, output_dir)

        with tqdm(
            total=num_files, desc="Loading files", unit=" files"
        ) as pbar:
            with multiprocessing.Pool(num_workers) as pool:
                for i, _ in enumerate(
                    pool.imap_unordered(
                        partial_func,
                        get_files_from_dir(directory),
                    )
                ):
                    pbar.update(1)
                    if max_files is not None and i + 1 >= max_files:
                        break

    def load_file(
        self, save_docs: bool, output_dir: Optional[str], file_path: str
    ) -> None:
        logging.debug(f"Loading {file_path}")
        docs = self.file_to_docs(file_path)
        if save_docs:
            assert output_dir is not None
            save_docs_to_file(docs, file_path, output_dir)
        logging.debug(f"{file_path} loaded")

    def file_to_docs(self, file_path: str) -> List[Document]:
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

    def fallback_loader(self, file_path) -> List[Document]:
        logging.info("Using fallback loader for %s.", file_path)
        loader = UnstructuredFileLoader(
            file_path,
            mode="elements",
            strategy="fast",
        )
        docs = loader.load()
        return docs

    def unzip_dataset(self, input_dir: str, unzip_dir: str) -> str:
        os.makedirs(unzip_dir, exist_ok=True)
        directory = os.path.join(
            unzip_dir,
            os.path.dirname(input_dir),
            # TODO(STP): Maybe use a helper to remove file extension here.
            os.path.basename(input_dir)[:-4],
        )
        unzip_recursively(input_dir, directory)
        return directory

    def _get_valid_autoloaders(self) -> Set[str]:
        """Returns the set of valid autoloaders.

        An autoloader is considered valid if the required arguments for
        the autoloader exist in the loader_config.

        This function will only be called once per dataset.
        """
        autoloaders = set()
        for autoloader, config in self.autoloader_config.items():
            usable = True
            required_config = config["required"]
            for required_arg, val in required_config.items():
                if val is None:
                    usable = False
                    break
            if usable:
                autoloaders.add(autoloader)

        return autoloaders


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