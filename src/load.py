import logging
import multiprocessing
import os
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
from enhanced_document import EnhancedDocument
from utils import get_files_from_dir, save_docs_to_file, unzip_recursively


class Loader:

    def __init__(
        self,
        autoloader_config: dict = DEFAULT_AUTOLOADER_CONFIG,
    ) -> None:
        """
        Initializes a Loader instance with a given autoloader configuration.

        Args:
            autoloader_config (dict): Configuration for autoloaders that
                determines how different file types are processed.

        Attributes:
            autoloader_config (dict): Stores the provided autoloader
                configuration.
            autoloaders (Set[str]): Set of valid autoloaders based on the
                configuration.
        """
        self.autoloader_config: dict = autoloader_config
        self.autoloaders: Set[str] = self._get_valid_autoloaders()

    def load_dataset(
        self,
        input_dir: str,
        output_dir: str,
        is_zipped: bool = False,
        unzip_dir: str = "unzipped",
        detailed_progress: bool = False,
        num_workers: int = 10,
        max_files: Optional[int] = None,
    ) -> None:
        """
        Loads a dataset from a specified directory, processes files into
        EnhancedDocument objects, and saves them to disk.

        Args:
            input_dir (str): Path to the directory containing the dataset.
            is_zipped (bool): Whether the dataset is in a zipped format.
            unzip_dir (str): Directory to unzip files to, if applicable.
            output_dir (str): Directory where processed documents should be
                saved.
            detailed_progress (bool): Whether to display detailed progress
                information.
            num_workers (int): Number of worker processes to use for loading
                files.
            max_files (int, optional): Maximum number of files to process.
        """
        logging.debug("Loading dataset from %s", input_dir)

        if is_zipped:
            # TODO(STP): Make this check cleaner.
            dataset_name = os.path.basename(input_dir)[:-4]
            if dataset_name[:-4] != ".zip":
                raise ValueError(
                    "Zipped dataset name must end in '.zip'. Received dataset "
                    "name: %s",
                    dataset_name,
                )
            directory = os.path.join(unzip_dir, dataset_name)
            directory = self.unzip_dataset(input_dir, unzip_dir)
        else:
            directory = input_dir

        num_files = None
        if detailed_progress:
            num_files = len(list(get_files_from_dir(directory)))

        partial_func = partial(
            self.load_file, save_docs=True, output_dir=output_dir
        )

        with tqdm(
            total=num_files, desc="Loading files", unit="files", smoothing=0
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
        """
        Loads a single file from the given path and optionally saves the
        processed document.

        Args:
            save_docs (bool): Whether to save the processed documents.
            output_dir (str, optional): Directory where processed documents
                should be saved.
            file_path (str): Path to the file being loaded.

        Raises:
            AssertionError: If `save_docs` is True but `output_dir` is None.
        """
        logging.debug("Loading file: %s", file_path)
        docs = self.file_to_docs(file_path)
        if save_docs:
            assert output_dir is not None
            save_docs_to_file(docs, file_path, output_dir)
        logging.debug("Loaded file: %s", file_path)

    def file_to_docs(self, file_path: str) -> List[EnhancedDocument]:
        """
        Processes a file into a list of EnhancedDocument objects based on the
        file extension and configured autoloaders.

        Args:
            file_path (str): Path to the file being processed.

        Returns:
            List[EnhancedDocument]: A list of EnhancedDocument objects created
                from the file.
        """
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
            except Exception as e:
                logging.debug(
                    "Filepath %s failed to load using JSONLoader: %s\n "
                    "Falling back to generic loader.",
                    file_path,
                    e,
                )
                docs = self.fallback_loader(file_path)

        elif file_extension == "csv" and "CSVLoader" in self.autoloaders:
            config = self.autoloader_config["CSVLoader"]
            kwargs = {**config["required"], **config["optional"]}
            try:
                loader = CSVLoader(file_path, **kwargs)
                docs = loader.load()

            except Exception as e:
                logging.debug(
                    "Filepath %s failed to load using CSVLoader: %s\n"
                    "Falling back to generic loader.",
                    file_path,
                    e,
                )
                docs = self.fallback_loader(file_path)

        else:
            # Fallback to unstructured loader.
            docs = self.fallback_loader(file_path)

        enhanced_docs = [EnhancedDocument.from_document(doc) for doc in docs]
        return enhanced_docs

    def fallback_loader(self, file_path) -> List[Document]:
        """
        Uses a generic loader to process files when specific loaders are not
        applicable or fail.

        Args:
            file_path (str): Path to the file being loaded.

        Returns:
            List[Document]: A list of Document objects loaded using the
                fallback method.
        """
        logging.info("Using fallback loader for %s.", file_path)
        loader = UnstructuredFileLoader(
            file_path,
            mode="elements",
            strategy="fast",
        )
        docs = loader.load()
        return docs

    def unzip_dataset(self, input_dir: str, unzip_dir: str) -> str:
        """
        Unzips a dataset from a specified input directory into a target unzip
        directory.

        Args:
            input_dir (str): Path to the zipped dataset.
            unzip_dir (str): Target directory for the unzipped files.

        Returns:
            str: Path to the directory containing the unzipped files.
        """
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

        Returns:
            Set[str]: A set of autoloaders that have all required arguments
                available.
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
