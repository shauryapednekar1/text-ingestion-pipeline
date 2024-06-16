from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import JSONFileLoader
from typing import List
import logging

from langchain_core.documents import Document


def test_generic_callback(x):
    """Converts a file into document(s)."""
    return str(x)


def test_json_callback(x):
    """Converts a json file into document(s)."""
    return str(x["content"])


autoloader_config = {
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

    def __init__(self, dataset_directory, output_location, autoloader_config):
        self.dataset_directory = dataset_directory
        self.output_location = output_location
        self.autoloader_config = autoloader_config
        self.autoloaders = self._get_autoloaders()

    def load_dataset(self, directory):
        """Takes in the location to a dataset and stores them as standardized documents."""
        # TODO(STP): Use multiprocessing
        for file_path in directory:
            self.load_file(file_path)

    def load_file(self, file_path) -> List[Document]:
        # TODO(STP): Convert pseudocode.
        if type(file_path) == "json" and "JSONLoader" in self.autoloaders:
            config = self.autoloader_config["JSONLoader"]
            kwargs = {**config["required"], **config["JSONLoader"]["optional"]}
            try:
                JSONFileLoader(file_path, **kwargs)
            except Exception as e:
                logging.debug(
                    "Filepath %s failed to load using jsonloader: %s\n Falling back to generic loader.",
                    file_path,
                    e,
                )
                self.fallback_loader(file_path)
            pass

        elif type(file_path) == "csv" and "CSVLoader" in self.autoloaders:
            pass

        else:
            # Fallback to unstructured loader.
            return self.fallback_loader(file_path)

    def fallback_loader(file_path) -> List[Document]:
        loader = UnstructuredFileLoader(
            file_path,
            mode="elements",
            strategy="fast",
        )
        docs = loader.load()
        return docs

    def _get_autoloaders(self):
        """Returns the set of valid autoloaders.

        An autoloader is considered valid if the required arguments for
        the autoloader exist in the loader_config.

        This function will only be called once per dataset.
        """
        for autoloader, config in self.autoloader_config:
            usable = True
            required_config = config["required"]
            for required_arg, val in required_config.items():
                if val is None:
                    usable = False
                    break
            if usable:
                self.autoloaders.add(autoloader)


class CustomLoader:

    def load_file(self, curr_file):
        # TODO(STP): Convert pseudocode.
        if type(curr_file) == "json":
            # transform_json()
            pass
        else:
            super().load_file(curr_file)
