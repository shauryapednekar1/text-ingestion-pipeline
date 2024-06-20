import logging
import multiprocessing
from functools import partial
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter
from tqdm import tqdm

from defaults import DEFAULT_SPLITTERS_CONFIG
from enhanced_document import EnhancedDocument
from utils import get_files_from_dir, load_docs_from_jsonl, save_docs_to_file


class Chunker:
    """Chunk documents.

    Potential functions to override if implementing a custom Chunker class:
    - `chunk_docs()`: the logic for how a document is is chunked.
    - `get_splitter()`: the logic for which splitter to use.
    """

    ALLOWED_SPLITTERS = {"custom", "recursive"}

    def __init__(
        self,
        splitter: str = "recursive",
        splitter_config: Dict = DEFAULT_SPLITTERS_CONFIG,
        num_workers: int = 10,
    ) -> None:
        """
        Initializes the Chunker instance with specified splitter configuration
        and multiprocessing settings.

        Args:
            splitter (str): The name of the splitter to use. Currently supports
                'custom' or 'recursive'.
            splitter_config (Dict): Configuration dictionary for the chosen
                splitter.
            num_workers (int): Number of worker processes to use for chunking.

        Raises:
            ValueError: If the specified splitter is not supported.
        """

        self.splitter_config = splitter_config
        self.num_workers = num_workers

        if splitter not in self.ALLOWED_SPLITTERS:
            raise ValueError(
                f"{splitter} is not a valid splitter."
                f" Choose from: {self.ALLOWED_SPLITTERS}"
            )
        self.splitter = self.get_splitter(splitter)

    def chunk_dataset(
        self,
        input_dir: str,
        detailed_progress: bool = False,
        output_dir: Optional[str] = None,
        num_workers: Optional[int] = None,
    ) -> None:
        """
        Processes and chunks all documents in a specified directory.

        Args:
            input_dir (str): Directory containing the documents to be chunked.
            output_dir (str, optional): Directory to save the chunked documents
                if save_chunks is True.
            detailed_progress (bool): Whether to show detailed progress during
                the chunking process.
            num_workers (int, optional): Number of worker processes to use;
                defaults to the instance's num_workers if not provided.

        Raises:
            ValueError: If `save_chunks` is True but `output_dir` is not provided.
        """
        if num_workers is None:
            num_workers = self.num_workers

        num_files = None
        if detailed_progress:
            num_files = len(list(get_files_from_dir(input_dir)))

        partial_func = partial(
            self.chunk_file, save_chunks=True, output_dir=output_dir
        )

        with tqdm(
            total=num_files, desc="Chunking files", unit=" files", smoothing=0
        ) as pbar:
            with multiprocessing.Pool(num_workers) as pool:
                for _ in pool.imap_unordered(
                    partial_func,
                    get_files_from_dir(input_dir),
                ):
                    pbar.update(1)

    def chunk_file(
        self, save_chunks: bool, output_dir: Optional[str], file_path: str
    ) -> List[EnhancedDocument]:
        """
        Chunks a single file into smaller EnhancedDocuments.

        Args:
            save_chunks (bool): Whether to save the chunked documents.
            output_dir (str): Directory to save the documents if save_chunks
                is True.
            file_path (str): Path to the file to be chunked.

        Returns:
            List[EnhancedDocument]: A list of chunked documents.
        """
        logging.debug("Chunking file: %s", file_path)
        raw_docs = load_docs_from_jsonl(file_path)
        chunked_docs = self.chunk_docs(raw_docs)
        if save_chunks:
            if output_dir is None:
                raise ValueError(
                    "Must provide an output directory when saving documents."
                )
            save_docs_to_file(chunked_docs, file_path, output_dir)
        logging.debug("Chunked file: %s", file_path)
        return chunked_docs

    def chunk_docs(
        self, raw_docs: List[EnhancedDocument]
    ) -> List[EnhancedDocument]:
        """
        Splits a list of EnhancedDocuments into smaller, chunked
        EnhancedDocuments.

        Args:
            raw_docs (List[EnhancedDocument]): List of documents to be chunked.

        Returns:
            List[EnhancedDocument]: A list of chunked documents.
        """

        chunked_docs = self.splitter.split_documents(raw_docs)
        # NOTE(STP): We need to remove the hashes here since the page content
        # for the chunk differs from the parent document.
        docs = [EnhancedDocument.remove_hashes(doc) for doc in chunked_docs]
        docs = [EnhancedDocument.from_document(doc) for doc in docs]
        return docs

    def get_splitter(self, splitter: str) -> TextSplitter:
        """
        Retrieves the appropriate document splitter based on the specified type.

        Args:
            splitter (str): The name of the splitter to use.

        Returns:
            TextSplitter: An instance of a TextSplitter.

        Raises:
            NotImplementedError: If a 'custom' splitter is specified but
                not implemented.
            ValueError: If the specified splitter type is not recognized.
        """
        if splitter == "custom":
            error_message = """
            "If using custom vectorstore, the Embedder.set_vectorstore() method
            must be overridden.
            """
            raise NotImplementedError(error_message)

        if splitter == "recursive":
            kwargs = self.splitter_config["recursive"]
            return RecursiveCharacterTextSplitter(**kwargs)
        else:
            raise ValueError("Splitter not recognized: %s", splitter)
