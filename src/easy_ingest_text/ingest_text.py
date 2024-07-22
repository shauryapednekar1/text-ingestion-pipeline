import logging
import multiprocessing as mp
import os
import pickle
from functools import partial
from itertools import chain, islice
from multiprocessing import Pool
from typing import List, Optional

from tqdm import tqdm

from .chunk_text import Chunker
from .embed_text import Embedder
from .enhanced_document import EnhancedDocument
from .load_text import Loader
from .utils import get_files_from_dir, save_docs_to_file


class Ingester:
    """Ingest files into a vectorstore."""

    def __init__(
        self,
        loader: Loader = Loader(),
        chunker: Chunker = Chunker(),
        embedder: Embedder = Embedder(),
    ) -> None:
        """
        Initializes an Ingester instance with components for loading, chunking,
        and embedding documents.

        Args:
            loader (Loader): An instance of Loader to handle document loading.
            chunker (Chunker): An instance of Chunker to handle document
                chunking.
            embedder (Embedder): An instance of Embedder to handle document
                embedding.
        """
        self.loader: Loader = loader
        self.chunker: Chunker = chunker
        self.embedder: Embedder = embedder

    def ingest_dataset(
        self,
        input_dir: str,
        is_zipped: bool = False,
        unzip_dir: Optional[str] = None,
        save_intermediate_docs: bool = False,
        output_dir: Optional[str] = None,
        num_workers: int = 10,
        max_files: Optional[int] = None,
        detailed_progress: bool = False,
        embed_batch_size: int = 1000,  # in number of docs
        chunk_batch_size: int = 100,  # in number of files
    ) -> None:
        """
        Processes a dataset through specified stages: loading, chunking, and
        embedding.

        Args:
            input_dir (str): Directory containing the documents.
            is_zipped (bool): Whether the input directory is zipped.
            unzip_dir (Optional[str]): Directory to unzip files if zipped.
            save_intermediate_docs (bool): Whether to save the loaded and
                chunked documents to disk.
            output_dir (Optional[str]): Directory where processed documents are
                saved.
            num_workers (int): Number of worker processes to use.
            max_files (Optional[int]): Max number of files to process.
            detailed_progress (bool): Whether to show detailed progress.
            embed_batch_size (int): Number of documents to process in each
                batch for embedding.
            chunk_batch_size (int): Number of files to process in each batch
                for chunking.
        """
        if is_zipped:
            directory = self.loader.unzip_dataset(input_dir, unzip_dir)
        else:
            directory = input_dir

        num_files = None
        if detailed_progress:
            total_files = len(list(get_files_from_dir(directory)))
            num_files = min(total_files, max_files)

        with tqdm(
            total=num_files, desc="Ingesting files", unit="files", smoothing=0
        ) as pbar:
            batched_docs = []
            prev_counter = 0
            for i, file_path in enumerate(get_files_from_dir(directory)):

                docs = self.load_and_chunk_file(
                    save_intermediate_docs=save_intermediate_docs,
                    output_dir=output_dir,
                    file_path=file_path,
                )
                batched_docs.extend(docs)

                if len(batched_docs) >= embed_batch_size:
                    self.embedder.embed_and_insert_docs(
                        batched_docs, embed_batch_size
                    )
                    batched_docs = []

                    pbar.update(i - prev_counter)
                    prev_counter = i

                if max_files is not None and i >= max_files:
                    break

            if batched_docs:
                self.embedder.embed_and_insert_docs(
                    batched_docs, embed_batch_size
                )

            # NOTE(STP): Save the vectorstore to disk if applicable.
            self.embedder.save_vectorstore_if_applicable()

    def load_and_chunk_file(
        self,
        save_intermediate_docs: bool,
        output_dir: Optional[str],
        file_path: str,
    ) -> List[EnhancedDocument]:
        """
        Loads and chunks a file, optionally saving both raw and chunked
            documents.

        Args:
            save_intermediate_docs (bool): Whether to save the documents after
                processing.
            output_dir (Optional[str]): Directory to save the documents if
                `save_docs` is True.
            file_path (str): Path to the file to be processed.

        Returns:
            List[EnhancedDocument]: A list of chunked EnhancedDocument objects.

        Raises:
            AssertionError: If `save_docs` is True but no output directory is
                provided.
        """
        logging.debug("Loading and chunking: %s", file_path)
        raw_docs = self.loader.file_to_docs(file_path)
        chunked_docs = self.chunker.chunk_docs(raw_docs)
        if save_intermediate_docs:
            assert output_dir is not None
            raw_documents_dir = os.path.join(output_dir, "raw_documents")
            chunked_documents_dir = os.path.join(
                output_dir, "chunked_documents"
            )
            save_docs_to_file(raw_docs, file_path, raw_documents_dir)
            save_docs_to_file(chunked_docs, file_path, chunked_documents_dir)

        logging.debug("Loaded and chunked: %s", file_path)
        return chunked_docs
