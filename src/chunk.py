import glob
import json
import logging
import multiprocessing
import os
from functools import partial
from typing import Callable, Dict, Iterable, List, Optional, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter
from tqdm import tqdm

from defaults import DEFAULT_SPLITTERS_CONFIG
from enhanced_document import EnhancedDocument
from utils import get_files_from_dir, load_docs_from_jsonl, save_docs_to_file


class Chunker:
    ALLOWED_SPLITTERS = {"recursive"}  # , 'markdown', 'code'}

    def __init__(
        self,
        splitter: str = "recursive",
        splitter_config: Dict = DEFAULT_SPLITTERS_CONFIG,
        num_workers: int = 10,
    ) -> None:
        self.splitter_config = splitter_config
        self.num_workers = num_workers

        if splitter not in self.ALLOWED_SPLITTERS:
            raise ValueError(
                f"{splitter} is not a valid splitter."
                f" Choose from: {self.ALLOWED_SPLITTERS}"
            )
        self.splitter = self._get_splitter(splitter)

    def chunk_dataset(
        self,
        input_dir: str,
        save_chunks: bool = False,
        output_dir: Optional[str] = None,
        detailed_progress: bool = False,
        num_workers: Optional[int] = None,
    ) -> None:
        if num_workers is None:
            num_workers = self.num_workers

        if save_chunks and output_dir is None:
            raise ValueError(
                "Must provide an output directory when saving documents."
            )

        num_files = None
        if detailed_progress:
            num_files = len(list(get_files_from_dir(input_dir)))

        partial_func = partial(self.chunk_file, save_chunks, output_dir)

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
        self, save_chunks: bool, output_dir: str, file_path: str
    ) -> List[EnhancedDocument]:
        logging.debug("Chunking file: %s", file_path)
        raw_docs = load_docs_from_jsonl(file_path)
        chunked_docs = self.chunk_docs(raw_docs)
        if save_chunks:
            save_docs_to_file(chunked_docs, file_path, output_dir)
        logging.debug("Chunked file: %s", file_path)
        return chunked_docs

    def chunk_docs(
        self, raw_docs: List[EnhancedDocument]
    ) -> List[EnhancedDocument]:

        chunked_docs = self.splitter.split_documents(raw_docs)
        docs = [EnhancedDocument.remove_hashes(doc) for doc in chunked_docs]
        docs = [EnhancedDocument.from_document(doc) for doc in docs]
        s = set([doc.content_hash for doc in docs])
        return docs

    def _get_splitter(self, splitter: str) -> TextSplitter:
        if splitter == "recursive":
            kwargs = self.splitter_config["recursive"]
            return RecursiveCharacterTextSplitter(**kwargs)
        else:
            raise ValueError("Splitter not recognized: %s", splitter)
