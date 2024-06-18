import glob
import json
import multiprocessing
import os
from functools import partial
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from defaults import DEFAULT_SPLITTERS_CONFIG
from utils import get_files_from_dir, load_docs_from_jsonl, save_docs_to_file


class Chunker:
    ALLOWED_SPLITTERS = {"recursive"}  # , 'markdown', 'code'}

    def __init__(
        self,
        splitter="recursive",
        splitter_config=DEFAULT_SPLITTERS_CONFIG,
        num_workers=10,
    ):
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
        output_dir: str = None,
        detailed_progress: bool = False,
        num_workers: int = None,
    ):
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
            total=num_files, desc="Chunking files", unit=" files"
        ) as pbar:
            with multiprocessing.Pool(num_workers) as pool:
                for _ in pool.imap_unordered(
                    partial_func,
                    get_files_from_dir(input_dir),
                ):
                    pbar.update(1)

    def chunk_file(
        self, save_chunks: bool, output_dir: str, file_path: str
    ) -> List[Document]:
        raw_docs = load_docs_from_jsonl(file_path)
        chunked_docs = self.chunk_docs(raw_docs)
        if save_chunks:
            save_docs_to_file(chunked_docs, file_path, output_dir)
        return chunked_docs

    def chunk_docs(self, raw_docs: List[Document]):
        return self.splitter.split_documents(raw_docs)

    def _get_splitter(self, splitter):
        if splitter == "recursive":
            kwargs = self.splitter_config["recursive"]
            return RecursiveCharacterTextSplitter(**kwargs)
