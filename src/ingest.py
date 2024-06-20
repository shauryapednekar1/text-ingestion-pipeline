import logging
import multiprocessing as mp
import os
import pickle
from chunk import Chunker
from functools import partial
from itertools import chain, islice
from multiprocessing import Pool
from typing import List, Optional

from tqdm import tqdm

from embed import Embedder
from enhanced_document import EnhancedDocument
from load import Loader
from utils import get_files_from_dir, save_docs_to_file, unzip_recursively

# def quick_ingest():
#     queue = mp.Queue()


class Ingester:
    ALLOWED_LEVELS = {"load", "chunk", "embed"}

    def __init__(
        self,
        loader: Loader = Loader(),
        chunker: Chunker = Chunker(),
        embedder: Embedder = Embedder(),
    ) -> None:
        self.loader: Loader = loader
        self.chunker: Chunker = chunker
        self.embedder: Embedder = embedder

    def ingest_dataset(
        self,
        input_dir: str,
        level: str = "load",
        is_zipped: bool = False,
        unzip_dir: str = "unzipped",
        save_docs: bool = False,
        output_dir: Optional[str] = None,
        num_workers: int = 10,
        max_files: Optional[int] = None,
        detailed_progress: bool = False,
        batch_size: int = 100,  # in number of files
    ):
        if is_zipped:
            directory = self.loader.unzip_dataset(input_dir, unzip_dir)
        else:
            directory = input_dir

        num_files = None
        if detailed_progress:
            num_files = (
                max_files or 612484 or len(list(get_files_from_dir(directory)))
            )

        with tqdm(
            total=num_files, desc="Ingesting files", unit="files", smoothing=0
        ) as pbar:
            batched_docs = []
            prev_counter = 0
            for i, file_path in enumerate(get_files_from_dir(directory)):

                docs = self.load_and_chunk_file(
                    save_docs=save_docs,
                    output_dir=output_dir,
                    file_path=file_path,
                )
                batched_docs.extend(docs)

                if i - prev_counter >= batch_size:
                    self.embedder.embed_and_save_docs(batched_docs)
                    batched_docs = []

                    pbar.update(i - prev_counter)
                    prev_counter = i

                if max_files is not None and i >= max_files:
                    break

            if batched_docs:
                self.embedder.embed_and_save_docs(batched_docs)

        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(self.embedder.vectorstore_client, f)

        self.embedder.vectorstore_client.save_local("faiss_index")

    def load_and_chunk_file(
        self,
        save_docs: bool,
        output_dir: Optional[str],
        file_path: str,
    ) -> List[EnhancedDocument]:
        logging.debug("Loading and chunking: %s", file_path)
        raw_docs = self.loader.file_to_docs(file_path)
        chunked_docs = self.chunker.chunk_docs(raw_docs)
        if save_docs:
            assert output_dir is not None
            raw_documents_dir = os.path.join(output_dir, "raw_documents")
            chunked_documents_dir = os.path.join(
                output_dir, "chunked_documents"
            )
            save_docs_to_file(raw_docs, file_path, raw_documents_dir)
            save_docs_to_file(chunked_docs, file_path, chunked_documents_dir)

        logging.debug("Loaded and chunked: %s", file_path)
        return chunked_docs
