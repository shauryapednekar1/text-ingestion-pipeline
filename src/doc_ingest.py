import os
import pickle
from functools import partial
from itertools import islice
from multiprocessing import Pool
from typing import List, Optional

from langchain_core.documents import Document
from tqdm import tqdm

from doc_chunk import Chunker
from doc_embed import Embedder
from doc_load import Loader
from utils import get_files_from_dir, save_docs_to_file, unzip_recursively


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
        batch_size: int = 50000,
    ):
        if is_zipped:
            self.loader.unzip_dataset(input_dir, unzip_dir)
        else:
            directory = input_dir

        num_files = None
        if detailed_progress:
            num_files = max_files or len(list(get_files_from_dir(directory)))

        partial_func = partial(self.ingest_file, save_docs, output_dir)
        with tqdm(
            total=num_files, desc="Ingesting files", unit=" files"
        ) as pbar:
            batch = []
            prev_counter = 0

            # with Pool(10) as pool:
            #     while True:
            #         file_chunk = list(
            #             islice(get_files_from_dir(directory), batch_size)
            #         )
            #         if not file_chunk:
            #             break

            #         processed_docs = pool.map(partial_func, file_chunk)
            #         self.embedder.embed_docs(processed_docs)
            #         pbar.update(len(file_chunk))

            for i, file_path in enumerate(get_files_from_dir(directory)):

                docs = self.ingest_file(
                    save_docs=save_docs,
                    output_dir=output_dir,
                    file_path=file_path,
                )
                batch.extend(docs)

                if len(batch) > batch_size:
                    print("embedding docs")
                    self.embedder.embed_docs(docs)
                    print("embedded docs")
                    batch = []

                    pbar.update(i - prev_counter)
                    prev_counter = i

                if max_files is not None and i >= max_files:
                    break

        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(self.embedder.vectorstore_client, f)

        self.embedder.vectorstore_client.save_local("faiss_index")


            # with multiprocessing.get_context("spawn").Pool(
            #     num_workers,
            # ) as pool:
            # for i, _ in enumerate(
            #     pool.imap_unordered(
            #         partial_func,
            #         get_files_from_dir(directory),
            #     )
            # ):
            #     pbar.update(1)
            #     if max_files is not None and i + 1 >= max_files:
            #         break

    def ingest_file(
        self,
        save_docs: bool,
        output_dir: Optional[str],
        file_path: str,
    ) -> List[Document]:
        raw_docs = self.loader.file_to_docs(self.loader, file_path)
        chunked_docs = self.chunker.chunk_docs(raw_docs)
        # self.embedder.embed_docs(chunked_docs)
        if save_docs:
            assert output_dir is not None
            raw_documents_dir = os.path.join(output_dir, "raw_documents")
            chunked_documents_dir = os.path.join(
                output_dir, "chunked_documents"
            )
            save_docs_to_file(raw_docs, file_path, raw_documents_dir)
            save_docs_to_file(chunked_docs, file_path, chunked_documents_dir)

        return chunked_docs
