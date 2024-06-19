import multiprocessing
from itertools import islice
from multiprocessing import Queue
from typing import List, Optional

import chromadb
import faiss
from langchain.vectorstores import utils as chromautils
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from defaults import DEFAULT_EMBEDDERS_CONFIG, DEFAULT_VECTORSTORES_CONFIG
from utils import get_files_from_dir, load_docs_from_jsonl


class Embedder:
    ALLOWED_EMBEDDERS = {"HuggingFace", "OpenAI", "custom"}
    ALLOWED_VECTORSTORES = {"FAISS", "Chroma", "custom"}

    def __init__(
        self,
        documents_dir=None,
        embedder="HuggingFace",
        embedders_config: dict = DEFAULT_EMBEDDERS_CONFIG,
        vectorstore="FAISS",
        vectorstore_config: dict = DEFAULT_VECTORSTORES_CONFIG,
    ) -> None:
        if embedder not in self.ALLOWED_EMBEDDERS:
            raise ValueError(
                f"{embedder} is not a valid embedder."
                f" Choose from: {self.ALLOWED_EMBEDDERS}"
            )
        if vectorstore not in self.ALLOWED_VECTORSTORES:
            raise ValueError(
                f"{vectorstore} is not a valid vectorstore."
                f" Choose from: {self.ALLOWED_VECTORSTORES}"
            )

        self.embedder_name: str = embedder
        self.embedders_config = embedders_config

        # NOTE(STP): Storing the vectorstore name since it's needed for a
        # workaround for the Chroma vectorstore. See the `self.embed_docs()`
        # method for more.
        self.vectorstore_name: str = vectorstore
        self.vectorstore_config = vectorstore_config

        self.embedder = self.get_embedder(embedder)
        self.vectorstore_client = self.get_vectorstore(vectorstore)

        self.documents_dir = documents_dir

    def lazy_init_non_pickleable(self):
        self.embedder = self.get_embedder(self.embedder_name)
        self.vectorstore_client = self.get_vectorstore(self.vectorstore_name)

    def embed_dataset(
        self,
        input_dir: str,
        detailed_progress: bool = False,
        num_workers: Optional[int] = None,
        batch_size: int = 1000,
    ) -> None:
        num_files = None
        if detailed_progress:
            num_files = len(list(get_files_from_dir(input_dir)))

        with tqdm(
            total=num_files, desc="Embedding files", unit=" files"
        ) as pbar:
            while True:
                file_chunk = list(
                    islice(get_files_from_dir(input_dir), batch_size)
                )
                if not file_chunk:
                    break

                self.embed_files(file_chunk)
                pbar.update(len(file_chunk))

            # input_files_iterator = get_files_from_dir(input_dir)
            # for f in input_files_iterator:
            #     self.embed_file(f)
            #     pbar.update(1)
            # with multiprocessing.Pool(num_workers) as pool:
            #     for _ in pool.imap_unordered(
            #         self.embed_file,
            #         get_files_from_dir(input_dir),
            #     ):
            #         pbar.update(1)

    def embed_files(self, file_paths: List[str]) -> None:
        docs = []
        for file_path in file_paths:
            docs.extend(load_docs_from_jsonl(file_path))
        print("embedding docs")
        self.embed_docs(docs)
        print("done embedding docs")

    def embed_docs(self, docs: List[Document]) -> None:
        # HACK(STP): The Chroma vectorstore doesn't support some data types
        # for the document's metadata values. To work around this, we remove
        # any metadata that isn't supported. Maybe a better approach in the
        # future is stringifying the value instead.
        # See https://github.com/langchain-ai/langchain/issues/8556#issuecomment-1806835287 # noqa: E501
        # vectorstore_client = self.get_vectorstore(self.vectorstore_name)
        if self.vectorstore_name == "Chroma":
            docs = chromautils.filter_complex_metadata(docs)
        self.vectorstore_client.add_documents(docs)

    def get_embedder(self, name: str) -> Embeddings:
        if name == "custom":
            error_message = """
            "If using custom embedder, the Embedder.get_embedder() method
            must be overridden.
            """
            raise NotImplementedError(error_message)

        embedder_config = self.embedders_config[name]
        if name == "OpenAI":
            return OpenAIEmbeddings(**embedder_config)
        elif name == "HuggingFace":
            return HuggingFaceEmbeddings(**embedder_config)
        else:
            raise ValueError("Embedding not recognized: %s", name)

    def get_vectorstore(self, name: str):
        if name == "custom":
            error_message = """
            "If using custom vectorstore, the Embedder.get_vectorstore() method
            must be overridden.
            """
            raise NotImplementedError(error_message)

        config = self.vectorstore_config[name]
        config["embedding_function"] = self.embedder
        if name == "Chroma":
            return Chroma(**config)
        elif name == "FAISS":
            config["index"] = faiss.IndexFlatL2(
                self.embedder.client.get_sentence_embedding_dimension()
            )
            config["docstore"] = InMemoryDocstore()
            config["index_to_docstore_id"] = {}
            return FAISS(**config)
