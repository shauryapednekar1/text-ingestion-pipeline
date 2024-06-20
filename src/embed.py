import logging
import multiprocessing
from collections import defaultdict
from itertools import islice
from multiprocessing import Queue
from typing import List, Optional

import faiss
from langchain.vectorstores import utils as chromautils
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from defaults import DEFAULT_EMBEDDERS_CONFIG, DEFAULT_VECTORSTORES_CONFIG
from enhanced_document import EnhancedDocument
from utils import get_files_from_dir, load_docs_from_jsonl


class Embedder:
    ALLOWED_EMBEDDERS = {"HuggingFace", "OpenAI", "custom"}
    ALLOWED_VECTORSTORES = {None, "FAISS", "Chroma", "custom"}

    def __init__(
        self,
        documents_dir=None,
        embedder="HuggingFace",
        embedders_config: dict = DEFAULT_EMBEDDERS_CONFIG,
        vectorstore: Optional[VectorStore] = "FAISS",
        vectorstore_config: dict = DEFAULT_VECTORSTORES_CONFIG,
    ) -> None:

        self.documents_dir = documents_dir
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
        self.embedder = self.get_embedder(embedder)

        # NOTE(STP): Storing the vectorstore name since it's needed for a
        # workaround for the Chroma vectorstore. See the `self.embed_docs()`
        # method for more.
        if vectorstore is not None:
            self.vectorstore_name: str = vectorstore
            self.vectorstore_config = vectorstore_config

            self.vectorstore_client = self.get_vectorstore(vectorstore)

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
            total=num_files, desc="Embedding files", unit=" files", smoothing=0
        ) as pbar:
            while True:
                file_chunk = list(
                    islice(get_files_from_dir(input_dir), batch_size)
                )
                if not file_chunk:
                    break

                self.embed_files(file_chunk)
                pbar.update(len(file_chunk))

    def embed_files(self, file_paths: List[str]) -> None:
        # TODO(STP): Explain why this takes in multiple files.
        logging.debug("Embedding files: %s", file_paths)
        docs = []
        for file_path in file_paths:
            docs.extend(load_docs_from_jsonl(file_path))
        self.embed_docs(docs)
        logging.debug("Embedded files: %s", file_paths)

    def embed_docs(self, docs: List[EnhancedDocument]) -> None:
        # TODO(STP): We might want to batch embed documents here if the number
        # of documents exceed a certain threshold. Would need to look more into
        # if and when that would be useful.
        logging.debug("Embedding %d docs", len(docs))

        # HACK(STP): The Chroma vectorstore doesn't support some data types
        # for the document's metadata values. To work around this, we remove
        # any metadata that isn't supported. Maybe a better approach in the
        # future is stringifying the value instead.
        # See https://github.com/langchain-ai/langchain/issues/8556#issuecomment-1806835287 # noqa: E501
        # vectorstore_client = self.get_vectorstore(self.vectorstore_name)
        if self.vectorstore_name == "Chroma":
            docs = chromautils.filter_complex_metadata(docs)

        ids = [doc.document_hash for doc in docs]
        if len(ids) != len(set(ids)):
            docs = self.filter_duplicates(docs)
            ids = [doc.document_hash for doc in docs]

        self.vectorstore_client.add_documents(docs, ids=ids)
        logging.debug("Embedded %d docs", len(docs))

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

    def filter_duplicates(self, documents):
        hash_freq = defaultdict(list)
        for doc in documents:
            hash_freq[doc.content_hash].append(doc)

        result = []
        for content_hash, docs in hash_freq.items():
            if len(docs) > 1:
                logging.debug(
                    "Found multiple documents from %s with the same content "
                    "hash. '%s...'",
                    docs[0].metadata["source"],
                    docs[0].page_content[:30],
                )
            result.append(docs[0])

        return result
