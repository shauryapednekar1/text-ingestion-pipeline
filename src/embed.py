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
    ALLOWED_VECTORSTORES = {None, "FAISS", "custom"}

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
        self.embedder = self.set_embedder(embedder)
        if vectorstore is not None:
            self.set_vectorstore(vectorstore, vectorstore_config)

    def set_vectorstore(self, name, config):
        assert name is not None and name in self.ALLOWED_VECTORSTORES

        if name == "custom":
            error_message = """
            "If using custom vectorstore, the Embedder.set_vectorstore() method
            must be overridden.
            """
            raise NotImplementedError(error_message)

        self.vectorstore_name = name
        self.vectorstore_config = config
        config = config[name]
        config["embedding_function"] = self.embedder
        if name == "FAISS":
            config["index"] = faiss.IndexFlatL2(
                self.embedder.client.get_sentence_embedding_dimension()
            )
            config["docstore"] = InMemoryDocstore()
            config["index_to_docstore_id"] = {}
            vectorstore_client = FAISS(**config)

        self.vectorstore_client = vectorstore_client

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
        # NOTE(STP): Allow passing multiple files to take advantage of batching
        # benefits.
        logging.debug("Embedding files: %s", file_paths)
        docs = []
        for file_path in file_paths:
            docs.extend(load_docs_from_jsonl(file_path))
        embeddings = self.embed_docs(docs)
        logging.debug("Embedded files: %s", file_paths)
        return docs, embeddings

    def embed_and_save_files(self, file_paths):
        self._verify_vectorstore_client()
        all_docs = []
        all_embeddings = []
        for file in file_paths:
            docs, embeddings = self.embed_files(file_paths)
            # NOTE(STP): We're not calling self.embed_and_save_docs() here
            # in order to allow us to batch embed multiple files.
            all_docs.extend(docs)
            all_embeddings.extend(embeddings)
        self.save_embeddings(all_docs, all_embeddings)

    def embed_and_save_docs(self, docs: List[EnhancedDocument]) -> None:
        self._verify_vectorstore_client()
        embeddings = self.embed_docs(docs)
        self.save_embeddings(docs, embeddings)

    def save_embeddings(self, docs, embeddings):
        self._verify_vectorstore_client()
        logging.debug("Saving %d embedded docs to vectorstore docs", len(docs))
        ids = [doc.document_hash for doc in docs]
        if len(ids) != len(set(ids)):
            # TODO(STP): Improve space efficiency here.
            unique_docs = []
            unique_embeddings = []
            unique_ids = []
            seen = set()
            for i, curr_id in enumerate(ids):
                if curr_id in seen:
                    logging.debug(
                        "Found multiple documents from %s with the "
                        " same content hash. '%s...'",
                        docs[i].metadata["source"],
                        docs[i].page_content[:30],
                    )
                else:
                    unique_ids.append(curr_id)
                    unique_docs.append(docs[i])
                    unique_embeddings.append(embeddings[i])
                    seen.add(curr_id)

            docs = unique_docs
            ids = unique_ids
            embeddings = unique_embeddings

        texts = [doc.page_content for doc in docs]
        text_embeddings = zip(texts, embeddings)
        metadatas = [doc.metadata for doc in docs]
        self.vectorstore_client.add_embeddings(
            text_embeddings=text_embeddings, ids=ids, metadatas=metadatas
        )
        logging.debug("Saved %d embedded docs to vectorstore docs", len(docs))
        return ids

    def embed_docs(self, docs: List[EnhancedDocument]) -> List[List[float]]:
        # NOTE(STP): This ignores metadata. If we want to include metadata in
        # the embedding, we would need to combine it with the page content
        # and stringify it in some manner.
        # TODO(STP): We might want to batch embed documents here if the number
        # of documents exceed a certain threshold. Would need to look more into
        # if and when that would be useful.
        logging.debug("Embedding %d docs", len(docs))
        page_contents = [doc.page_content for doc in docs]
        embeddings = self.embedder.embed_documents(page_contents)
        logging.debug("Embedded %d docs", len(docs))
        return embeddings

    def set_embedder(self, name: str) -> Embeddings:
        if name == "custom":
            error_message = """
            "If using custom embedder, the Embedder.set_embedder() method
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

    def _verify_vectorstore_client(self):
        if self.vectorstore_client is None:
            raise ValueError(
                "Vectorstore must be set when saving document embeddings."
            )
