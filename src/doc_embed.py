import multiprocessing
from typing import List

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from defaults import DEFAULT_EMBEDDERS_CONFIG, DEFAULT_VECTORSTORES_CONFIG
from utils import get_files_from_dir, load_docs_from_jsonl


class Embedder:
    ALLOWED_EMBEDDERS = {"HuggingFace", "OpenAI", "custom"}
    ALLOWED_VECTORSTORES = {"Chroma", "custom"}

    def __init__(
        self,
        documents_dir=None,
        embedder="HuggingFace",
        embedders_config: dict = DEFAULT_EMBEDDERS_CONFIG,
        vectorstore="Chroma",
        vectorstore_config: dict = DEFAULT_VECTORSTORES_CONFIG,
        num_workers: int = 10,
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

        self.embedders_config = embedders_config
        self.embedder = self.get_embedder(embedder)
        self.vectorstore_config = vectorstore_config
        self.vectorstore_client = self.get_vectorstore(vectorstore)

        self.documents_dir = documents_dir
        self.num_workers = num_workers

    def embed_dataset(
        self,
        input_dir: str,
        detailed_progress: bool = False,
        num_workers: int = None,
    ) -> None:
        if num_workers is None:
            num_workers = self.num_workers

        num_files = None
        if detailed_progress:
            num_files = len(list(get_files_from_dir(input_dir)))

        with tqdm(
            total=num_files, desc="Embedding files", unit=" files"
        ) as pbar:
            with multiprocessing.Pool(num_workers) as pool:
                for _ in pool.imap_unordered(
                    self.embed_file,
                    get_files_from_dir(input_dir),
                ):
                    pbar.update(1)

    def embed_file(self, file_path: str) -> None:
        docs = load_docs_from_jsonl(file_path)
        self.embed_docs(docs)

    def embed_docs(self, docs: List[Document]):
        self.vectorstore.add_documents(docs)

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

    def get_vectorstore(self, name: str):
        if name == "custom":
            error_message = """
            "If using custom vectorstore, the Embedder.get_vectorstore() method
            must be overridden.
            """
            raise NotImplementedError(error_message)

        vectorstore_config = self.vectorstore_config[name]
        if name == "Chroma":
            vectorstore_config["embedding_function"] = self.embedder
            # chroma_client = chromadb.Client()
            # collection = chroma_client.get_or_create_collection(
            #     vectorstore_config["collection_name"]
            # )
            return Chroma(**vectorstore_config)
