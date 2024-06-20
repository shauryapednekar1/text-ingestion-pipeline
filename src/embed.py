import logging
import warnings
from itertools import islice
from typing import Dict, List, Optional, Tuple

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from defaults import DEFAULT_EMBEDDERS_CONFIG, DEFAULT_VECTORSTORES_CONFIG
from enhanced_document import EnhancedDocument
from utils import get_files_from_dir, load_docs_from_jsonl

# HACK(STP): Suppress warning about a deprecated arg. The fix has been merged.
# See https://github.com/huggingface/transformers/pull/30620/files.
warnings.filterwarnings(
    "ignore",
    message=".*`resume_download` is deprecated.*",
    category=FutureWarning,
)


class Embedder:
    """Creating embeddings for documents. Optionally store to a vectorstore.

    Potential functions to override if implementing a custom Embedder class:

    - `set_embedder()`: the logic for how the embedder is initialized.
    - `set_vectorstore()`: the logic for how the vectorstore is initialized.
    - `embed_docs()`: the logic for how documents are embedded.
    - `insert_embeddings()`: the logic for how embeddings are inserted into the
        vectorstore.
    """

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
        """
        Initializes an Embedder instance with specified configuration for
        embedding and vector storage.

        Args:
            documents_dir (str, optional): Directory containing the documents
                to embed.
            embedder (str): Type of embedder to use, options include
                'HuggingFace', 'OpenAI', or 'custom'.
            embedders_config (dict): Configuration settings for the embedder.
            vectorstore (Optional[VectorStore]): Type of vector store to use,
                options include 'FAISS', 'custom', or None.
            vectorstore_config (dict): Configuration settings for the vector
                store.

        Raises:
            ValueError: If the specified embedder or vectorstore is not valid.
        """
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
        self.set_embedder(embedder, config)
        if vectorstore is not None:
            self.set_vectorstore(vectorstore, vectorstore_config)

    def embed_and_insert_dataset(
        self,
        input_dir: str,
        detailed_progress: bool = False,
        num_workers: Optional[int] = None,
        batch_size: int = 1000,
    ) -> None:
        """
        Processes, embeds, and writes documents from the specified directory
        to the vectorstore in batches.

        Args:
            input_dir (str): Directory containing documents to embed.
            detailed_progress (bool): Whether to show detailed progress during
                embedding.
            num_workers (int, optional): Number of worker processes to use;
                defaults to the instance's configuration if not provided.
            batch_size (int): Number of files to process in each batch.

        Note:
            Uses multiprocessing to enhance performance.
        """
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

                self.embed_and_insert_files(file_chunk)
                pbar.update(len(file_chunk))

    def embed_and_insert_files(
        self, file_paths: List[str]
    ) -> Tuple[List[str], List[EnhancedDocument], List[List[float]]]:
        """
        Embeds documents from specified file paths and inserts them into the
        vector store.

        Args:
            file_paths (List[str]): File paths to embed and insert.
        Returns:
            tuple: A tuple containing lists of ids, docs, and their embeddings.
        """
        self._verify_vectorstore_client()
        all_docs = []
        all_embeddings = []
        for file in file_paths:
            curr_docs, curr_embeddings = self.embed_files(file_paths)
            # NOTE(STP): We're not calling self.embed_and_insert_docs() here
            # in order to allow us to batch embed multiple files.
            all_docs.extend(curr_docs)
            all_embeddings.extend(curr_embeddings)
        docs, ids, embeddings = self.insert_embeddings(
            all_docs, all_embeddings
        )
        return ids, docs, embeddings

    def embed_files(
        self, file_paths: List[str]
    ) -> Tuple[EnhancedDocument, List[List[float]]]:
        """
        Embeds a batch of files specified by their paths.

        Args:
            file_paths (List[str]): List of file paths to embed.

        Returns:
            tuple: A tuple containing lists of ids, docs, and their embeddings.
        """
        # NOTE(STP): We allow passing multiple files to take advantage of
        # batching benefits.
        logging.debug("Embedding files: %s", file_paths)
        docs = []
        for file_path in file_paths:
            docs.extend(load_docs_from_jsonl(file_path))
        embeddings = self.embed_docs(docs)
        logging.debug("Embedded files: %s", file_paths)
        return docs, embeddings

    def embed_and_insert_docs(
        self, docs: List[EnhancedDocument]
    ) -> Tuple[List[str], List[EnhancedDocument], List[List[float]]]:
        """
        Embeds documents and inserts their embeddings into the vectorstore,
        then returns the IDs, documents, and embeddings.

        Args:
            docs (List[EnhancedDocument]): Documents to embed and insert.

        Returns:
            tuple: A tuple containing lists of ids, docs, and their embeddings.


        Raises:
            ValueError: If the vectorstore instance is not set.
        """
        self._verify_vectorstore_client()
        embeddings = self.embed_docs(docs)
        ids, docs, embeddings = self.insert_embeddings(docs, embeddings)
        return ids, docs, embeddings

    def embed_docs(self, docs: List[EnhancedDocument]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents.

        Args:
            docs (List[EnhancedDocument]): Documents to embed.

        Returns:
            List[List[float]]: List of embeddings for each document.
        """
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

    def insert_embeddings(
        self, docs: List[EnhancedDocument], embeddings: List[List[float]]
    ) -> Tuple[List[str], List[EnhancedDocument], List[List[float]]]:
        """
        Inserts the embeddings of the provided documents into the vectorstore
        and ensures all documents are unique based on their content hash.

        Args:
            docs (List[EnhancedDocument]): Documents whose embeddings are to be
                inserted.
            embeddings (List[List[float]]): Embeddings corresponding to the
                documents.

        Returns:
            tuple: A tuple containing lists of ids, docs, and their embeddings.


        Raises:
            ValueError: If the vectorstore instance is not set.
        """
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
        self.vectorstore_instance.add_embeddings(
            text_embeddings=text_embeddings, ids=ids, metadatas=metadatas
        )
        logging.debug("Saved %d embedded docs to vectorstore docs", len(docs))
        return ids, docs, embeddings

    def save_vectorstore(self) -> None:
        """
        Saves the current state of the vector store locally.
        """
        if self.vectorstore_name == "FAISS":
            save_local_config = self.vectorstore_config["FAISS"][
                "save_local_config"
            ]
            if save_local_config["save_local"]:
                self.vectorstore_client.save_local(
                    save_local_config["folder_path"],
                    save_local_config["index_name"],
                )

    def set_embedder(self, name: str, config: Dict) -> Embeddings:
        """
        Configures and initializes the embedder based on specified name and
        configuration.

        Args:
            name (str): Name of the embedder to configure.
            config (dict): Configuration dictionary for the embedder.

        Raises:
            NotImplementedError: If a 'custom' embedder is specified but
                not implemented.
            ValueError: If embedder name is not recognized or none provided
                when required.
        """
        if name == "custom":
            error_message = """
            "If using custom embedder, the Embedder.set_embedder() method
            must be overridden.
            """
            raise NotImplementedError(error_message)

        embedder_config = config[name]
        if name == "OpenAI":
            return OpenAIEmbeddings(**embedder_config)
        elif name == "HuggingFace":
            return HuggingFaceEmbeddings(**embedder_config)
        else:
            raise ValueError("Embedding not recognized: %s", name)

    def set_vectorstore(self, name: str, config: Dict):
        """
        Configures and initializes the vector store based on specified name and
        configuration.

        Args:
            name (str): Name of the vector store to configure.
            config (dict): Configuration dictionary for the vector store.

        Raises:
            NotImplementedError: If a 'custom' vector store is specified but
                not implemented.
            ValueError: If vector store name is not recognized or none provided
                when required.
        """
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

        if name == "FAISS":
            if config["load_local"]:
                load_local_config = config["load_local_args"]
                load_local_config["embeddings"] = self.embedder
                vectorstore_instance = FAISS.load_local(**load_local_config)
                num_documents = len(vectorstore_instance.index_to_docstore_id)
                logging.debug(
                    "Total number of documents loaded from saved FAISS "
                    "vectorstore: %d",
                    num_documents,
                )

            else:
                config = config["init_args"]
                config["embedding_function"] = self.embedder
                config["index"] = faiss.IndexFlatL2(
                    self.embedder.client.get_sentence_embedding_dimension()
                )
                config["docstore"] = InMemoryDocstore()
                config["index_to_docstore_id"] = {}
                vectorstore_instance = FAISS(**config)

        self.vectorstore_instance = vectorstore_instance

    def _verify_vectorstore_client(self) -> None:
        """
        Verifies that the vectorstore instance is properly set up.

        Raises:
            ValueError: If the vectorstore instance is not set when required.
        """
        if self.vectorstore_instance is None:
            raise ValueError(
                "Vectorstore must be set when saving document embeddings."
            )
