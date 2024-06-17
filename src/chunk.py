from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_SPLITTERS_CONFIG = {
    "recursive": {
        "chunk_size": 500,
        "chunk_overlap": 20,
        "length_function": len,
        "is_separator_regex": False,
    },
}


class Chunker:
    ALLOWED_SPLITTERS = {"recursive"}  # , 'markdown', 'code'}

    def __init__(
        self,
        splitter="recursive",
        splitter_config=DEFAULT_SPLITTERS_CONFIG,
        documents_dir=None,
    ):
        self.splitter_config = splitter_config
        self.documents_dir = documents_dir

        if splitter not in self.ALLOWED_SPLITTERS:
            raise ValueError(
                f"{splitter} is not a valid splitter."
                f" Choose from: {self.ALLOWED_SPLITTERS}"
            )
        self.splitter = self._get_splitter(splitter)

    def chunk_docs(self, docs) -> List[Document]:
        return self.splitter.split_documents(docs)

    def _get_splitter(self, splitter):
        if splitter == "recursive":
            kwargs = self.splitter_config["recursive"]
            return RecursiveCharacterTextSplitter(**kwargs)
