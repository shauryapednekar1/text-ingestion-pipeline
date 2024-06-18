from typing import Optional

from doc_chunk import Chunker
from doc_embed import Embedder
from doc_load import Loader


class Ingest:
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
        unzip: bool = True,
        unzip_dir: str = "unzipped",
        save_docs: bool = False,
        output_dir: Optional[str] = None,
        num_workers: Optional[int] = None,
        max_files: Optional[int] = None,
    ):
        pass

    def ingest_file(self, file_path: str, level: str):
        pass
