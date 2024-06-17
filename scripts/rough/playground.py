import json
import logging
import pickle
import sys
from typing import Iterable, List

import pudb

from easy_ingest_text.chunk_text import Chunker
from easy_ingest_text.embed_text import Embedder
from easy_ingest_text.enhanced_document import EnhancedDocument
from easy_ingest_text.ingest_text import Ingester
from easy_ingest_text.load_text import Loader

# Configure the logging
logging.basicConfig(
    format="%(asctime)s -  %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class CustomLoader(Loader):
    def file_to_docs(self, file_path: str) -> List[EnhancedDocument]:
        file_extension = file_path.split(".")[-1]
        if file_extension == "json":
            with open(file_path) as fin:
                try:
                    data = json.load(fin)
                    text = data["text"]
                    # TODO(STP): Add the filename to the metadata.
                    metadata = {}
                    for key in {
                        "title",
                        "url",
                        "site_full",
                        "language",
                        "published",
                    }:
                        if key in data:
                            metadata[key] = data[key]
                    if "source" in metadata:
                        # HACK(STP): Since source is a reserved keyword for
                        # document metadata, we need to rename it here.
                        metadata["source_"] += metadata["source"]
                    metadata["source"] = file_path
                    return [
                        EnhancedDocument(page_content=text, metadata=metadata)
                    ]
                except Exception as e:
                    print(f"Failed to parse {fin}: {e}. Skipping for now")
                    return []
        else:
            return super().file_to_docs(file_path)


if __name__ == "__main__":
    print("hello")
    logging.debug("Hello - logging")

    # loader = CustomLoader()
    # loader.load_dataset(
    #     # input_dir="../unzipped/financial_dataset",
    #     input_dir="../financial_dataset.zip",
    #     is_zipped=True,
    #     save_docs=True,
    #     output_dir="raw_documents",
    #     detailed_progress=True,
    #     num_workers=10,
    #     # max_files=1000,
    # )

    # chunker = Chunker()
    # chunker.chunk_dataset(
    #     input_dir="raw_documents/financial_dataset",
    #     save_chunks=True,
    #     output_dir="chunked_documents",
    #     detailed_progress=False,
    #     num_workers=15,
    # )

    # embedder = Embedder(vectorstore="FAISS")
    # embedder.embed_and_insert_dataset(
    #     "chunked_documents/raw_documents/financial_dataset",
    #     detailed_progress=True,
    #     chunk_batch_size=100,
    # )

    ingester = Ingester(loader=CustomLoader())
    ingester.ingest_dataset(
        input_dir="unzipped/financial_dataset",
        # input_dir="football_dataset.zip",
        # is_zipped=True,
        # save_docs=True,
        output_dir="test_output",
        detailed_progress=True,
        chunk_batch_size=100,
        max_files=1000,
    )
    ingester.embedder.vectorstore_instance.save_local("faiss_index")
