"""Example script illustrating usage of the package."""

import json
import logging
from typing import List

import easy_ingest_text.defaults
from easy_ingest_text.embed_text import Embedder
from easy_ingest_text.enhanced_document import EnhancedDocument
from easy_ingest_text.ingest_text import Ingester
from easy_ingest_text.load_text import Loader

# Configure the logging
logging.basicConfig(
    format="%(asctime)s -  %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


class CustomLoader(Loader):
    """Custom logic for converting files to EnhancedDocuments."""

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

    vectorstore_config = easy_ingest_text.defaults.DEFAULT_VECTORSTORES_CONFIG
    vectorstore_config["FAISS"]["save_local_config"]["save_local"] = True
    embedder = Embedder(vectorstore_config=vectorstore_config)
    ingester = Ingester(loader=CustomLoader(), embedder=embedder)
    ingester.ingest_dataset(
        input_dir="financial_dataset.zip",
        is_zipped=True,
        save_intermediate_docs=True,
        output_dir="test_output_financial_dataset",
        detailed_progress=True,
        embed_batch_size=1000,
        chunk_batch_size=100,
        max_files=500,
    )
