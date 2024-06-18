import getpass
import json
import logging
import os
from typing import Iterable, List

import pudb
from langchain_core.documents import Document

from doc_chunk import Chunker
from doc_embed import Embedder
from doc_load import Loader

# os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


class CustomLoader(Loader):

    def file_to_docs(self, file_path: str) -> List[Document]:
        # TODO(STP): Convert pseudocode.
        file_extension = file_path.split(".")[-1]
        if file_extension == "json":
            with open(file_path) as fin:
                data = json.load(fin)
                text = data["text"]
                del data["text"]
                # TODO(STP): Add the filename to the metadata.
                metadata = data
                if "source" in metadata:
                    # HACK(STP): Since source is a reserved keyword for
                    # document metadata, we need to rename it here.
                    metadata["source_"] += metadata["source"]
                metadata["source"] = file_path
                return [Document(page_content=text, metadata=metadata)]
        else:
            super().file_to_docs(file_path)


# pudb.set_trace()


def load_docs_from_jsonl(file_path) -> Iterable[Document]:
    array = []
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array


if __name__ == "__main__":
    print("hello")
    logging.debug("Hello - logging")

    # loader = CustomLoader()
    # loader.load_dataset(
    #     dataset_dir="unzipped/datasets/financial_dataset",
    #     save_docs=True,
    #     output_dir="raw_documents",
    #     detailed_progress=False,
    # )

    # chunker = Chunker()
    # chunker.chunk_dataset(
    #     input_dir="raw_documents/unzipped/datasets/financial_dataset",
    #     save_chunks=True,
    #     output_dir="chunked_documents",
    #     detailed_progress=False,
    # )

    embedder = Embedder()
    embedder.embed_dataset("chunked_documents/financial_dataset")
