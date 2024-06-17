import json
import logging
from chunk import Chunker
from typing import Iterable, List

import pudb
from langchain_core.documents import Document

from load import Loader

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
    # loader = CustomLoader(
    #     "data/financial_dataset.zip",
    #     is_zipped=True,
    #     # unzip=False,
    #     output_location="foo",
    #     num_workers=10,
    # )

    current_files = [
        "/Users/shauryapednekar/foo/src/text-ingestion-pipeline/foo/unzipped_data/financial_dataset.zip/2018_02_112b52537b67659ad3609a234388c50a/blogs_0000101.json.jsonl",
        "/Users/shauryapednekar/foo/src/text-ingestion-pipeline/foo/unzipped_data/financial_dataset.zip/2018_02_112b52537b67659ad3609a234388c50a/blogs_0000301.json.jsonl",
    ]

    chunker = Chunker()

    for file_path in current_files:
        docs = load_docs_from_jsonl(file_path=file_path)
        print(chunker.chunk_docs(docs))
    # loader = CustomLoader(
    #     "data/financial_dataset",
    #     is_zipped=False,
    #     output_location="foo_regular",
    #     num_workers=10,
    # )

    # loader.load_dataset()
