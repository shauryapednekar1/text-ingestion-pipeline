import json
import logging
from typing import Iterable, List

from langchain_core.documents import Document

from doc_chunk import Chunker
from doc_embed import Embedder
from doc_ingest import Ingester
from doc_load import Loader

# os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


class CustomLoader(Loader):
    def file_to_docs(self, file_path: str) -> List[Document]:
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
                    return [Document(page_content=text, metadata=metadata)]
                except Exception as e:
                    print(f"Failed to parse {fin}: {e}. Skipping for now")
                    return []
        else:
            return super().file_to_docs(file_path)


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
    #     input_dir="../unzipped/financial_dataset",
    #     # is_zipped=True,
    #     save_docs=True,
    #     output_dir="raw_documents",
    #     detailed_progress=False,
    #     num_workers=5,
    #     max_files=1000,
    # )

    # chunker = Chunker()
    # chunker.chunk_dataset(
    #     input_dir="raw_documents/unzipped/financial_dataset",
    #     save_chunks=True,
    #     output_dir="chunked_documents",
    #     detailed_progress=False,
    #     num_workers=5,
    # )

    # embedder = Embedder(vectorstore="Chroma")
    # embedder.embed_dataset(
    #     "chunked_documents/raw_documents/unzipped/financial_dataset",
    #     detailed_progress=True,
    # )

    # assert issubclass(CustomLoader, Loader)
    ingester = Ingester(loader=CustomLoader)
    ingester.ingest_dataset(
        input_dir="../unzipped/financial_dataset",
        save_docs=True,
        output_dir="test_output",
        detailed_progress=True,
        # max_files=10000,
    )

    # ingester = Ingester()
