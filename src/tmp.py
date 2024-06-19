import concurrent.futures
import os
import queue
from typing import List

from tqdm import tqdm

from doc_chunk import Chunker
from doc_embed import Embedder
from doc_load import Loader
from utils import get_files_from_dir, save_docs_to_file


def load_and_chunk_file(
    loader, chunker, file_path, save_docs, output_dir, preprocessed_queue
):
    # print("Loading and chunking file:", file_path)
    raw_docs = loader.file_to_docs(file_path)
    chunked_docs = chunker.chunk_docs(raw_docs)
    if save_docs:
        assert (
            output_dir is not None
        ), "Output directory must be specified if saving docs"
        raw_documents_dir = os.path.join(output_dir, "raw_documents")
        chunked_documents_dir = os.path.join(output_dir, "chunked_documents")
        save_docs_to_file(raw_docs, file_path, raw_documents_dir)
        save_docs_to_file(chunked_docs, file_path, chunked_documents_dir)
    preprocessed_queue.put(chunked_docs)
    # print("Done with file:", file_path)


def quick_embed_docs(embedder, docs):
    print("Embedding docs")
    embedder.embed_docs(docs)
    print("Embedded docs")


def quick_ingest(
    input_dir,
    loader=Loader(),
    chunker=Chunker(),
    embedder=Embedder(),
    batch_size=10000,
    num_workers=10,
    save_docs=False,
    output_dir=None,
):
    # Create a queue to hold preprocessed documents
    preprocessed_queue = queue.Queue(maxsize=1000)

    file_count = 612484 or len(list(get_files_from_dir(input_dir)))
    print(f"Found {file_count} files to process")

    # Executor for preprocessing
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
        # Submit preprocessing tasks
        futures = {
            executor.submit(
                load_and_chunk_file,
                loader,
                chunker,
                f,
                save_docs,
                output_dir,
                preprocessed_queue,
            ): f
            for f in get_files_from_dir(input_dir)
        }

        # Collect batches and process them
        batch = []
        count = 0
        with tqdm(
            total=file_count, desc="Embedding Files", unit="files"
        ) as pbar:
            while futures:
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                while not preprocessed_queue.empty():
                    preprocessed_docs = preprocessed_queue.get()
                    count += 1
                    batch.extend(preprocessed_docs)
                    if len(batch) >= batch_size:
                        quick_embed_docs(embedder, batch)
                        pbar.update(count)
                        count = 0
                        batch = []

                # Remove done futures from the list
                for fut in done:
                    futures.pop(fut)

            # Process any remaining documents in the batch
            if batch:
                quick_embed_docs(embedder, batch)

            # Wait for all futures to complete
            concurrent.futures.wait(
                futures.values(), return_when=concurrent.futures.ALL_COMPLETED
            )

        print("All files processed.")


# Example usage
if __name__ == "__main__":
    loader = Loader()
    chunker = Chunker()
    embedder = Embedder()
    quick_ingest("path_to_input_dir", loader, chunker, embedder)
