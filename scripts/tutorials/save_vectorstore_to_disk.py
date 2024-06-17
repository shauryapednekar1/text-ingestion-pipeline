import easy_ingest_text.defaults
from easy_ingest_text.embed_text import Embedder
from easy_ingest_text.ingest_text import Ingester

vectorstore_config = easy_ingest_text.defaults.DEFAULT_VECTORSTORES_CONFIG
vectorstore_config["FAISS"]["save_local_config"]["save_local"] = True
embedder = Embedder(vectorstore_config=vectorstore_config)
ingester = Ingester(embedder=embedder)
ingester.ingest_dataset(
    input_dir="unzipped/financial_dataset",
    save_intermediate_docs=True,
    output_dir="test_output",
    detailed_progress=True,
    chunk_batch_size=100,
    max_files=1000,
)
