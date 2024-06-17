# Usage Guide

An example script can be found at `/scripts/example_ingestion.py`.
Note: Currently, this package isn't tested when passing relative paths as input. 
Thus, it's advised to store the dataset in the same directory as the script to
prevent undesired behavior.

---
See the [README](https://github.com/shauryapednekar1/text-ingestion-pipeline/tree/user/shaurya/add-load-framework) for more info on how the design is packaged.

---

### Loading a textual dataset into a vectorstore

This is the quickest way to ingest a dataset and store the results in a vectorstore. 
It uses a default loading, chunking, and embedding strategy.
This can be done in the following manner:
```python
from easy_ingest_text.ingest_text import Ingester

ingester = Ingester()
ingester.ingest_dataset(
    input_dir="financial_dataset.zip",
    is_zipped=True,
    save_intermediate_docs=True,
    output_dir="financial_documents_output",
    detailed_progress=True,
    chunk_batch_size=100,
    max_files=500,
)
```


If you want to specify different configuration options for the default `Loader`,
`Chunker`, or `Embedder`, you can do this by instantiating them individually
and passing them to the `Ingester`.

 Here's an example where we set the
configuration options for the provided `JSONLoader`, The `Chunker` and 
`Embedder` can be overridden in a similar manner.
```python
autoloader_config = {
    "JSONLoader": {
        "required": {
            "jq_schema": ".",  
        },
        "optional": {
            "content_key": None,
            "is_content_key_jq_parsable": False,
            "metadata_func": None,
            "text_content": True,
            "json_lines": False,
        },
    },
}
loader = Loader(autoloader_config)
ingester = Ingester(loader=loader)
...
```

### Using Custom Classes

If you want to include custom logic for how to load files, chunk documents,
or embed documents, you can subclass the relevant class and pass it to the
`Ingester`. Each class contains information on which methods should be
overriden.

Here's an example where a custom loader for loading json documents
of a specific format. 
```python
from easy_ingest_text.load_text import Loader
from easy_ingest_text.enhanced_document import EnhancedDocument

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
```

### Saving the vectorstore to disk

If using a custom vectorstore, it might offer its own functionality to persist
to disk. If using the default FAISS vectorstore, you can save it to disk by
setting the `save_local` flag in its config to `True`, as shown in the example
below.
```python
from easy_ingest_text.ingest_text import Ingester
from easy_ingest_text.embed_text import Embedder
import easy_ingest_text.defaults


vectorstore_config = defaults.DEFAULT_VECTORSTORES_CONFIG
vectorstore_config["FAISS"]["save_local_config"]["save_local"] = True
embedder = Embedder(vectorstore_config=vectorstore_config)
ingester = Ingester(embedder=embedder)
ingester.ingest_dataset(
    input_dir="financial_dataset",
    save_intermediate_docs=True,
    output_dir="financial_documents_output",
    detailed_progress=True,
    chunk_batch_size=100,
    max_files=500,
)
```