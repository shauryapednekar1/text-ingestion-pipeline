# Usage Guide

See the home page for more info on how the design is packaged.

### Loading a textual dataset into a vectorstore

This is the quickest way to ingest a dataset and store the results in a vectorstore. 
It uses a default loading, chunking, and embedding strategy.
This can be done in the following manner:
```python
import ...

ingester = Ingester()
ingester.ingest_dataset(
    input_dir="financial_dataset",
    save_intermediate_docs=True,
    output_dir="test_output",
    detailed_progress=True,
    batch_size=100,
    max_files=1000,
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