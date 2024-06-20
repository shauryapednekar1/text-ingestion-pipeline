DEFAULT_AUTOLOADER_CONFIG = {
    "JSONLoader": {
        "required": {
            "jq_schema": None,  # "",
        },
        "optional": {
            "content_key": None,
            "is_content_key_jq_parsable": False,
            "metadata_func": None,
            "text_content": True,
            "json_lines": False,
        },
    },
    "CSVLoader": {
        "required": {},
        "optional": {
            "source_column": None,
            "metadata_columns": (),
            "csv_args": None,
            "encoding": None,
            "autodetect_encoding": False,
        },
    },
}

DEFAULT_SPLITTERS_CONFIG = {
    "recursive": {
        "chunk_size": 500,
        "chunk_overlap": 20,
        "length_function": len,
        "is_separator_regex": False,
    },
}

DEFAULT_EMBEDDERS_CONFIG = {
    "OpenAI": {"model": "text-embedding-3-large"},
    "HuggingFace": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "model_kwargs": {"device": "cuda"},
        "encode_kwargs": {"normalize_embeddings": False},
    },
}
DEFAULT_VECTORSTORES_CONFIG = {
    "FAISS": {},
}
