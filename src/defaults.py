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
    # NOTE(STP): Currently, the LangChain FAISS vectorstore does not handle
    # upserts well. If you try to insert the same value twice, it will fail.
    # To handle this, we would probably need to switch to using LangChain's
    # Indexing API, or come up with some custom logic for this.
    # See: https://github.com/langchain-ai/langchain/issues/3896 and
    # https://python.langchain.com/v0.1/docs/modules/data_connection/indexing/
    "FAISS": {
        "init_args": {},
        "load_local": True,
        "load_local_args": {
            "folder_path": "faiss_index",
            "index_name": "index",
            # NOTE(STP): This must be set to true when loading from local
            # pickled file.
            "allow_dangerous_deserialization": True,
        },
        "save_local_config": {
            "save_local": False,
            "folder_path": "faiss_index",
            "index_name": "index",
        },
    },
}
