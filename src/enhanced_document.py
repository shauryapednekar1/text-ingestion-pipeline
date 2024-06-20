"""Module contains logic for indexing documents into vector stores."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any, Dict, Optional

from langchain_core.documents import Document
from pydantic.v1 import root_validator

# Magic UUID to use as a namespace for hashing.
# Used to try and generate a unique UUID for each document
# from hashing the document content and metadata.
NAMESPACE_UUID = uuid.UUID(int=1984)


def _hash_string_to_uuid(input_string: str) -> uuid.UUID:
    """Hashes a string and returns the corresponding UUID."""
    hash_value = hashlib.sha1(input_string.encode("utf-8")).hexdigest()
    return uuid.uuid5(NAMESPACE_UUID, hash_value)


def _hash_nested_dict_to_uuid(data: dict[Any, Any]) -> uuid.UUID:
    """Hashes a nested dictionary and returns the corresponding UUID."""
    serialized_data = json.dumps(data, sort_keys=True)
    hash_value = hashlib.sha1(serialized_data.encode("utf-8")).hexdigest()
    return uuid.uuid5(NAMESPACE_UUID, hash_value)


class EnhancedDocument(Document):
    """A hashed document with a unique ID."""

    source: str
    """The file path of the document."""
    document_hash: str
    """The hash of the document including content and metadata."""
    content_hash: str
    """The hash of the document content."""
    metadata_hash: str
    """The hash of the document metadata."""

    @root_validator(pre=True)
    def calculate_hashes_and_source(cls, values) -> Dict[str, Any]:
        """Calculate content, metadata and overall document hash.

        Also, update the metadata to include these hashes in there, in order
        to make it easier to query on them if required.
        """
        content = values.get("page_content")
        metadata = values.get("metadata")

        if "source" not in metadata:
            raise KeyError(
                "'source' not found in metadata. Each EnhancedDocument must "
                "have a source."
            )

        values["source"] = metadata["source"]

        forbidden_keys = ("document_hash", "content_hash", "metadata_hash")

        # HACK(STP): If we're reloading EnhancedDocuments from their JSON
        # representation, the forbidden keys will already be present. We
        # simply use them here.
        if all(key in metadata for key in forbidden_keys):
            for key in forbidden_keys:
                values[key] = metadata[key]

        else:
            for key in forbidden_keys:
                if key in metadata:
                    raise ValueError(
                        f"Metadata cannot contain key {key} as it "
                        f"is reserved for internal use."
                    )

            content_hash = str(_hash_string_to_uuid(content))

            try:
                metadata_hash = str(_hash_nested_dict_to_uuid(metadata))
            except Exception as e:
                raise ValueError(
                    f"Failed to hash metadata: {e}. "
                    f"Please use a dict that can be serialized using json."
                )

            document_hash = str(
                _hash_string_to_uuid(content_hash + metadata_hash)
            )

            # Update metadata with hashes
            hashes = {}
            hashes["content_hash"] = content_hash
            hashes["metadata_hash"] = metadata_hash
            hashes["document_hash"] = document_hash
            metadata.update(hashes)

            # Set hash values in the model
            # Ensure values are explicitly set
            values["content_hash"] = content_hash
            values["metadata_hash"] = metadata_hash
            values["document_hash"] = document_hash
        return values

    def to_document(self) -> Document:
        """Return a Document object."""
        return Document(
            page_content=self.page_content,
            metadata=self.metadata,
        )

    @classmethod
    def from_document(
        cls, document: Document, *, uid: Optional[str] = None
    ) -> EnhancedDocument:
        """Create a HashedDocument from a Document."""
        return cls(  # type: ignore[call-arg]
            uid=uid,  # type: ignore[arg-type]
            page_content=document.page_content,
            metadata=document.metadata,
        )

    @classmethod
    def remove_hashes(cls, document: Document) -> Document:
        forbidden_keys = ("document_hash", "content_hash", "metadata_hash")
        metadata = document.metadata
        for key in forbidden_keys:
            if key in metadata:
                del metadata[key]
        return document
