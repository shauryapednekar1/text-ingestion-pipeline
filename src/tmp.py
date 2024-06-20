from enhanced_document import EnhancedDocument

# Example instantiation
page_content = "This is an example document content."
metadata = {
    "author": "John Doe",
    "title": "Example Document",
    "date": "2024-06-19",
}

enhanced_document = EnhancedDocument(
    page_content=page_content, metadata=metadata
)

print(enhanced_document.content_hash)

# from pydantic import BaseModel, root_validator
# import hashlib
# import json
# import uuid

# def _hash_string_to_uuid(input_string: str) -> uuid.UUID:
#     hash_value = hashlib.sha1(input_string.encode("utf-8")).hexdigest()
#     return uuid.uuid5(NAMESPACE_UUID, hash_value)

# def _hash_nested_dict_to_uuid(data: dict) -> uuid.UUID:
#     serialized_data = json.dumps(data, sort_keys=True)
#     hash_value = hashlib.sha1(serialized_data.encode("utf-8")).hexdigest()
#     return uuid.uuid5(NAMESPACE_UUID, hash_value)

# NAMESPACE_UUID = uuid.UUID('00000000-0000-0000-0000-000000001984')

# class Document(BaseModel):
#     page_content: str
#     metadata: dict

# class EnhancedDocument(Document):
#     document_hash: uuid.UUID = None
#     content_hash: uuid.UUID = None
#     metadata_hash: uuid.UUID = None

#     @root_validator(pre=True)
#     def calculate_hashes(cls, values):
#         print("Root validator called")
#         content = values.get('page_content', '')
#         metadata = values.get('metadata', {})

#         content_hash = _hash_string_to_uuid(content)
#         metadata_hash = _hash_nested_dict_to_uuid(metadata)
#         document_hash = _hash_string_to_uuid(str(content_hash) + str(metadata_hash))

#         values['content_hash'] = content_hash
#         values['metadata_hash'] = metadata_hash
#         values['document_hash'] = document_hash

#         return values

# # Example instantiation
# page_content = "This is an example document content."
# metadata = {
#     "author": "John Doe",
#     "title": "Example Document",
#     "date": "2024-06-19"
# }

# enhanced_document = EnhancedDocument(page_content=page_content, metadata=metadata)
# print("Content Hash Output:", enhanced_document.content_hash)
# print("Metadata Hash Output:", enhanced_document.metadata_hash)
# print("Document Hash Output:", enhanced_document.document_hash)
