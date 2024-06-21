import unittest
from unittest.mock import MagicMock, patch

from easy_ingest_text.chunk_text import Chunker, EnhancedDocument


class TestChunker(unittest.TestCase):
    """Test the Chunker class."""

    def setUp(self):
        """Set up common properties for tests."""
        self.splitter_config = {
            "recursive": {
                "chunk_size": 500,
                "chunk_overlap": 20,
                "length_function": len,
                "is_separator_regex": False,
            }
        }
        self.input_dir = "/path/to/input"
        self.output_dir = "/path/to/output"
        self.enhanced_doc1 = EnhancedDocument(
            page_content="foo1", metadata={"source": "bar.txt"}
        )
        self.enhanced_doc2 = EnhancedDocument(
            page_content="foo2", metadata={"source": "bar.json"}
        )
        self.enhanced_doc3 = EnhancedDocument(
            page_content="foo3", metadata={"source": "bar.csv"}
        )

    def test_init_valid_splitter(self):
        """Test initialization with valid splitter."""
        chunker = Chunker("recursive", self.splitter_config, num_workers=5)
        self.assertEqual(chunker.num_workers, 5)
        self.assertIsNotNone(chunker.splitter)

    def test_init_invalid_splitter(self):
        """Test initialization with invalid splitter raises ValueError."""
        with self.assertRaises(ValueError):
            Chunker("invalid_splitter", self.splitter_config)

    @patch("easy_ingest_text.chunk_text.save_docs_to_file")
    @patch("easy_ingest_text.chunk_text.logging")
    def test_chunk_file(self, mock_logging, mock_save_docs_to_file):
        """Test the chunking of a single file."""
        documents = [self.enhanced_doc1]
        chunked_documents = [self.enhanced_doc2, self.enhanced_doc3]
        with patch(
            "easy_ingest_text.chunk_text.load_docs_from_jsonl",
            return_value=documents,
        ), patch.object(Chunker, "chunk_docs", return_value=chunked_documents):
            chunker = Chunker("recursive", self.splitter_config, num_workers=5)
            result = chunker.chunk_file(True, self.output_dir, "file1.jsonl")

            mock_save_docs_to_file.assert_called_once_with(
                chunked_documents, "file1.jsonl", self.output_dir
            )
            self.assertEqual(result, chunked_documents)
            mock_logging.debug.assert_called()

    def test_chunk_docs(self):
        """Test the splitting of documents into chunks."""
        raw_docs = [self.enhanced_doc1]
        expected_chunks = [self.enhanced_doc2, self.enhanced_doc3]
        with patch.object(Chunker, "get_splitter") as mockget_splitter:
            splitter = MagicMock()
            splitter.split_documents.return_value = expected_chunks
            mockget_splitter.return_value = splitter

            chunker = Chunker("recursive", self.splitter_config)
            result = chunker.chunk_docs(raw_docs)

            self.assertEqual(result, expected_chunks)
            splitter.split_documents.assert_called_once_with(raw_docs)


if __name__ == "__main__":
    unittest.main()
