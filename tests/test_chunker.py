import unittest
from unittest.mock import MagicMock, patch

from src.chunk import Chunker
from src.enhanced_document import EnhancedDocument


class TestChunker(unittest.TestCase):
    def setUp(self):
        """Setup for all tests."""
        self.chunker = Chunker()
        # Mocking an EnhancedDocument with initial content and metadata
        self.sample_docs = [
            EnhancedDocument(
                page_content="Example content", metadata={"source": "initial"}
            )
        ]

    @patch("chunker.RecursiveCharacterTextSplitter")
    def test_chunk_document_processing(self, mock_splitter):
        """Test processing of documents through chunk_docs."""
        mock_splitter_instance = mock_splitter.return_value
        mock_splitter_instance.split_documents.return_value = self.sample_docs
        chunked_docs = self.chunker.chunk_docs(self.sample_docs)
        self.assertTrue(
            all(isinstance(doc, EnhancedDocument) for doc in chunked_docs)
        )
        self.assertTrue(all("source" in doc.metadata for doc in chunked_docs))

    def test_enhanced_document_initialization_in_chunking(self):
        """Test that EnhancedDocument is correctly initialized during chunking."""
        with patch("chunker.EnhancedDocument.from_document") as mock_from_doc:
            mock_doc = MagicMock(spec=EnhancedDocument)
            mock_doc.page_content = "Chunked content"
            mock_doc.metadata = {"source": "derived"}
            mock_from_doc.return_value = mock_doc

            self.chunker.chunk_docs(self.sample_docs)
            mock_from_doc.assert_called_with(mock_doc)

    def test_error_handling_for_unspecified_output_directory(self):
        """Test error handling when output directory is not provided but saving is required."""
        with self.assertRaises(ValueError):
            self.chunker.chunk_file(
                save_chunks=True, output_dir=None, file_path="dummy_path"
            )


if __name__ == "__main__":
    unittest.main()
