import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

from langchain_core.documents import Document

# from enhanced_document import EnhancedDocument
# from easy_ingest_text.load_text import EnhancedDocument, Loader
from easy_ingest_text.load_text import EnhancedDocument, Loader


class TestLoader(unittest.TestCase):
    """Test the Loader class."""

    def setUp(self):
        """Sets up test conditions for each test case.

        Initializes the Loader with a specific configuration to ensure the
        autoloaders are set up correctly.
        """
        self.config = {
            "JSONLoader": {
                "required": {"arg1": "value1"},
                "optional": {"arg2": "value2"},
            },
            "CSVLoader": {
                "required": {"arg3": "value3"},
                "optional": {"arg4": "value4"},
            },
            "otherLoader": {"required": {"arg5": None}, "optional": {}},
        }
        self.loader = Loader(autoloader_config=self.config)
        self.doc1 = Document(page_content="foo1", metadata={"source": "bar1"})
        self.enhanced_doc1 = EnhancedDocument.from_document(self.doc1)
        self.doc2 = Document(page_content="foo2", metadata={"source": "bar2"})
        self.enhanced_doc2 = EnhancedDocument.from_document(self.doc2)

    def test_init(self):
        """Tests the initialization of the Loader class.

        Ensures that the autoloaders are correctly initialized based on the
        provided configuration, particularly verifying that autoloaders with
        incomplete required configurations are not included.
        """
        self.assertEqual(self.loader.autoloader_config, self.config)
        self.assertIn("JSONLoader", self.loader.autoloaders)
        self.assertIn("CSVLoader", self.loader.autoloaders)
        self.assertNotIn("otherLoader", self.loader.autoloaders)

    @patch("easy_ingest_text.load_text.UnstructuredFileLoader")
    @patch("easy_ingest_text.load_text.JSONLoader")
    def test_file_to_docs(self, MockJSONLoader, MockUnstructuredLoader):
        """Tests the transformation of a file into EnhancedDocument objects.

        Checks that the correct loader is invoked based on the file extension
        and that the returned documents are correctly enhanced.
        """
        MockJSONLoader.return_value.load.return_value = [self.doc1]
        MockUnstructuredLoader.return_value.load.return_value = [self.doc2]
        docs = self.loader.file_to_docs("example.json")
        self.assertIsInstance(docs[0], EnhancedDocument)
        MockJSONLoader.assert_called_once()
        docs = self.loader.file_to_docs("example.txt")
        self.assertIsInstance(docs[0], EnhancedDocument)
        MockUnstructuredLoader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
