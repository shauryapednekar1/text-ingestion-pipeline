import unittest
from unittest.mock import patch

from src.defaults import DEFAULT_AUTOLOADER_CONFIG
from src.load import Loader


class TestSimpleLoader(unittest.TestCase):
    def test_initialization_with_default_config(self):
        """Test that the Loader initializes with default autoloader config."""
        loader = Loader()
        self.assertEqual(loader.autoloader_config, DEFAULT_AUTOLOADER_CONFIG)

    def test_valid_autoloaders_identification(self):
        """Test that valid autoloaders are identified correctly."""
        # Mock configuration for simplicity
        autoloader_config = {
            "JSONLoader": {"required": {"param1": "value1"}, "optional": {}},
            "CSVLoader": {
                "required": {
                    "param1": None
                },  # Should be identified as invalid
                "optional": {},
            },
            "XMLLoader": {"required": {"param1": "value1"}, "optional": {}},
        }
        loader = Loader(autoloader_config=autoloader_config)
        expected_autoloaders = {"JSONLoader", "XMLLoader"}
        valid_autoloaders = loader._get_valid_autoloaders()
        self.assertEqual(set(valid_autoloaders), expected_autoloaders)
