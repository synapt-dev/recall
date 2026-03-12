"""Tests for synapt.server module."""

import os
import sys
import unittest
from unittest.mock import patch


class TestServerDevMode(unittest.TestCase):
    """Test --dev flag handling and helper functions."""

    def test_main_without_dev(self):
        """main() without --dev calls _serve()."""
        with patch("synapt.server._serve") as mock_serve:
            with patch.object(sys, "argv", ["synapt server"]):
                from synapt.server import main
                main()
                mock_serve.assert_called_once()

    def test_main_with_dev(self):
        """main() with --dev calls _dev_serve()."""
        with patch("synapt.server._dev_serve") as mock_dev:
            with patch.object(sys, "argv", ["synapt server", "--dev"]):
                from synapt.server import main
                main()
                mock_dev.assert_called_once()

    def test_dev_flag_removed_from_argv(self):
        """--dev flag should be removed from sys.argv."""
        captured_argv = []

        def capture_dev_serve():
            captured_argv.extend(sys.argv)

        with patch("synapt.server._dev_serve", side_effect=capture_dev_serve):
            with patch.object(sys, "argv", ["synapt server", "--dev"]):
                from synapt.server import main
                main()
                self.assertNotIn("--dev", captured_argv)

    def test_find_watch_paths(self):
        """_find_watch_paths returns at least the synapt package directory."""
        from synapt.server import _find_watch_paths
        paths = _find_watch_paths()
        self.assertTrue(len(paths) >= 1)
        # First path should be the synapt package dir
        self.assertTrue(
            paths[0].endswith("synapt"),
            f"Expected synapt dir, got {paths[0]}",
        )
        # Should be an actual directory
        self.assertTrue(os.path.isdir(paths[0]))


class TestNovelEntitiesInServer(unittest.TestCase):
    """Verify that _novel_entities from clustering is importable."""

    def test_import(self):
        from synapt.recall.clustering import _novel_entities
        self.assertIsNotNone(_novel_entities)


if __name__ == "__main__":
    unittest.main()
