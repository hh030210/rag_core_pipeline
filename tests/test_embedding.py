import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag_core.embedding import EmbeddingModel


class EmbeddingDeviceTests(unittest.TestCase):
    def test_auto_uses_cuda_when_available(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
        )
        with patch.dict("sys.modules", {"torch": fake_torch}):
            self.assertEqual(EmbeddingModel._resolve_device("auto"), "cuda")

    def test_auto_falls_back_to_cpu_without_cuda(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        with patch.dict("sys.modules", {"torch": fake_torch}):
            self.assertEqual(EmbeddingModel._resolve_device("auto"), "cpu")

    def test_explicit_device_is_preserved(self):
        self.assertEqual(EmbeddingModel._resolve_device("cpu"), "cpu")
        self.assertEqual(EmbeddingModel._resolve_device("cuda:0"), "cuda:0")


if __name__ == "__main__":
    unittest.main()
