"""Test how the source viewer identifies binary files."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import STATview as statview


class BinarySourceTests(unittest.TestCase):
    def test_binary_paths_use_extensions_and_numeric_library_versions(self):
        """Recognize libraries while allowing source names that contain .so."""
        cases = (
            ("libfoo.a", True),
            ("module.o", True),
            ("module.pyc", True),
            ("libfoo.so", True),
            ("libfoo.so.1.2", True),
            ("libfoo.SO", True),
            ("data.bin", False),
            ("module.so.py", False),
            ("solver.source.py", False),
        )

        for filename, expected in cases:
            with self.subTest(filename=filename):
                self.assertEqual(statview.is_binary_source_path(filename), expected)

    def test_content_probe_accepts_utf8_split_at_its_boundary(self):
        """Probe real files, including a UTF-8 character split at byte 4096."""
        cases = (
            ("empty", b"", False),
            ("text", b"int main() {}", False),
            ("elf", b"\x7fELF\0", True),
            ("invalid_utf8", b"\xff", True),
            ("truncated_utf8", b"\xc3", True),
            ("utf8_boundary", b"a" * 4095 + "é".encode("utf-8"), False),
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "source"

            for name, data, expected in cases:
                with self.subTest(content=name):
                    path.write_bytes(data)

                    self.assertEqual(statview.is_binary_file(str(path)), expected)


if __name__ == '__main__':
    unittest.main()
