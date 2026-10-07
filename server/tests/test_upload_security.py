from __future__ import annotations

import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

from services.security import validate_script_upload
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


class DocxUploadSecurityTests(unittest.TestCase):
    def _archive(self, root: str, extra: tuple[str, bytes] | None = None) -> Path:
        path = Path(root) / "script.docx"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", b"<Types/>")
            archive.writestr("word/document.xml", b"<document>hello</document>")
            if extra:
                archive.writestr(*extra)
        return path

    def test_accepts_bounded_docx_archive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="comic-agent-docx-") as root:
            validate_script_upload(
                self._archive(root),
                ".docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

    def test_rejects_archive_traversal_and_symlink_entries(self) -> None:
        with tempfile.TemporaryDirectory(prefix="comic-agent-docx-") as root:
            traversal = self._archive(root, ("../outside.xml", b"bad"))
            with self.assertRaisesRegex(ValueError, "路径"):
                validate_script_upload(traversal, ".docx", "application/octet-stream")

            symlink = Path(root) / "symlink.docx"
            with zipfile.ZipFile(symlink, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("[Content_Types].xml", b"<Types/>")
                archive.writestr("word/document.xml", b"<document/>")
                info = zipfile.ZipInfo("word/link.xml")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, "../../outside")
            with self.assertRaisesRegex(ValueError, "符号链接"):
                validate_script_upload(symlink, ".docx", "application/octet-stream")

    def test_rejects_encrypted_and_abnormal_ratio_entries(self) -> None:
        with tempfile.TemporaryDirectory(prefix="comic-agent-docx-") as root:
            encrypted = self._archive(root)
            content = bytearray(encrypted.read_bytes())
            local = content.index(b"PK\x03\x04")
            central = content.index(b"PK\x01\x02")
            content[local + 6 : local + 8] = (int.from_bytes(content[local + 6 : local + 8], "little") | 1).to_bytes(
                2, "little"
            )
            content[central + 8 : central + 10] = (
                int.from_bytes(content[central + 8 : central + 10], "little") | 1
            ).to_bytes(2, "little")
            encrypted.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "加密"):
                validate_script_upload(encrypted, ".docx", "application/octet-stream")

            bomb = Path(root) / "bomb.docx"
            with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("[Content_Types].xml", b"<Types/>")
                archive.writestr("word/document.xml", b"A" * (2 * 1024 * 1024))
            with self.assertRaisesRegex(ValueError, "压缩比"):
                validate_script_upload(bomb, ".docx", "application/octet-stream")


if __name__ == "__main__":
    unittest.main()
