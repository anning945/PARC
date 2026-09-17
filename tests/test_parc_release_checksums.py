import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from parc_release_checksums import (  # noqa: E402
    ARTIFACT_MANIFEST,
    MANIFEST,
    check_manifest,
    digest,
    read_manifest,
    release_path,
    update_manifest,
)


class ReleaseChecksumTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="parc release test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.source = self.root / "README.md"
        self.source.write_text("synthetic release\n", encoding="utf-8")
        self.artifact = self.root / "artifacts/parc_selector/model.joblib"
        self.artifact.parent.mkdir(parents=True)
        self.artifact.write_bytes(b"synthetic artifact, never unpickled")
        relative = self.artifact.relative_to(self.root).as_posix()
        (self.root / ARTIFACT_MANIFEST).write_text(
            f"{digest(self.artifact)}  {relative}\n", encoding="utf-8"
        )
        self.git("add", "README.md", "artifacts")
        update_manifest(self.root)

    def git(self, *args):
        subprocess.run(
            ["git", *args], cwd=self.root, check=True, capture_output=True
        )

    def test_manifest_uses_only_tracked_release_files(self):
        before = (self.root / MANIFEST).read_bytes()
        (self.root / ".git/private-test").write_text("git metadata")
        (self.root / "outputs").mkdir()
        (self.root / "outputs/result.json").write_text("{}")
        (self.root / "untracked.txt").write_text("local notes")
        update_manifest(self.root)
        self.assertEqual(before, (self.root / MANIFEST).read_bytes())
        self.assertEqual(check_manifest(self.root, MANIFEST), 3)

    def test_source_archive_without_git_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)
            for relative in [MANIFEST, *read_manifest(self.root, MANIFEST)]:
                destination = archive / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.root / relative, destination)
            self.assertFalse((archive / ".git").exists())
            self.assertEqual(check_manifest(archive, MANIFEST), 3)
            self.assertEqual(check_manifest(archive, ARTIFACT_MANIFEST), 1)

    def test_changed_source_is_rejected(self):
        self.source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch: README.md"):
            check_manifest(self.root, MANIFEST)

    def test_missing_source_is_rejected(self):
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            check_manifest(self.root, MANIFEST)

    def test_new_tracked_source_requires_manifest_update(self):
        (self.root / "new.py").write_text("pass\n")
        self.git("add", "new.py")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            check_manifest(self.root, MANIFEST)
        update_manifest(self.root)
        self.assertEqual(check_manifest(self.root, MANIFEST), 4)

    def test_changed_model_blocks_source_manifest_update(self):
        before = (self.root / MANIFEST).read_bytes()
        self.artifact.write_bytes(b"changed artifact")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            update_manifest(self.root)
        self.assertEqual(before, (self.root / MANIFEST).read_bytes())

    def test_private_and_escaping_paths_are_rejected(self):
        for path in (".git/HEAD", "outputs/result.json", "data/test.json",
                     "../secret", "/tmp/secret", "a/../../secret", "a\\b"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                release_path(self.root, path)

    def test_symlink_is_rejected(self):
        (self.root / "alias").symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            release_path(self.root, "alias")

    def test_malformed_duplicate_and_empty_manifests_are_rejected(self):
        entry = f"{digest(self.source)}  README.md\n"
        for content in ("", "invalid README.md\n", entry + entry):
            with self.subTest(content=content):
                (self.root / MANIFEST).write_text(content)
                with self.assertRaises(ValueError):
                    check_manifest(self.root, MANIFEST)


if __name__ == "__main__":
    unittest.main()
