#!/usr/bin/env python3
"""Check release files, or regenerate their manifest from the Git index."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "SHA256SUMS"
ARTIFACT_MANIFEST = "artifacts/parc_selector/SHA256SUMS"
EXCLUDED_PARTS = frozenset({
    ".git", ".venv", "__pycache__", ".pytest_cache", ".DS_Store",
    "outputs", "data", "models", "cache",
})


def release_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or path.as_posix() != relative
        or ".." in path.parts
        or "\\" in relative
        or any(ord(char) < 32 for char in relative)
        or EXCLUDED_PARTS.intersection(path.parts)
    ):
        raise ValueError(f"Disallowed release path: {relative!r}")
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Release path escapes root: {relative}")
    for parent in (target, *target.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"Symlinks are not release files: {relative}")
    return target


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def tracked_paths(root: Path) -> list[str]:
    if not (root / ".git").exists():
        raise ValueError("Updating checksums requires a Git checkout")
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "-z"], cwd=root
    ).decode("utf-8")
    paths = sorted(set(output.rstrip("\0").split("\0")) - {MANIFEST, ""})
    if not paths:
        raise ValueError("No tracked release files")
    for relative in paths:
        release_path(root, relative)
    return paths


def read_manifest(root: Path, name: str) -> dict[str, str]:
    entries = {}
    lines = release_path(root, name).read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"Malformed checksum in {name}:{number}")
        expected, relative = match.groups()
        release_path(root, relative)
        if relative == name or relative in entries:
            raise ValueError(f"Duplicate or self-referencing checksum: {relative}")
        entries[relative] = expected
    if not entries:
        raise ValueError(f"Empty checksum manifest: {name}")
    return entries


def check_manifest(root: Path, name: str) -> int:
    entries = read_manifest(root, name)
    if name == MANIFEST and (root / ".git").exists():
        tracked = set(tracked_paths(root))
        if tracked != set(entries):
            missing = sorted(tracked - set(entries))
            extra = sorted(set(entries) - tracked)
            raise ValueError(f"Manifest inventory mismatch: missing={missing}, extra={extra}")
    for relative, expected in entries.items():
        if digest(release_path(root, relative)) != expected:
            raise ValueError(f"SHA256 mismatch: {relative}")
    return len(entries)


def update_manifest(root: Path) -> int:
    # Do not let a source update silently bless a changed frozen model.
    check_manifest(root, ARTIFACT_MANIFEST)
    paths = tracked_paths(root)
    content = "".join(f"{digest(release_path(root, path))}  {path}\n" for path in paths)
    (root / MANIFEST).write_text(content, encoding="utf-8")
    return len(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "update"))
    args = parser.parse_args()
    try:
        if args.command == "update":
            print(f"Updated {MANIFEST}: {update_manifest(ROOT)} tracked files")
        else:
            for name in (MANIFEST, ARTIFACT_MANIFEST):
                print(f"{name}: OK ({check_manifest(ROOT, name)} files)")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Release integrity check failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
