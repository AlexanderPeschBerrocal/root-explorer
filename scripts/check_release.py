"""Check that HEAD is a complete release commit as created by scripts/release.py.

Used by the release workflow after a dry run and before publishing a release:
HEAD must change only the version in pyproject.toml and uv.lock, and carry an
annotated CalVer tag that agrees with both. The repository is not modified.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The same tag format that the Docker publish workflow accepts.
TAG_PATTERN = re.compile(r"v[1-9][0-9]{3}\.([1-9]|1[0-2])\.[0-9]+")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def main() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    previous = tomllib.loads(git("show", "HEAD~1:pyproject.toml"))["project"]["version"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = [p["version"] for p in lock["package"] if p["name"] == "root-explorer"]
    tag = f"v{version}"
    tag_type = subprocess.run(
        ["git", "cat-file", "-t", f"refs/tags/{tag}"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()

    problems = []
    if version_key(version) <= version_key(previous):
        problems.append(f"version {version} does not increase {previous}")
    if not TAG_PATTERN.fullmatch(tag):
        problems.append(f"{tag} is not a vYYYY.M.N CalVer tag")
    if locked != [version]:
        problems.append(f"uv.lock records root-explorer {locked}, not {version}")
    if git("log", "-1", "--format=%s") != f"Release: {version}":
        problems.append("HEAD is not the release commit")
    changed = set(git("diff", "--name-only", "HEAD~1", "HEAD").splitlines())
    if changed != {"pyproject.toml", "uv.lock"}:
        problems.append(f"release commit changes {sorted(changed)}")
    if tag_type != "tag":
        problems.append(f"{tag} is not an annotated tag")
    elif git("rev-parse", f"{tag}^{{commit}}") != git("rev-parse", "HEAD"):
        problems.append(f"{tag} does not point to HEAD")
    if git("status", "--porcelain"):
        problems.append("working tree is not clean")
    if subprocess.run(["uv", "lock", "--check"], cwd=ROOT).returncode != 0:
        problems.append("uv.lock does not match pyproject.toml")

    if problems:
        sys.exit("Release check failed:\n- " + "\n- ".join(problems))
    print(f"Release {tag} verified: {previous} -> {version}")


if __name__ == "__main__":
    main()
