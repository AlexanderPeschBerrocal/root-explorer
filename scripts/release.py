"""Create one atomic CalVer release commit with a matching uv.lock and annotated tag.

Run from a clean main checkout: uv run --no-sync --group release python scripts/release.py patch
The script intentionally does not push; review before git push --follow-tags.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=["patch", "release"])
    args = parser.parse_args()
    if run("git", "branch", "--show-current") != "main":
        parser.error("Release must be prepared on main.")
    if run("git", "status", "--porcelain"):
        parser.error("Working tree must be clean before starting a release.")
    # bump-my-version itself must not create a commit or tag: the lockfile must
    # be updated after the version change, but before either git operation.
    subprocess.run(["bump-my-version", "bump", args.part], cwd=ROOT, check=True)
    subprocess.run(["uv", "lock"], cwd=ROOT, check=True)
    subprocess.run(["uv", "lock", "--check"], cwd=ROOT, check=True)
    with (ROOT / "pyproject.toml").open("rb") as file:
        new_version = tomllib.load(file)["project"]["version"]
    with (ROOT / "uv.lock").open("rb") as file:
        lock = tomllib.load(file)
    locked = [p["version"] for p in lock["package"] if p["name"] == "root-explorer"]
    if locked != [new_version]:
        raise RuntimeError(f"Version mismatch: project {new_version}, lock {locked}")
    subprocess.run(["git", "add", "pyproject.toml", "uv.lock"], cwd=ROOT, check=True)
    if not run("git", "diff", "--cached", "--name-only"):
        raise RuntimeError("No release changes have been staged.")
    subprocess.run(["git", "commit", "-m", f"Release: {new_version}"], cwd=ROOT, check=True)
    subprocess.run(
        ["git", "tag", "-a", f"v{new_version}", "-m", f"Release {new_version}"],
        cwd=ROOT,
        check=True,
    )
    print(f"Created local release v{new_version}. Review then git push --follow-tags.")


if __name__ == "__main__":
    main()
