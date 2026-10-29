"""Create one atomic CalVer release commit with a matching uv.lock and annotated tag.

Run from a clean main checkout after uv sync --locked: uv run --no-sync python scripts/release.py patch
The script intentionally does not push; review before git push --follow-tags.
"""

from __future__ import annotations

import argparse
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as file:
        return str(tomllib.load(file)["project"]["version"])


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def tag_exists(tag: str) -> bool:
    command = ["git", "rev-parse", "--quiet", "--verify", f"refs/tags/{tag}"]
    return subprocess.run(command, cwd=ROOT, capture_output=True).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=["patch", "release"])
    args = parser.parse_args()
    if run("git", "branch", "--show-current") != "main":
        parser.error("Release must be prepared on main.")
    if run("git", "status", "--porcelain"):
        parser.error("Working tree must be clean before starting a release.")
    old_version = project_version()
    # bump-my-version itself must not create a commit or tag: the lockfile must
    # be updated after the version change, but before either git operation.
    subprocess.run(["bump-my-version", "bump", args.part], cwd=ROOT, check=True)
    new_version = project_version()
    # A "release" bump within a month that already has a release resets the
    # patch number, giving an older version whose tag usually exists already.
    if version_key(new_version) <= version_key(old_version) or tag_exists(f"v{new_version}"):
        # The working tree was clean before the bump, so undoing it is safe.
        subprocess.run(["git", "checkout", "--", "pyproject.toml"], cwd=ROOT, check=True)
        parser.error(
            f"Bumping {args.part!r} gives {new_version} after {old_version}, or tag "
            f"v{new_version} exists. Use 'patch' for another release in the same month."
        )
    subprocess.run(["uv", "lock"], cwd=ROOT, check=True)
    subprocess.run(["uv", "lock", "--check"], cwd=ROOT, check=True)
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
