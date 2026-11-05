"""Read-only verification of source pins, Git links, and initialized checkouts."""

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*arguments: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *arguments], cwd=cwd, text=True).strip()


def verify(*, metadata_only: bool = False) -> dict:
    manifest = json.loads((ROOT / "third_party/sources.lock.json").read_text())
    errors, sources, optional_missing = [], [], []
    for source in manifest["sources"]:
        name, path, sha = source["id"], source["path"], source["commit"]
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(f"invalid commit for {name}")
        if not re.fullmatch(r"https://github.com/[\w.-]+/[\w.-]+\.git", source["url"]):
            raise ValueError(f"invalid public source URL for {name}")
        directory = ROOT / path
        directory.resolve().relative_to(ROOT)
        for key, expected in (("path", path), ("url", source["url"])):
            actual = git("config", "-f", ".gitmodules", "--get", f"submodule.{name}.{key}")
            if actual != expected:
                errors.append(f"{name}: .gitmodules {key} differs from lock")
        index = git("ls-files", "--stage", "--", path).split()
        if len(index) < 3 or index[:3] != ["160000", sha, "0"]:
            errors.append(f"{name}: Git index must contain the locked submodule commit")
        if not metadata_only:
            if not (directory / ".git").exists():
                errors.append(f"{name}: checkout is not initialized")
                continue
            if (
                Path(git("rev-parse", "--show-toplevel", cwd=directory)).resolve()
                != directory.resolve()
            ):
                errors.append(f"{name}: wrong checkout root")
            if git("rev-parse", "HEAD", cwd=directory) != sha:
                errors.append(f"{name}: checkout HEAD differs from lock")
            if git("status", "--porcelain", "--untracked-files=all", cwd=directory):
                errors.append(f"{name}: upstream working tree has local changes")
        sources.append({"id": name, "path": path, "commit": sha})
    if not metadata_only:
        optional = {item["path"]: item for item in manifest["optional_asset_submodules"]}
        # Keep leading spaces: submodule status encodes readiness in its first character.
        status = subprocess.check_output(
            ["git", "submodule", "status", "--recursive"], cwd=ROOT, text=True
        )
        for line in status.splitlines():
            match = re.match(r"^([ +\-U])([0-9a-f]{40}) ([^ ]+)", line)
            if match is None:
                errors.append("unrecognized submodule status")
                continue
            state, sha, path = match.groups()
            if state == "-" and path in optional and optional[path]["commit"] == sha:
                optional_missing.append({"path": path, "commit": sha, "url": optional[path]["url"]})
            elif state != " ":
                errors.append(f"{path}: missing, conflicted, or changed source checkout")
    return {
        "mode": "metadata_only" if metadata_only else "source_checkouts",
        "ok": not errors,
        "sources": sources,
        "recursive_checkout_status_checked": not metadata_only,
        "missing_optional_asset_submodules": optional_missing,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true", help="check pins without checkouts")
    args = parser.parse_args()
    try:
        result = verify(metadata_only=args.metadata_only)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
