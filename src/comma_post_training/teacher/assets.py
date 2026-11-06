"""Immutable official LFS identity; model bytes remain outside Git."""

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelIdentity:
    kind: str
    commit: str
    sha256: str
    size: int
    url: str

    def __post_init__(self):
        if self.kind not in ("official", "synthetic_fixture"):
            raise ValueError("teacher kind must identify official weights or a synthetic fixture")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("teacher identity requires a SHA256")
        if type(self.size) is not int or self.size < 1:
            raise ValueError("teacher identity requires a positive file size")

    def to_dict(self):
        return asdict(self)


OFFICIAL = ModelIdentity(
    kind="official",
    commit="084747c75d2cbd23af65ab7a9e770bbd7b98bac9",
    sha256="a501760a9d1d5fef0eab2b8c5d122d06124fc26dc8e0782e0aa94b82a208f0ff",
    size=1757355221,
    url=(
        "https://media.githubusercontent.com/media/commaai/openpilot/"
        "084747c75d2cbd23af65ab7a9e770bbd7b98bac9/"
        "openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx"
    ),
)


def verify_model(path: Path, identity: ModelIdentity = OFFICIAL) -> dict:
    if not path.is_file():
        raise ValueError(f"Teacher model missing: {path}; run comma-teacher fetch --model {path}")
    if path.stat().st_size != identity.size:
        raise ValueError(
            "Teacher size differs from its pinned LFS identity (a pointer is not weights)"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != identity.sha256:
        raise ValueError("Teacher SHA256 differs from its pinned official identity")
    return {"model": str(path.resolve()), "verified": True, **identity.to_dict()}


def fetch_model(path: Path) -> dict:
    if path.exists():
        return verify_model(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".download")
    if pending.exists():
        raise ValueError(
            f"incomplete download exists: {pending}; inspect/remove it before retrying"
        )
    try:
        request = urllib.request.Request(
            OFFICIAL.url, headers={"User-Agent": "comma-post-training"}
        )
        with urllib.request.urlopen(request, timeout=60) as response, pending.open("xb") as stream:
            count = 0
            while chunk := response.read(8 * 1024 * 1024):
                count += len(chunk)
                if count > OFFICIAL.size:
                    raise ValueError("download exceeds the pinned official file size")
                stream.write(chunk)
        verify_model(pending)
        pending.replace(path)
    except urllib.error.HTTPError as error:
        raise ValueError(
            f"Teacher download HTTP {error.code} from media.githubusercontent.com; "
            "check environment network access; no alternate weights were substituted"
        ) from error
    finally:
        pending.unlink(missing_ok=True)
    report = verify_model(path)
    receipt = path.with_suffix(path.suffix + ".identity.json")
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    return report
