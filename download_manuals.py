"""Download the two official public manuals listed in sources.json."""

from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    sources = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    target_dir = ROOT / "manuals"
    target_dir.mkdir(exist_ok=True)

    for source in sources:
        target = target_dir / source["filename"]
        if target.exists() and target.stat().st_size > 10_000:
            print(f"Already present: {target.name}")
            continue
        request = urllib.request.Request(
            source["url"], headers={"User-Agent": "Mozilla/5.0 coursework downloader"}
        )
        print(f"Downloading {source['title']} ...")
        with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
            shutil.copyfileobj(response, out)
        print(f"Saved {target.name} ({target.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()

