"""Resume ranged base model downloads from HF mirror and verify model SHA-256."""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import httpx  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

ROOT = Path("data/external/faster-whisper-base")
REVISION = "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"
SIZE = 145217532
BLOCK = 4194304
SHA256 = "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9"


def fetch(start):
    end = min(SIZE, start + BLOCK) - 1
    path = ROOT / str(start)
    if path.exists() and path.stat().st_size == end - start + 1:
        return path
    endpoint = os.environ["HF_ENDPOINT"].rstrip("/")
    url = f"{endpoint}/Systran/faster-whisper-base/resolve/{REVISION}/model.bin"
    for attempt in range(4):
        try:
            download_url = (
                url + f"?part={start}&attempt={attempt}"
                if attempt < 2
                else "https://modelscope.cn/models/pengzhendong/"
                "faster-whisper-base/resolve/master/model.bin"
            )
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                response = client.get(
                    download_url,
                    headers={"Range": f"bytes={start}-{end}"},
                )
                response.raise_for_status()
                assert response.status_code == 206
                assert response.headers["content-range"].startswith(f"bytes {start}-{end}/")
                assert len(response.content) == end - start + 1
                path.write_bytes(response.content)
                return path
        except (httpx.HTTPError, AssertionError):
            if attempt == 3:
                raise
    raise RuntimeError("Download failed")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ["config.json", "tokenizer.json", "vocabulary.txt"]:
        if not (ROOT / name).exists():
            source = hf_hub_download("Systran/faster-whisper-base", name, revision=REVISION)
            (ROOT / name).write_bytes(Path(source).read_bytes())
    target = ROOT / "model.bin"
    if not target.exists():
        with ThreadPoolExecutor(max_workers=8) as pool:
            parts = list(pool.map(fetch, range(0, SIZE, BLOCK)))
        with target.open("xb") as stream:
            for part in parts:
                stream.write(part.read_bytes())
    assert target.stat().st_size == SIZE
    assert hashlib.sha256(target.read_bytes()).hexdigest() == SHA256
    print("base weights verified:", SHA256, flush=True)


if __name__ == "__main__":
    main()
