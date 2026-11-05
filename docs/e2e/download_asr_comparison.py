"""Download AliMeeting Eval and pinned comparison weights into ignored storage."""

import argparse
import hashlib
import json
import os
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
ROOT = Path("data/external")
ALI_URL = (
    "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz"
)
MODELS = {
    "small": ("Systran/faster-whisper-small", "536b0662742c02347bc0e980a01041f333bce120"),
    "medium": ("Systran/faster-whisper-medium", "08e178d48790749d25932bbc082711ddcfdfbc4f"),
    "large-v3-turbo": (
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    ),
}
BLOCK = 4 * 1024 * 1024


def ranged_download(url, target, size, sha256=None, fallback=None):
    parts = target.parent / (target.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    mirror_failed = threading.Event()

    def fetch(start):
        end = min(size, start + BLOCK) - 1
        part = parts / str(start)
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(6):
            source = fallback if fallback and (mirror_failed.is_set() or attempt >= 1) else url
            try:
                with httpx.Client(
                    timeout=45 if source == fallback else 12, follow_redirects=True
                ) as client:
                    response = client.get(
                        source + (f"?part={start}&attempt={attempt}" if fallback else ""),
                        headers={"Range": f"bytes={start}-{end}"},
                    )
                    response.raise_for_status()
                    assert response.status_code == 206
                    assert response.headers["content-range"].startswith(f"bytes {start}-{end}/")
                    assert len(response.content) == end - start + 1
                    part.write_bytes(response.content)
                    return part
            except (httpx.HTTPError, AssertionError):
                if fallback:
                    mirror_failed.set()
                if attempt == 5:
                    raise RuntimeError(
                        f"Range download failed: {target.name}, offset {start}"
                    ) from None
        raise RuntimeError("Download failed")

    if not target.exists():
        started = time.perf_counter()
        offsets = list(range(0, size, BLOCK))
        with ThreadPoolExecutor(max_workers=12 if fallback else 24) as pool:
            futures = [pool.submit(fetch, start) for start in offsets]
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                if i % 32 == 0 or i == len(offsets):
                    print(
                        target.name,
                        i,
                        "/",
                        len(offsets),
                        round(time.perf_counter() - started),
                        "seconds",
                        flush=True,
                    )
        with target.open("xb") as stream:
            for start in offsets:
                stream.write((parts / str(start)).read_bytes())
    assert target.stat().st_size == size
    hasher = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    if sha256:
        assert digest == sha256
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["audio", *MODELS])
    args = parser.parse_args()
    if args.kind == "audio":
        local = ROOT / "alimeeting"
        target = local / "Eval_Ali.tar.gz"
        digest = ranged_download(ALI_URL, target, 3673718355)
        selected = []
        with tarfile.open(target, "r|gz") as archive:
            for member in archive:
                if "R8001_M8004" not in member.name or not member.isfile():
                    continue
                if "near" not in member.name and not member.name.endswith(".TextGrid"):
                    continue
                output = local / Path(member.name).name
                if not output.exists():
                    output.write_bytes(archive.extractfile(member).read())
                selected.append({"archive_member": member.name, "local": output.as_posix()})
                print("Extracted", member.name, flush=True)
        (local / "matching_sources.json").write_text(
            json.dumps({"url": ALI_URL, "archive_sha256": digest, "members": selected}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    else:
        repo, revision = MODELS[args.kind]
        local = ROOT / ("faster-whisper-" + args.kind)
        local.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            cache = local / "download_provenance.json"
            if cache.exists():
                metadata = json.loads(cache.read_text(encoding="utf-8"))
            else:
                response = client.get(
                    f"https://hf-mirror.com/api/models/{repo}/revision/{revision}",
                    params={"blobs": "true"},
                )
                response.raise_for_status()
                metadata = response.json()
                cache.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            assert metadata["sha"] == revision
            for item in metadata["siblings"]:
                name = item["rfilename"]
                if name not in {
                    "config.json",
                    "preprocessor_config.json",
                    "tokenizer.json",
                    "vocabulary.txt",
                    "vocabulary.json",
                    "model.bin",
                }:
                    continue
                url = f"https://hf-mirror.com/{repo}/resolve/{revision}/{name}"
                target = local / name
                if name == "model.bin":
                    ranged_download(
                        url,
                        target,
                        item["size"],
                        item["lfs"]["sha256"],
                        f"https://modelscope.cn/models/pengzhendong/"
                        f"faster-whisper-{args.kind}/resolve/master/model.bin",
                    )
                elif not target.exists():
                    try:
                        downloaded = client.get(url)
                        downloaded.raise_for_status()
                    except httpx.HTTPError:
                        downloaded = client.get(
                            f"https://modelscope.cn/models/pengzhendong/"
                            f"faster-whisper-{args.kind}/resolve/master/{name}"
                        )
                        downloaded.raise_for_status()
                    assert len(downloaded.content) == item["size"]
                    target.write_bytes(downloaded.content)
                if name != "model.bin":
                    blob = b"blob " + str(target.stat().st_size).encode() + b"\0"
                    assert hashlib.sha1(blob + target.read_bytes()).hexdigest() == item["blobId"]
        (local / "download_provenance.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(args.kind, "weights verified", flush=True)


if __name__ == "__main__":
    main()
