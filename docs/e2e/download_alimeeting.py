"""Download only the gzip prefix needed for five minutes of AliMeeting channel 0."""

import hashlib
import io
import json
import struct
import time
import wave
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

URL = "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz"
MEMBER = "Eval_Ali/Eval_Ali_far/audio_dir/R8001_M8004_MS801.wav"
LOCAL = Path("data/external/alimeeting")


def fetch(start):
    end = start + 4 * 1024 * 1024 - 1
    path = LOCAL / f"prefix-{start}.part"
    if path.exists() and path.stat().st_size == end - start + 1:
        return path.read_bytes()
    for attempt in range(3):
        try:
            with httpx.Client(timeout=45, follow_redirects=True) as client:
                response = client.get(URL, headers={"Range": f"bytes={start}-{end}"})
                response.raise_for_status()
                assert response.status_code == 206
                assert response.headers["content-range"].startswith(f"bytes {start}-{end}/")
                assert len(response.content) == end - start + 1
                path.write_bytes(response.content)
                return response.content
        except (httpx.HTTPError, AssertionError):
            if attempt == 2:
                raise
    raise RuntimeError("Download failed")


def main():
    LOCAL.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    decoder = zlib.decompressobj(31)
    raw = bytearray()
    downloaded = 0
    needed = 2048 + 80 + 300 * 16000 * 16
    with ThreadPoolExecutor(max_workers=4) as pool:
        for base in range(0, 128 * 1024 * 1024, 16 * 1024 * 1024):
            blocks = list(pool.map(fetch, range(base, base + 16 * 1024 * 1024, 4194304)))
            for block in blocks:
                downloaded += len(block)
                raw.extend(decoder.decompress(block))
            print(f"Downloaded {downloaded} bytes; expanded {len(raw)} bytes", flush=True)
            if len(raw) >= needed:
                break
    assert len(raw) >= needed
    assert bytes(raw[1536:1636]).split(b"\0")[0].decode() == MEMBER
    audio = bytes(raw[2048:needed])
    assert audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"
    assert struct.unpack("<HHIIHH", audio[20:36]) == (65534, 8, 16000, 256000, 16, 16)
    assert audio[72:76] == b"data"
    pcm = np.frombuffer(audio[80:], dtype="<i2").reshape(-1, 8)[:, 0].tobytes()
    clip = LOCAL / "R8001_M8004_MS801.ch0.first300s.wav"
    with wave.open(str(clip), "wb") as target:
        target.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        target.writeframes(pcm)
    with wave.open(io.BytesIO(clip.read_bytes())) as check:
        assert check.getnframes() / check.getframerate() == 300
    metadata = {
        "source_url": URL,
        "source_page": "https://www.openslr.org/119/",
        "license": "CC BY-SA 4.0",
        "attribution": "Alibaba Group / AliMeeting (Yu et al., ICASSP 2022)",
        "archive_member": MEMBER,
        "transformation": "First 300 seconds, channel index 0, PCM16 mono 16kHz",
        "downloaded_bytes": downloaded,
        "download_wall_seconds": time.perf_counter() - started,
        "audio_seconds": 300,
        "sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
        "clip": clip.as_posix(),
    }
    (LOCAL / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
