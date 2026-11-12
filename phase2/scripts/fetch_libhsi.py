"""Parallel, resumable range download of LIB-HSI.zip (CSIRO DAP collection 58044, 60 GB).

The server limits each connection to ~40 kB/s, so the file is fetched as many byte ranges at once.
Runs on the host (needs only python3 + curl). Bytes are written in place into a sparse <out> file;
finished chunks are recorded in <out>.chunks.json, so the script can be stopped and restarted at any time.
An existing prefix (e.g. an earlier browser download) is kept. The tail (zip central directory) is fetched first.

Usage:  python3 phase2/scripts/fetch_libhsi.py /home/nagano/workspace/LIB-HSI.zip [--workers 32]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

API = "https://data.csiro.au/dap/ws/v2/collections/58044/data/64182114"
SIZE = 60059070054
CHUNK = 64 << 20
TAIL = 64 << 20  # central directory lives in the last few MB


def signed_url() -> str:
    """The API answers with a 302 to a pre-signed S3 URL (valid 48 h); ask for a fresh one per chunk.
    The API call itself fails now and then, so retry with backoff instead of giving up."""
    for k in range(20):
        r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{redirect_url}", "--max-time", "60", API],
                           capture_output=True, text=True)
        if r.stdout.startswith("https://"):
            return r.stdout
        time.sleep(min(10 * (k + 1), 120))
    raise RuntimeError(f"no redirect after retries: {r.stdout!r} {r.stderr!r}")


def fetch(fd, start, end, stop) -> int:
    """Download [start, end] into fd at the same offsets. Returns bytes written; raises on short reads."""
    pos = start
    while pos <= end and not stop.is_set():
        p = subprocess.Popen(["curl", "-s", "-S", "-f", "--speed-limit", "2000", "--speed-time", "60",
                              "-r", f"{pos}-{end}", signed_url()], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while not stop.is_set():
            buf = p.stdout.read(1 << 20)
            if not buf:
                break
            os.pwrite(fd, buf, pos)
            pos += len(buf)
        p.stdout.close()
        rc = p.wait()
        if pos <= end and not stop.is_set():
            time.sleep(5)  # transient failure (rc != 0) or early close: retry from the current offset
    return pos - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    state_path = a.out + ".chunks.json"
    done = set(json.load(open(state_path))["done"]) if os.path.exists(state_path) else set()
    if not done and os.path.exists(a.out):  # whole chunks inside an existing contiguous prefix count as done
        done = set(range(os.path.getsize(a.out) // CHUNK))
    fd = os.open(a.out, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size < SIZE:
        os.ftruncate(fd, SIZE)  # sparse; unwritten regions stay holes until fetched

    n = (SIZE + CHUNK - 1) // CHUNK
    tail_first = sorted(range(n), key=lambda i: (i * CHUNK < SIZE - TAIL, i))
    todo = [i for i in tail_first if i not in done]
    lock, stop, t0, got = threading.Lock(), threading.Event(), time.time(), [0]

    def job(i):
        s, e = i * CHUNK, min(SIZE, (i + 1) * CHUNK) - 1
        try:
            k = fetch(fd, s, e, stop)
        except Exception as ex:  # leave the chunk undone; a rerun picks it up
            print(f"{time.strftime('%H:%M:%S')} chunk {i} failed: {ex}", flush=True)
            return
        if stop.is_set():
            return
        with lock:
            done.add(i)
            got[0] += k
            json.dump({"size": SIZE, "chunk": CHUNK, "done": sorted(done)}, open(state_path + ".tmp", "w"))
            os.replace(state_path + ".tmp", state_path)
            el = time.time() - t0
            print(f"{time.strftime('%H:%M:%S')} chunk {i} ok | {len(done)}/{n} chunks "
                  f"{100 * len(done) / n:.1f}% | {got[0] / el / 1e6:.2f} MB/s | "
                  f"ETA {(n - len(done)) * CHUNK / max(got[0] / el, 1) / 3600:.1f} h", flush=True)

    print(f"{len(done)}/{n} chunks already present, {len(todo)} to fetch with {a.workers} workers", flush=True)
    try:
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(job, todo))
    except KeyboardInterrupt:
        stop.set()
    os.fsync(fd)
    os.close(fd)
    print("complete" if len(done) == n else f"stopped with {len(done)}/{n} chunks", flush=True)
    sys.exit(0 if len(done) == n else 1)


if __name__ == "__main__":
    main()
