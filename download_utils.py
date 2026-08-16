import os
import time

from PyQt6.QtCore import QThread


# ── Download / cache helpers (shared across AI features) ────────────────────

def _cache_dir(subpath):
    root = os.environ.get("AI_TAGGER_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "ai_tagger"
    )
    out = os.path.join(root, subpath)
    os.makedirs(out, exist_ok=True)
    return out


def _stream_download(url, dest, progress_cb=None, label=None, max_retries=5):
    import requests
    label = label or os.path.basename(dest)
    tmp = dest + ".part"

    for attempt in range(1, max_retries + 1):
        resume_pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = {"Accept-Encoding": "identity"}
        if resume_pos:
            headers["Range"] = f"bytes={resume_pos}-"
        try:
            with requests.get(url, stream=True, headers=headers, timeout=(30, 60)) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                content_length = int(r.headers.get("Content-Length", 0))
                total = (content_length + resume_pos) if r.status_code == 206 else content_length
                mode = "ab" if r.status_code == 206 and resume_pos else "wb"
                if mode == "wb":
                    resume_pos = 0
                downloaded = resume_pos
                last_report = 0
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        thread = QThread.currentThread()
                        if thread is not None and thread.isInterruptionRequested():
                            raise RuntimeError("Download cancelled")
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb and downloaded - last_report >= 2 * 1024 * 1024:
                            mb = downloaded / (1024 * 1024)
                            if total:
                                tot = total / (1024 * 1024)
                                pct = downloaded / total * 100
                                progress_cb(f"Downloading {label}: {mb:.1f}/{tot:.1f} MB ({pct:.1f}%)")
                            else:
                                progress_cb(f"Downloading {label}: {mb:.1f} MB")
                            last_report = downloaded
            break
        except (requests.exceptions.RequestException, OSError) as e:
            if progress_cb:
                progress_cb(f"Download {label} interrupted ({e}); retrying {attempt}/{max_retries}...")
            if attempt == max_retries:
                raise
            time.sleep(min(2 ** attempt, 10))

    os.replace(tmp, dest)


def _ensure_files(cache_dir, base_url, filenames, progress_cb=None, label_prefix=""):
    paths = {}
    for fname in filenames:
        dest = os.path.join(cache_dir, fname)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            _stream_download(
                f"{base_url}/{fname}", dest,
                progress_cb=progress_cb, label=f"{label_prefix}{fname}",
            )
        paths[fname] = dest
    return paths
