import os
import sys
import json
import time
import traceback
import threading

from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image, ImageOps


# ── Device / ONNX Runtime providers ──────────────────────────────────────────

def get_onnx_device():
    try:
        import onnxruntime as rt
        available = rt.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            return "GPU (CUDA)", available
        if 'DmlExecutionProvider' in available:
            return "GPU (DirectML)", available
        return "CPU", available
    except ImportError:
        return None, []


def _select_providers(force_cpu=False):
    if force_cpu:
        return ['CPUExecutionProvider']
    import onnxruntime as ort
    available = set(ort.get_available_providers())
    providers = []
    if 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')
    if 'DmlExecutionProvider' in available:
        providers.append('DmlExecutionProvider')
    providers.append('CPUExecutionProvider')
    return providers


# ── Download / cache helpers (shared) ────────────────────────────────────────

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


def _clean_onnx_model(model_path, progress_cb=None):
    """Rewrite baked-in device-transfer nodes so the model loads on any provider.

    Some ONNX files (e.g. OppaiOracle, saved after CUDA-EP graph partitioning)
    carry MemcpyToHost/MemcpyFromHost nodes in the serialized graph. The CPU EP in
    onnxruntime-directml has no kernel for them, so a CPU-only session fails to
    initialize with NOT_IMPLEMENTED ("Could not find an implementation for
    MemcpyToHost"). Those nodes are 1-in/1-out identity copies, so we rewrite them
    to Identity (verified byte-identical output) and cache the result next to the
    original. Returns the cleaned path, or the original path when cleaning isn't
    needed or onnx isn't importable.
    """
    clean_path = model_path + ".clean.onnx"
    try:
        if (os.path.exists(clean_path) and os.path.getsize(clean_path) > 0
                and os.path.getmtime(clean_path) >= os.path.getmtime(model_path)):
            return clean_path
        import onnx
    except ImportError:
        return model_path
    try:
        model = onnx.load(model_path)
        changed = 0
        for node in model.graph.node:
            if node.op_type in ("MemcpyToHost", "MemcpyFromHost", "Memcpy"):
                node.op_type = "Identity"
                node.domain = ""
                del node.attribute[:]
                changed += 1
        if changed == 0:
            return model_path
        if progress_cb:
            progress_cb(f"Preparing ONNX graph ({changed} device-transfer node(s))...")
        tmp = clean_path + ".part"
        onnx.save(model, tmp)
        os.replace(tmp, clean_path)
        return clean_path
    except Exception:
        traceback.print_exc()
        return model_path


def _composite_rgb(img, background_color):
    """Flatten RGBA/LA/transparent images onto a solid background.
    Returns (rgb_image, was_composited)."""
    if img.mode in ('RGBA', 'LA') or 'transparency' in img.info:
        background = Image.new('RGB', img.size, background_color)
        rgba = img.convert('RGBA')
        background.paste(rgba.convert('RGB'), mask=rgba.getchannel('A'))
        return background, True
    return img.convert('RGB'), False


# ── OppaiOracle (Grio43/OppaiOracle) ─────────────────────────────────────────

OPPAI_REPO_ID = "Grio43/OppaiOracle"
OPPAI_BASE_URL = f"https://huggingface.co/{OPPAI_REPO_ID}/resolve/main"
_oppai_cache = {}
_oppai_cache_lock = threading.Lock()


def _oppai_load(model_variant="V1.1", progress_cb=None, force_cpu=False):
    cache_key = (model_variant, force_cpu)
    with _oppai_cache_lock:
        if cache_key in _oppai_cache:
            return _oppai_cache[cache_key]

    cache = _cache_dir(f"oppai_oracle/{model_variant}_onnx")
    base = f"{OPPAI_BASE_URL}/{model_variant}_onnx"
    files = _ensure_files(
        cache, base,
        ("model.onnx", "vocabulary.json", "preprocessing.json"),
        progress_cb=progress_cb, label_prefix=f"{model_variant}/",
    )

    with open(files["vocabulary.json"], "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    tag_to_index = vocab_data.get("tag_to_index", vocab_data)
    index_to_tag = {int(idx): tag for tag, idx in tag_to_index.items()}

    with open(files["preprocessing.json"], "r", encoding="utf-8") as f:
        pp = json.load(f)
    image_size = int(pp.get("image_size", 448))

    import onnxruntime as ort
    import numpy as np
    model_path = _clean_onnx_model(files["model.onnx"], progress_cb=progress_cb)
    providers = _select_providers(force_cpu=force_cpu)

    # Try the GPU provider(s) first, but verify with a real inference call before
    # committing to the session: some GPU EP / model combinations (e.g. DirectML
    # on this graph) load fine but fail deep inside the EP kernel on the first
    # run(). Detecting that here -- once, at load time -- means the user sees one
    # quiet fallback instead of a per-image crash-and-retry during a batch.
    session = None
    if providers != ['CPUExecutionProvider']:
        if progress_cb:
            progress_cb(f"Loading OppaiOracle {model_variant} ONNX session (GPU)...")
        so = ort.SessionOptions()
        # The failing EP's C++ exception text is emitted in the OS locale codepage
        # (e.g. CP932 on Japanese Windows); pybind11 fails to UTF-8-decode it and
        # onnxruntime also logs a garbled [E:onnxruntime ... ExecuteKernel] line to
        # stderr. Severity 4 (FATAL-only) is the only level that suppresses it.
        so.log_severity_level = 4
        try:
            candidate = ort.InferenceSession(model_path, sess_options=so, providers=providers)
            input_names = [i.name for i in candidate.get_inputs()]
            feed = {input_names[0]: np.zeros((1, 3, image_size, image_size), dtype=np.float32)}
            if "padding_mask" in input_names:
                feed["padding_mask"] = np.zeros((1, image_size, image_size), dtype=bool)
            candidate.run(None, feed)
            session = candidate
        except Exception:
            # Known-bad EP/model combination; not actionable here, so fall
            # through to CPU quietly instead of surfacing a raw traceback.
            if progress_cb:
                progress_cb(
                    f"GPU inference not usable for OppaiOracle {model_variant}; using CPU..."
                )

    if session is None:
        if progress_cb and providers == ['CPUExecutionProvider']:
            progress_cb(f"Loading OppaiOracle {model_variant} ONNX session (CPU)...")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        force_cpu = True

    input_names = [i.name for i in session.get_inputs()]
    bundle = {
        "session": session,
        "index_to_tag": index_to_tag,
        "pad_idx": int(tag_to_index.get("<PAD>", 0)),
        "unk_idx": int(tag_to_index.get("<UNK>", 1)),
        "image_size": image_size,
        "mean": pp.get("normalize_mean", [0.5, 0.5, 0.5]),
        "std": pp.get("normalize_std", [0.5, 0.5, 0.5]),
        "pad_color": tuple(pp.get("pad_color_rgb", [114, 114, 114])),
        "primary_input": input_names[0],
        "has_mask_input": "padding_mask" in input_names,
        "model_variant": model_variant,
        "force_cpu": force_cpu,
    }
    with _oppai_cache_lock:
        _oppai_cache[cache_key] = bundle
        if force_cpu:
            # Also serve any future explicit force_cpu=True request from this
            # same CPU session instead of rebuilding it.
            _oppai_cache[(model_variant, True)] = bundle
    return bundle


def _oppai_invalidate_gpu_cache(model_variant):
    with _oppai_cache_lock:
        _oppai_cache.pop((model_variant, False), None)


def _oppai_preprocess(image_path, image_size, mean, std, pad_color):
    """Letterbox-resize to image_size×image_size, padded with pad_color."""
    import numpy as np
    with Image.open(image_path) as img:
        img.load()
        img = ImageOps.exif_transpose(img)
        img, was_composited = _composite_rgb(img, pad_color)
        arr = np.asarray(img, dtype=np.uint8)

    h, w = arr.shape[:2]
    scale = min(image_size / w, image_size / h, 1.0)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    if new_w != w or new_h != h:
        arr = np.asarray(
            Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR),
            dtype=np.uint8,
        )

    canvas = np.full((image_size, image_size, 3), pad_color, dtype=np.uint8)
    top = (image_size - new_h) // 2
    left = (image_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = arr

    mask = np.ones((image_size, image_size), dtype=bool)
    mask[top:top + new_h, left:left + new_w] = False

    x = canvas.astype(np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32).reshape(1, 1, 3)) \
        / np.array(std, dtype=np.float32).reshape(1, 1, 3)
    x = x.transpose(2, 0, 1)
    return np.expand_dims(x, axis=0), np.expand_dims(mask, axis=0), was_composited


def _sigmoid_if_needed(scores):
    import numpy as np
    if scores.min() < 0.0 or scores.max() > 1.0:
        scores = 1.0 / (1.0 + np.exp(-scores.astype(np.float64)))
        return scores.astype(np.float32)
    return scores


class ImageReadError(Exception):
    """Raised when reading/decoding an image fails, as opposed to a model or
    execution-provider failure. Callers must not treat this as a GPU failure."""


def _oppai_infer(bundle, image_path, threshold, top_k):
    import numpy as np
    try:
        inp, padding_mask, was_composited = _oppai_preprocess(
            image_path, bundle["image_size"], bundle["mean"], bundle["std"], bundle["pad_color"],
        )
    except Exception as e:
        raise ImageReadError(f"{image_path}: {e}") from e
    feed = {bundle["primary_input"]: inp}
    if bundle["has_mask_input"]:
        feed["padding_mask"] = padding_mask

    scores = _sigmoid_if_needed(bundle["session"].run(None, feed)[0][0])

    idxs = np.argsort(scores)[::-1]
    pad_idx, unk_idx = bundle["pad_idx"], bundle["unk_idx"]
    index_to_tag = bundle["index_to_tag"]

    tags = []
    for idx in idxs:
        score = float(scores[idx])
        if score < threshold:
            break
        idx_int = int(idx)
        if idx_int in (pad_idx, unk_idx):
            continue
        name = index_to_tag.get(idx_int)
        if name is None:
            continue
        if was_composited and name == 'gray_background':
            continue
        tags.append(name)
        if len(tags) >= top_k:
            break
    return tags


# ── Worker classes ───────────────────────────────────────────────────────────

def _merge_tags(file_manager, img_path, new_tags):
    current = file_manager.read_tags(img_path)
    added = False
    for tag in new_tags:
        if tag not in current:
            current.append(tag)
            added = True
    if added:
        file_manager.save_tags(img_path, current)


class _BaseBatchTaggerWorker(QThread):
    """Common scaffolding for batch taggers. Subclasses provide:
       - LABEL (str): human-readable tagger name for status messages
       - _load(progress_cb, force_cpu): warm up / download model, return a bundle
       - _infer(bundle, image_path): return list[str] of tags for one image
       - _invalidate_gpu_cache(): drop the cached GPU bundle so future loads use CPU
    """
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int, int, str)

    LABEL = ""

    def __init__(self, file_manager, image_paths):
        super().__init__()
        self.file_manager = file_manager
        self.image_paths = image_paths

    def _load(self, progress_cb=None, force_cpu=False):
        raise NotImplementedError

    def _infer(self, bundle, image_path):
        raise NotImplementedError

    def _invalidate_gpu_cache(self):
        pass

    def run(self):
        total = len(self.image_paths)
        device_name, _ = get_onnx_device()
        if device_name is None:
            self.finished.emit(0, total, "ONNX Runtime not installed.")
            return

        try:
            bundle = self._load(progress_cb=lambda msg: self.progress.emit(0, total, msg))
        except Exception as e:
            traceback.print_exc()
            self.finished.emit(0, total, str(e))
            return

        success_count = 0
        for i, img_path in enumerate(self.image_paths):
            if self.isInterruptionRequested():
                break
            self.progress.emit(i, total, os.path.basename(img_path))
            try:
                new_tags = self._infer(bundle, img_path)
            except ImageReadError as e:
                print(f"[{self.LABEL}] {e}; skipped", file=sys.stderr)
                continue
            except Exception as e:
                if bundle.get("force_cpu"):
                    print(f"[{self.LABEL}] {os.path.basename(img_path)}: "
                          f"{type(e).__name__}: {e}; skipped", file=sys.stderr)
                    continue
                print(f"[{self.LABEL}] {os.path.basename(img_path)}: "
                      f"{type(e).__name__}; falling back to CPU", file=sys.stderr)
                self._invalidate_gpu_cache()
                try:
                    self.progress.emit(i, total, "GPU inference failed; reloading on CPU...")
                    bundle = self._load(
                        force_cpu=True,
                        progress_cb=lambda msg: self.progress.emit(i, total, msg),
                    )
                except Exception as e2:
                    traceback.print_exc()
                    self.finished.emit(success_count, total, f"CPU fallback failed to load: {e2}")
                    return
                try:
                    new_tags = self._infer(bundle, img_path)
                except ImageReadError as e2:
                    print(f"[{self.LABEL}] {e2}; skipped", file=sys.stderr)
                    continue
                except Exception as e2:
                    print(f"[{self.LABEL}] {os.path.basename(img_path)}: "
                          f"{type(e2).__name__}: {e2}; skipped", file=sys.stderr)
                    continue
            _merge_tags(self.file_manager, img_path, new_tags)
            success_count += 1
        self.finished.emit(success_count, total, "")


class _BaseSingleTaggerWorker(QThread):
    """Common scaffolding for single-image taggers."""
    finished = pyqtSignal(list, str)
    progress = pyqtSignal(str)

    LABEL = ""

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def _load(self, progress_cb=None, force_cpu=False):
        raise NotImplementedError

    def _infer(self, bundle, image_path):
        raise NotImplementedError

    def _invalidate_gpu_cache(self):
        pass

    def run(self):
        try:
            device_name, _ = get_onnx_device()
            if device_name is None:
                self.finished.emit([], "ONNX Runtime not installed.")
                return
            self.progress.emit(f"Running {self.LABEL} on {device_name}...")
            bundle = self._load(progress_cb=self.progress.emit)
            try:
                tags = self._infer(bundle, self.image_path)
            except ImageReadError:
                raise
            except Exception as e:
                if bundle.get("force_cpu"):
                    raise
                print(f"[{self.LABEL}] {type(e).__name__}; falling back to CPU", file=sys.stderr)
                self._invalidate_gpu_cache()
                self.progress.emit("GPU inference failed; reloading on CPU...")
                bundle = self._load(progress_cb=self.progress.emit, force_cpu=True)
                tags = self._infer(bundle, self.image_path)
            self.finished.emit(tags, "")
        except Exception as e:
            traceback.print_exc()
            self.finished.emit([], str(e))


class OppaiOracleWorker(_BaseSingleTaggerWorker):
    def __init__(self, image_path, threshold=0.4, top_k=50, model_variant="V1.1"):
        super().__init__(image_path)
        self.threshold = threshold
        self.top_k = top_k
        self.model_variant = model_variant
        self.LABEL = f"OppaiOracle {model_variant}"

    def _load(self, progress_cb=None, force_cpu=False):
        return _oppai_load(self.model_variant, progress_cb=progress_cb, force_cpu=force_cpu)

    def _infer(self, bundle, image_path):
        return _oppai_infer(bundle, image_path, self.threshold, self.top_k)

    def _invalidate_gpu_cache(self):
        _oppai_invalidate_gpu_cache(self.model_variant)


class BatchOppaiOracleWorker(_BaseBatchTaggerWorker):
    def __init__(self, file_manager, image_paths, threshold=0.4, top_k=50, model_variant="V1.1"):
        super().__init__(file_manager, image_paths)
        self.threshold = threshold
        self.top_k = top_k
        self.model_variant = model_variant
        self.LABEL = f"OppaiOracle {model_variant}"

    def _load(self, progress_cb=None, force_cpu=False):
        return _oppai_load(self.model_variant, progress_cb=progress_cb, force_cpu=force_cpu)

    def _infer(self, bundle, image_path):
        return _oppai_infer(bundle, image_path, self.threshold, self.top_k)

    def _invalidate_gpu_cache(self):
        _oppai_invalidate_gpu_cache(self.model_variant)
