import os
import csv
import json
import traceback
import sys
import threading
from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image, ImageOps

def get_onnx_device():
    try:
        import onnxruntime as rt
        available_providers = rt.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            return "GPU (CUDA)", available_providers
        elif 'DmlExecutionProvider' in available_providers:
            return "GPU (DirectML)", available_providers
        else:
            return "CPU", available_providers
    except ImportError:
        return None, []

def get_torch_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── OppaiOracle helpers ──────────────────────────────────────────────────────

OPPAI_REPO_ID = "Grio43/OppaiOracle"
OPPAI_BASE_URL = f"https://huggingface.co/{OPPAI_REPO_ID}/resolve/main"
_oppai_cache = {}
_oppai_cache_lock = threading.Lock()


def _oppai_select_providers():
    import onnxruntime as ort
    available = set(ort.get_available_providers())
    providers = []
    if 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')
    if 'DmlExecutionProvider' in available:
        providers.append('DmlExecutionProvider')
    providers.append('CPUExecutionProvider')
    return providers


def _oppai_cache_dir(model_variant):
    root = os.environ.get("OPPAI_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "oppai_oracle"
    )
    out = os.path.join(root, f"{model_variant}_onnx")
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
                if r.status_code in (416,):
                    # already complete
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
            import time
            time.sleep(min(2 ** attempt, 10))

    os.replace(tmp, dest)


def _oppai_download_files(model_variant, progress_cb=None):
    cache = _oppai_cache_dir(model_variant)
    paths = {}
    for fname in ("model.onnx", "vocabulary.json", "preprocessing.json"):
        dest = os.path.join(cache, fname)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            url = f"{OPPAI_BASE_URL}/{model_variant}_onnx/{fname}"
            _stream_download(url, dest, progress_cb=progress_cb,
                             label=f"{model_variant}/{fname}")
        paths[fname] = dest
    return paths


def _oppai_load(model_variant="V1.1", progress_cb=None):
    with _oppai_cache_lock:
        if model_variant in _oppai_cache:
            return _oppai_cache[model_variant]

    files = _oppai_download_files(model_variant, progress_cb)

    if progress_cb:
        progress_cb(f"Loading OppaiOracle {model_variant} ONNX session...")
    import onnxruntime as ort
    session = ort.InferenceSession(files["model.onnx"], providers=_oppai_select_providers())

    with open(files["vocabulary.json"], "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    tag_to_index = vocab_data.get("tag_to_index", vocab_data)
    index_to_tag = {int(idx): tag for tag, idx in tag_to_index.items()}
    pad_idx = int(tag_to_index.get("<PAD>", 0))
    unk_idx = int(tag_to_index.get("<UNK>", 1))

    with open(files["preprocessing.json"], "r", encoding="utf-8") as f:
        pp = json.load(f)
    image_size = int(pp.get("image_size", 448))
    mean = pp.get("normalize_mean", [0.5, 0.5, 0.5])
    std = pp.get("normalize_std", [0.5, 0.5, 0.5])
    pad_color = tuple(pp.get("pad_color_rgb", [114, 114, 114]))

    input_names = [i.name for i in session.get_inputs()]
    has_mask_input = "padding_mask" in input_names
    primary_input = input_names[0]

    bundle = {
        "session": session,
        "index_to_tag": index_to_tag,
        "pad_idx": pad_idx,
        "unk_idx": unk_idx,
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "pad_color": pad_color,
        "primary_input": primary_input,
        "has_mask_input": has_mask_input,
    }

    with _oppai_cache_lock:
        _oppai_cache[model_variant] = bundle
    return bundle


def _oppai_preprocess(image_path, image_size, mean, std, pad_color):
    import numpy as np
    was_composited = False
    with Image.open(image_path) as img:
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode in ('RGBA', 'LA') or 'transparency' in img.info:
            was_composited = True
            background = Image.new('RGB', img.size, pad_color)
            img_rgba = img.convert('RGBA')
            alpha = img_rgba.getchannel('A')
            background.paste(img_rgba.convert('RGB'), mask=alpha)
            img = background
        else:
            img = img.convert('RGB')
        arr = np.asarray(img, dtype=np.uint8)

    h, w = arr.shape[:2]
    target = image_size
    scale = min(target / w, target / h, 1.0)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    if new_w != w or new_h != h:
        pil_img = Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR)
        arr = np.asarray(pil_img, dtype=np.uint8)

    canvas = np.full((target, target, 3), pad_color, dtype=np.uint8)
    top = (target - new_h) // 2
    left = (target - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = arr

    mask = np.ones((target, target), dtype=bool)
    mask[top:top + new_h, left:left + new_w] = False

    x = canvas.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    x = (x - mean_arr) / std_arr
    x = x.transpose(2, 0, 1)
    return np.expand_dims(x, axis=0), np.expand_dims(mask, axis=0), was_composited


def _oppai_infer(bundle, image_path, threshold, top_k):
    import numpy as np
    inp, padding_mask, was_composited = _oppai_preprocess(
        image_path, bundle["image_size"], bundle["mean"], bundle["std"], bundle["pad_color"]
    )
    feed = {bundle["primary_input"]: inp}
    if bundle["has_mask_input"]:
        feed["padding_mask"] = padding_mask

    outputs = bundle["session"].run(None, feed)
    scores = outputs[0][0]
    # Output is sigmoid already for V1/V1.1; clip just in case for legacy.
    if scores.min() < 0.0 or scores.max() > 1.0:
        scores = 1.0 / (1.0 + np.exp(-scores.astype(np.float64)))
        scores = scores.astype(np.float32)

    idxs = np.argsort(scores)[::-1]
    pad_idx, unk_idx = bundle["pad_idx"], bundle["unk_idx"]
    index_to_tag = bundle["index_to_tag"]

    tags = []
    for idx in idxs:
        idx_int = int(idx)
        score = float(scores[idx])
        if score < threshold:
            break
        if idx_int in (pad_idx, unk_idx):
            continue
        tag_name = index_to_tag.get(idx_int)
        if tag_name is None:
            continue
        if was_composited and tag_name == 'gray_background':
            continue
        tags.append(tag_name)
        if len(tags) >= top_k:
            break
    return tags

class PixAITaggerWorker(QThread):
    finished = pyqtSignal(list, str) # tags, error_msg
    progress = pyqtSignal(str)

    def __init__(self, image_path, threshold=0.35):
        super().__init__()
        self.image_path = image_path
        self.threshold = threshold
        self.model_name = "v0.9"

    def run(self):
        print(f"--- Starting PixAI Tagger ---")
        device_name, providers = get_onnx_device()
        
        if device_name is None:
            self.finished.emit([], "ONNX Runtime not installed.")
            return

        try:
            self.progress.emit(f"Running inference on {device_name}...")
            try:
                from imgutils.tagging import get_pixai_tags
            except ImportError:
                from imgutils.tagging.pixai import get_pixai_tags
            import inspect

            sig = inspect.signature(get_pixai_tags)
            params = sig.parameters

            tagger_kwargs = {"model_name": self.model_name}
            if "threshold" in params:
                tagger_kwargs["threshold"] = self.threshold
            elif "thresholds" in params:
                tagger_kwargs["thresholds"] = self.threshold

            general_tags, character_tags = get_pixai_tags(self.image_path, **tagger_kwargs)
            result_tags = list(character_tags.keys()) + list(general_tags.keys())
            self.finished.emit(result_tags, "")
        except Exception as e:
            print(f"PixAI failed ({type(e).__name__}: {e}). Falling back to SwinV2...")
            try:
                from imgutils.tagging import get_wd14_tags
                general_tags, character_tags = get_wd14_tags(
                    self.image_path, model_name='SwinV2',
                    general_threshold=self.threshold, character_threshold=self.threshold
                )
                result_tags = list(character_tags.keys()) + list(general_tags.keys())
                self.finished.emit(result_tags, "")
            except Exception as e2:
                traceback.print_exc()
                self.finished.emit([], str(e2))

class BatchPixAITaggerWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int, int, str)

    def __init__(self, file_manager, image_paths, threshold=0.35):
        super().__init__()
        self.file_manager = file_manager
        self.image_paths = image_paths
        self.threshold = threshold
        self.model_name = "v0.9"

    def run(self):
        device_name, _ = get_onnx_device()
        if device_name is None:
            self.finished.emit(0, len(self.image_paths), "ONNX Runtime not installed.")
            return

        success_count = 0
        total = len(self.image_paths)
        use_fallback = False
        
        try:
            try:
                from imgutils.tagging import get_pixai_tags
            except ImportError:
                from imgutils.tagging.pixai import get_pixai_tags
            import inspect
            sig = inspect.signature(get_pixai_tags)
            params = sig.parameters
        except Exception as e:
            print(f"PixAI import failed ({type(e).__name__}: {e}). Using SwinV2.")
            use_fallback = True
            from imgutils.tagging import get_wd14_tags

        for i, img_path in enumerate(self.image_paths):
            if self.isInterruptionRequested(): break
            self.progress.emit(i, total, os.path.basename(img_path))
            try:
                if not use_fallback:
                    tagger_kwargs = {"model_name": self.model_name}
                    if "threshold" in params:
                        tagger_kwargs["threshold"] = self.threshold
                    elif "thresholds" in params:
                        tagger_kwargs["thresholds"] = self.threshold
                        
                    general_tags, character_tags = get_pixai_tags(img_path, **tagger_kwargs)
                else:
                    general_tags, character_tags = get_wd14_tags(img_path, model_name='SwinV2', general_threshold=self.threshold, character_threshold=self.threshold)
                
                new_tags = list(character_tags.keys()) + list(general_tags.keys())
                current_tags = self.file_manager.read_tags(img_path)
                added = False
                for tag in new_tags:
                    if tag not in current_tags:
                        current_tags.append(tag)
                        added = True
                if added: self.file_manager.save_tags(img_path, current_tags)
                success_count += 1
            except Exception as e:
                print(f"Error: {e}")
        self.finished.emit(success_count, total, "")

class OppaiOracleWorker(QThread):
    finished = pyqtSignal(list, str)
    progress = pyqtSignal(str)

    def __init__(self, image_path, threshold=0.4, top_k=50, model_variant="V1.1"):
        super().__init__()
        self.image_path = image_path
        self.threshold = threshold
        self.top_k = top_k
        self.model_variant = model_variant

    def run(self):
        try:
            device_name, _ = get_onnx_device()
            if device_name is None:
                self.finished.emit([], "ONNX Runtime not installed.")
                return
            self.progress.emit(f"Running OppaiOracle {self.model_variant} on {device_name}...")
            bundle = _oppai_load(self.model_variant, progress_cb=self.progress.emit)
            tags = _oppai_infer(bundle, self.image_path, self.threshold, self.top_k)
            self.finished.emit(tags, "")
        except Exception as e:
            traceback.print_exc()
            self.finished.emit([], str(e))


class BatchOppaiOracleWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int, int, str)

    def __init__(self, file_manager, image_paths, threshold=0.4, top_k=50, model_variant="V1.1"):
        super().__init__()
        self.file_manager = file_manager
        self.image_paths = image_paths
        self.threshold = threshold
        self.top_k = top_k
        self.model_variant = model_variant

    def run(self):
        device_name, _ = get_onnx_device()
        if device_name is None:
            self.finished.emit(0, len(self.image_paths), "ONNX Runtime not installed.")
            return

        try:
            bundle = _oppai_load(self.model_variant)
        except Exception as e:
            traceback.print_exc()
            self.finished.emit(0, len(self.image_paths), str(e))
            return

        success_count = 0
        total = len(self.image_paths)
        for i, img_path in enumerate(self.image_paths):
            if self.isInterruptionRequested():
                break
            self.progress.emit(i, total, os.path.basename(img_path))
            try:
                new_tags = _oppai_infer(bundle, img_path, self.threshold, self.top_k)
                current_tags = self.file_manager.read_tags(img_path)
                added = False
                for tag in new_tags:
                    if tag not in current_tags:
                        current_tags.append(tag)
                        added = True
                if added:
                    self.file_manager.save_tags(img_path, current_tags)
                success_count += 1
            except Exception:
                traceback.print_exc()
        self.finished.emit(success_count, total, "")
