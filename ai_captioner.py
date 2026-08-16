import os
import sys
import io
import time
import base64
import threading
import subprocess
import traceback

from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image, ImageOps

from download_utils import _cache_dir, _ensure_files
from ai_tagger import _composite_rgb, ImageReadError


# ── Shared image preprocessing (both backends) ───────────────────────────────

def _load_rgb_image(image_path, background_color=(255, 255, 255)):
    """Decode `image_path`, apply EXIF orientation, and flatten any
    transparency onto `background_color`. Returns a plain RGB PIL Image.

    Raises ImageReadError (same exception ai_tagger.py's batch loop already
    knows how to catch-and-skip) on any decode failure, so callers can tell
    "bad image, skip it" apart from a model/backend failure exactly like the
    existing tagger does.
    """
    try:
        with Image.open(image_path) as img:
            img.load()
            img = ImageOps.exif_transpose(img)
            rgb, _was_composited = _composite_rgb(img, background_color)
            # _composite_rgb() either returns a brand-new Image (the
            # composited-onto-background case) or img.convert('RGB') (a
            # fresh Image decoupled from the file handle either way), so
            # this is safe to hand back after the `with` block closes img.
            return rgb
    except ImageReadError:
        raise
    except Exception as e:
        raise ImageReadError(f"{image_path}: {e}") from e


def _rgb_to_png_base64(rgb_image):
    """Encode a PIL RGB Image as a base64 PNG string (no data: prefix)."""
    buf = io.BytesIO()
    rgb_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Cancellation ──────────────────────────────────────────────────────────────

class _Cancelled(Exception):
    """Raised internally by a CaptionBackend to signal that a caption() call
    was aborted because of cancellation, as opposed to completing normally or
    failing for a genuine model/backend reason. Worker classes catch this
    specifically so a cancelled, partial/garbage generation can never be
    mistaken for a real caption and written to a sidecar .txt file.
    """


# ── Backend protocol ──────────────────────────────────────────────────────────

class CaptionBackend:
    """Common interface for a local vision-language-model captioning backend.

    A single backend instance is created by the UI layer, loaded once (load()
    is idempotent -- safe to call again as a cheap no-op once warm), and then
    reused across an entire batch and/or many single-image caption calls.
    close() releases GPU/CPU resources when the UI switches models/backends.

    MODEL_STATUS is a short human-readable string the UI can surface next to
    the backend's name/label (e.g. to flag a backend as experimental).
    """

    MODEL_STATUS = "unknown"

    def load(self, progress_cb=None):
        """Download (if needed) and warm up the model. Idempotent: calling
        this again after a successful load should be a fast no-op."""
        raise NotImplementedError

    def caption(self, image_path, instruction, cancel_check=None):
        """Caption a single image. `instruction` is the caption-style prompt
        text supplied by the UI layer -- this module does not hardcode any
        prompt wording. `cancel_check` is an optional zero-arg callable that
        returns True when the in-flight generation should stop early.
        Returns the caption text (str). Raises ImageReadError for image
        decode failures, _Cancelled if cancellation is what stopped
        generation, or any other Exception for a model/backend failure.
        """
        raise NotImplementedError

    def cancel(self):
        """Best-effort request to interrupt an in-flight caption() call."""
        pass

    def close(self):
        """Release GPU/CPU resources held by this backend."""
        pass


# ── Shared provenance helper (both backends) ─────────────────────────────────

def _pinned_revision(repo_id, cache_root, progress_cb=None):
    """Resolve and persist the exact commit SHA to pin all downloads for
    `repo_id` to, so repeated loads of an already-downloaded cache never
    depend on a floating 'main' ref (and don't need network access to
    re-resolve it once cached). Shared by both backends: BF16Backend passes
    the returned SHA into snapshot_download(revision=...), GGUFBackend uses
    it to build a /resolve/<sha>/ URL for download_utils._ensure_files -- so
    a repo update on 'main' after first download can never silently change
    what gets loaded on a later run.

    Resolution failure is a hard error, not a silent fallback to 'main':
    both source repos' *content* provenance is unverified (only their
    config/API metadata and, for GGUFBackend, a live inference run, have
    been verified), so pinning a specific commit is mandatory here, not
    best-effort.
    """
    rev_file = os.path.join(cache_root, "revision.txt")
    if os.path.exists(rev_file):
        with open(rev_file, "r", encoding="utf-8") as f:
            sha = f.read().strip()
        if sha:
            return sha

    from huggingface_hub import HfApi
    if progress_cb:
        progress_cb(f"Resolving pinned revision for {repo_id}...")
    sha = HfApi().model_info(repo_id).sha
    if not sha:
        raise RuntimeError(
            f"Could not resolve a commit SHA for {repo_id}; refusing to "
            "download from an unpinned 'main' ref."
        )

    os.makedirs(cache_root, exist_ok=True)
    tmp = rev_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(sha)
    os.replace(tmp, rev_file)
    return sha


# ── BF16Backend (transformers + torch, CUDA) ─────────────────────────────────

AEON7_REPO_ID = "AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16"


class _CancelStoppingCriteria:
    """transformers StoppingCriteria that stops generation as soon as a
    threading.Event is set. Implemented as a plain object exposing __call__
    with the signature transformers.StoppingCriteria expects, so we don't
    need to import the base class just to satisfy an isinstance check --
    transformers' StoppingCriteriaList only requires the callable protocol.
    """

    def __init__(self, cancel_event):
        self._cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs):
        import torch
        if self._cancel_event.is_set():
            # Every sequence in the batch is "done" -> generate() stops.
            return torch.ones(input_ids.shape[0], dtype=torch.bool)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool)


def _bf16_make_tqdm_class(progress_cb):
    """A tqdm subclass that forwards progress into this codebase's
    progress_cb(str) convention (see ai_tagger.py / download_utils.py's
    percentage/MB-formatted progress strings), for use as snapshot_download's
    tqdm_class= argument. Also aborts the download on cancellation, using the
    same QThread.currentThread().isInterruptionRequested() convention
    download_utils._stream_download already uses: huggingface_hub's
    snapshot_download drives this tqdm instance's update() synchronously,
    inside its own chunked download loop, so raising here is what actually
    lets Cancel interrupt a ~54GB in-progress download instead of letting it
    run to completion regardless.
    """
    from tqdm.asyncio import tqdm_asyncio

    class _ProgressForwardingTqdm(tqdm_asyncio):
        def update(self, n=1):
            thread = QThread.currentThread()
            if thread is not None and thread.isInterruptionRequested():
                raise RuntimeError("Download cancelled")

            result = super().update(n)
            if progress_cb is not None:
                try:
                    total = self.total
                    current = self.n
                    desc = self.desc or "Downloading"
                    if total:
                        pct = current / total * 100
                        if (self.unit or "").startswith("B"):
                            cur_mb = current / (1024 * 1024)
                            tot_mb = total / (1024 * 1024)
                            progress_cb(f"{desc}: {cur_mb:.1f}/{tot_mb:.1f} MB ({pct:.1f}%)")
                        else:
                            progress_cb(f"{desc}: {current}/{total} ({pct:.1f}%)")
                    else:
                        progress_cb(f"{desc}: {current}")
                except Exception:
                    pass
            return result

    return _ProgressForwardingTqdm


class BF16Backend(CaptionBackend):
    """AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16 via transformers + torch.

    EXPERIMENTAL / UNVERIFIED END-TO-END: the load/generate code path below is
    verified against the repo's config.json and processor metadata only
    (architectures=["Qwen3_5ForConditionalGeneration"], model_type="qwen3_5",
    natively resolved by transformers 5.15.0 with no trust_remote_code needed;
    AutoProcessor resolves to Qwen3VLProcessor but requires torchvision to be
    installed -- add it to requirements.txt). A full ~54GB weights download and
    load has NOT been executed as part of writing this module, so treat this
    backend as unproven relative to GGUFBackend (verified end-to-end on real
    hardware) until someone runs it for real.
    """

    MODEL_STATUS = (
        "Experimental / unverified end-to-end -- config & processor metadata "
        "checked only, a full weights load has not been run. Prefer the GGUF "
        "backend unless you specifically need this one."
    )

    REPO_ID = AEON7_REPO_ID
    # Everything needed to load + run inference; skips extra repo files
    # (README, example scripts, etc.) that snapshot_download would otherwise
    # also pull down.
    ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.jinja", "tokenizer*"]

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None
        # Per-call cancel signal -- NOT shared/reset instance state. A fresh
        # threading.Event is created for each caption() call and published
        # here (under _cancel_lock) only while that call is in flight, so:
        #  - a cancel() landing in the gap *between* two images can't be
        #    silently discarded by the next call's setup (there's nothing to
        #    discard -- _active_cancel_event is None between calls), and
        #  - two concurrent caption() calls on one backend instance (if a
        #    caller ever did that) don't share/stomp on one Event.
        self._active_cancel_event = None
        self._cancel_lock = threading.Lock()
        self._lock = threading.Lock()

    def load(self, progress_cb=None):
        with self._lock:
            if self._model is not None:
                return

            try:
                import torch
            except ImportError as e:
                raise RuntimeError(
                    "PyTorch is not installed; the BF16 caption backend requires torch "
                    "with CUDA support."
                ) from e
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "No CUDA device visible to PyTorch; the BF16 caption backend "
                    "requires a CUDA GPU (this ~27B-parameter BF16 model is not "
                    "practical to run on CPU)."
                )

            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except ImportError as e:
                raise RuntimeError(
                    "transformers is not installed or too old; the BF16 caption "
                    "backend requires transformers >= 5.12.1."
                ) from e

            cache_root = _cache_dir("aeon7_bf16")
            revision = _pinned_revision(self.REPO_ID, cache_root, progress_cb=progress_cb)

            if progress_cb:
                progress_cb(f"Downloading {self.REPO_ID} @ {revision[:12]} (this is ~54GB)...")

            from huggingface_hub import snapshot_download
            tqdm_class = _bf16_make_tqdm_class(progress_cb)
            local_dir = snapshot_download(
                self.REPO_ID,
                revision=revision,
                cache_dir=cache_root,
                allow_patterns=self.ALLOW_PATTERNS,
                tqdm_class=tqdm_class,
            )

            if progress_cb:
                progress_cb("Loading BF16 model onto CUDA (this can take a while for 54GB)...")

            # NOTE: no trust_remote_code=True here, deliberately. config.json's
            # architectures=["Qwen3_5ForConditionalGeneration"] / model_type=
            # "qwen3_5" resolve natively against an up-to-date transformers
            # install. If this raises because the installed transformers is
            # too old to recognize "qwen3_5", we surface that clearly instead
            # of silently retrying with trust_remote_code=True -- this repo's
            # provenance is unverified, and executing arbitrary repo Python is
            # a decision only the user should make, not an automatic fallback.
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    local_dir,
                    dtype=torch.bfloat16,
                    device_map="cuda:0",
                )
            except (KeyError, ValueError) as e:
                raise RuntimeError(
                    "Failed to load AEON-7 Qwen3.5-VL model: the installed "
                    "transformers version may not recognize model_type "
                    f"'qwen3_5' yet. Try `pip install -U transformers`. ({e})"
                ) from e

            try:
                processor = AutoProcessor.from_pretrained(local_dir)
            except ImportError as e:
                raise RuntimeError(
                    "AutoProcessor for this model requires the 'torchvision' package "
                    "(its video-processor sub-component imports it even though we "
                    "only caption still images). Install it with `pip install "
                    "torchvision` and try again."
                ) from e

            model.eval()
            self._model = model
            self._processor = processor
            self._device = model.device
            if progress_cb:
                progress_cb("BF16 model loaded.")

    def caption(self, image_path, instruction, cancel_check=None):
        if self._model is None:
            raise RuntimeError("BF16Backend.caption() called before load()")

        rgb_image = _load_rgb_image(image_path)

        import torch
        from transformers import StoppingCriteriaList

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        # enable_thinking=False renders an already-closed, empty <think></think>
        # block into the prompt itself, so generation goes straight to the
        # caption instead of burning the token budget on the chat template's
        # default reasoning_effort='xhigh' trace. No post-hoc stripping is
        # needed on this backend (unlike GGUFBackend) because this is decided
        # at prompt-construction time, not left to a black-box completion API.
        prompt_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self._processor(
            images=[rgb_image], text=[prompt_text], return_tensors="pt",
        ).to(self._device)

        cancel_event = threading.Event()
        with self._cancel_lock:
            self._active_cancel_event = cancel_event

        try:
            # Always attach the stopping criteria (not only when cancel_check
            # is given) so a direct backend.cancel() call -- with no
            # cancel_check-based watcher at all -- can still interrupt an
            # in-flight generate() call on its own.
            stopping_criteria = StoppingCriteriaList([_CancelStoppingCriteria(cancel_event)])

            def _watch_cancel():
                # Bridges the caller's cancel_check() poll (e.g. the worker's
                # isInterruptionRequested()) into cancel_event, checked once
                # per generation step by generate()'s stopping-criteria loop.
                while not cancel_event.is_set():
                    if cancel_check() if cancel_check is not None else False:
                        cancel_event.set()
                        return
                    time.sleep(0.2)

            watcher = None
            if cancel_check is not None:
                watcher = threading.Thread(target=_watch_cancel, daemon=True)
                watcher.start()

            was_cancelled = False
            try:
                with torch.no_grad():
                    output_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        stopping_criteria=stopping_criteria,
                    )
                # Checked *before* the finally below unconditionally sets the
                # event to unblock the watcher thread -- this is the only
                # place that can tell "stopped because of cancellation" apart
                # from "finished normally", so it must run first.
                was_cancelled = cancel_event.is_set()
            finally:
                cancel_event.set()  # let the watcher thread exit promptly
                if watcher is not None:
                    watcher.join(timeout=1.0)
        finally:
            with self._cancel_lock:
                if self._active_cancel_event is cancel_event:
                    self._active_cancel_event = None

        if was_cancelled:
            raise _Cancelled(f"Caption generation cancelled for {image_path}")

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        text = self._processor.decode(new_tokens, skip_special_tokens=True).strip()
        return text

    def cancel(self):
        with self._cancel_lock:
            event = self._active_cancel_event
        if event is not None:
            event.set()

    def close(self):
        with self._lock:
            self._model = None
            self._processor = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


# ── GGUFBackend (llama-cpp-python, CUDA) ─────────────────────────────────────

GGUF_REPO_ID = "JonathanColetti/Qwen3.8-27B-Uncensored-GGUF"
GGUF_MMPROJ_FILENAME = "Qwen3.8-27B-Uncensored-vision-f16.gguf"

# VRAM tier -> quant filename. Presets, not guarantees -- see the free-VRAM
# check in load() for why.
GGUF_VRAM_TIERS = {
    "12GB": "Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf",
    "16GB": "Qwen3.8-27B-Uncensored-noMTP-IQ4_XS.gguf",
    "24GB": "Qwen3.8-27B-Uncensored-noMTP-Q6_K.gguf",
    "32GB": "Qwen3.8-27B-Uncensored-noMTP-Q8_0.gguf",
}


def _gguf_free_vram_mb():
    """Best-effort free-VRAM query (MB). Tries pynvml, then falls back to
    `nvidia-smi`. Returns None if neither is available -- callers must treat
    that as "unknown", not as "plenty of room"."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.free / (1024 * 1024)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            first_line = out.stdout.strip().splitlines()[0].strip()
            return float(first_line)
    except Exception:
        pass

    return None


class GGUFBackend(CaptionBackend):
    """JonathanColetti/Qwen3.8-27B-Uncensored-GGUF via llama-cpp-python +
    llama.cpp's generic MTMD vision handler.

    VERIFIED END-TO-END on real hardware (RTX PRO 6000 Blackwell,
    llama-cpp-python 0.3.34 built with CUDA): a real inference call produced a
    correct image description using this exact load/caption sequence.
    Downloads are pinned to a specific resolved commit SHA (see
    _pinned_revision) rather than the floating 'main' ref, same as
    BF16Backend -- this repo's *content* provenance is unverified even though
    its functional behavior has been verified live.
    """

    MODEL_STATUS = "Verified end-to-end on real hardware."

    REPO_ID = GGUF_REPO_ID

    def __init__(self, vram_tier="24GB", n_ctx=4096):
        if vram_tier not in GGUF_VRAM_TIERS:
            raise ValueError(f"Unknown VRAM tier {vram_tier!r}; choose one of {list(GGUF_VRAM_TIERS)}")
        self.vram_tier = vram_tier
        self.n_ctx = n_ctx
        self._llm = None
        self._chat_handler = None
        self._lock = threading.Lock()
        # Best-effort only: llama-cpp-python's create_chat_completion() does
        # not expose a mid-token abort hook in its stable API surface (as of
        # 0.3.34), so this Event is only ever checked *before* a generation
        # starts (see the worker's per-image isInterruptionRequested() check
        # and the pre-flight check at the top of caption(), plus the check
        # before the reasoning-retry generation). Once create_chat_completion()
        # is actually running there is no way to interrupt it from here --
        # this is a real gap vs. BF16Backend's StoppingCriteria-based abort,
        # not something this flag secretly works around.
        self._cancel_event = threading.Event()

    def load(self, progress_cb=None):
        with self._lock:
            if self._llm is not None:
                return

            quant_filename = GGUF_VRAM_TIERS[self.vram_tier]
            cache_root = _cache_dir(f"aeon7_gguf/{self.vram_tier}")

            # Pin to the exact resolved commit SHA -- not the floating 'main'
            # ref -- so a repo update after first download can never silently
            # change what a cached copy loads, and so a resumed partial
            # download can never get spliced together from two different
            # revisions of the same filename.
            revision = _pinned_revision(self.REPO_ID, cache_root, progress_cb=progress_cb)
            base_url = f"https://huggingface.co/{GGUF_REPO_ID}/resolve/{revision}"

            files = _ensure_files(
                cache_root, base_url,
                (quant_filename, GGUF_MMPROJ_FILENAME),
                progress_cb=progress_cb, label_prefix=f"{self.vram_tier}/",
            )
            quant_path = files[quant_filename]
            mmproj_path = files[GGUF_MMPROJ_FILENAME]

            # Don't attempt to load a half-downloaded pair.
            for path in (quant_path, mmproj_path):
                if not os.path.exists(path) or os.path.getsize(path) == 0:
                    raise RuntimeError(
                        f"GGUF file missing or empty after download: {path}"
                    )

            free_mb = _gguf_free_vram_mb()
            if free_mb is not None:
                quant_size_mb = os.path.getsize(quant_path) / (1024 * 1024)
                mmproj_size_mb = os.path.getsize(mmproj_path) / (1024 * 1024)
                # Quant + mmproj file size is a floor, not the real VRAM use --
                # CUDA context buffers, KV cache, and image-embedding memory
                # add on top. Warn generously rather than failing deep inside
                # a batch run.
                estimated_need_mb = (quant_size_mb + mmproj_size_mb) * 1.25
                if free_mb < estimated_need_mb and progress_cb:
                    progress_cb(
                        f"Warning: only {free_mb:.0f} MB free VRAM detected; the "
                        f"{self.vram_tier} tier ({quant_size_mb:.0f} MB quant + "
                        f"{mmproj_size_mb:.0f} MB mmproj, plus KV cache/context "
                        f"overhead) may not fit. Consider a smaller VRAM tier."
                    )

            if progress_cb:
                progress_cb(f"Loading GGUF model ({self.vram_tier} tier) onto GPU...")

            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import MTMDChatHandler

            chat_handler = MTMDChatHandler(clip_model_path=mmproj_path, verbose=False)
            llm = Llama(
                model_path=quant_path,
                chat_handler=chat_handler,
                n_ctx=self.n_ctx,   # short-caption use case, not the model's 262144 max
                n_gpu_layers=-1,
                verbose=False,
            )

            self._llm = llm
            self._chat_handler = chat_handler
            if progress_cb:
                progress_cb("GGUF model loaded.")

    def caption(self, image_path, instruction, cancel_check=None):
        if self._llm is None:
            raise RuntimeError("GGUFBackend.caption() called before load()")

        def _is_cancelled():
            return self._cancel_event.is_set() or (cancel_check is not None and cancel_check())

        if _is_cancelled():
            raise _Cancelled(f"Caption generation cancelled before starting for {image_path}")

        rgb_image = _load_rgb_image(image_path)
        b64_png = _rgb_to_png_base64(rgb_image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64_png}},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        # The chat template baked into this repo defaults to
        # reasoning_effort='xhigh' (a <think>...</think> block before the
        # actual answer), and create_chat_completion() in llama-cpp-python
        # 0.3.34 has no kwarg to pass enable_thinking/chat_template_kwargs
        # through to disable it. Verified live: with too small a max_tokens,
        # the entire budget is consumed by reasoning and generation never
        # reaches the caption (finish_reason == 'length', no '</think>'
        # anywhere in the output). So: use a generous max_tokens, then keep
        # only the text after '</think>'; if it's missing or empty/whitespace
        # (also reproduced live -- the model can close its reasoning right at
        # the token boundary with nothing meaningful after it), this was a
        # failed generation -- retry once with a bigger budget, and only
        # raise if that still fails. Never save raw/empty text as a caption.
        # 512 was the first value tried and reliably left the more detailed
        # instruction presets (e.g. the Character/Style LoRA prompts, which
        # ask for multi-clause descriptions) truncated mid-sentence after the
        # reasoning trace ate most of the budget -- 768 gives real headroom
        # for those while still being fast for short instructions.
        return self._generate_with_thinking_retry(
            messages, max_tokens=768, retry_max_tokens=1536, is_cancelled=_is_cancelled,
        )

    def _generate_with_thinking_retry(self, messages, max_tokens, retry_max_tokens, is_cancelled=None):
        text, finished_cleanly = self._generate_once(messages, max_tokens)
        if finished_cleanly:
            return text

        # First attempt was truncated before '</think>' appeared. Re-check
        # cancellation before paying for a second, larger generation --
        # otherwise a cancel requested while the first attempt was running
        # would silently extend the already-acknowledged "no mid-generation
        # abort" window by a whole extra generation call.
        if is_cancelled is not None and is_cancelled():
            raise _Cancelled("Caption generation cancelled during reasoning-retry")

        text, finished_cleanly = self._generate_once(messages, retry_max_tokens)
        if finished_cleanly:
            return text

        raise RuntimeError(
            "Generation did not produce usable caption text even at "
            f"max_tokens={retry_max_tokens} (either truncated before reasoning "
            "finished, with no '</think>' anywhere in the output, or the model "
            "closed its reasoning with nothing but whitespace after it); "
            "refusing to save raw/empty text as a caption."
        )

    def _generate_once(self, messages, max_tokens):
        resp = self._llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=0.7,
        )
        choice = resp["choices"][0]
        content = choice["message"].get("content") or ""
        if "</think>" in content:
            caption = content.split("</think>", 1)[1].strip()
            # A closing think-tag is necessary but not sufficient: the model
            # can close its reasoning right at the token boundary and produce
            # nothing (or only whitespace) after it -- reproduced live, where
            # a second call in the same session returned an empty string this
            # way despite '</think>' being present. Treat that the same as
            # "not finished cleanly" so the retry-with-bigger-budget path
            # (and, ultimately, the raise below) kicks in instead of silently
            # handing back an empty caption to be saved to disk.
            if caption:
                return caption, True
            return content, False
        return content, False

    def cancel(self):
        # Best-effort only -- see the note in __init__. This flags future
        # per-image pre-flight checks (and the reasoning-retry check) to
        # skip; it does not interrupt a generation already in progress inside
        # create_chat_completion().
        self._cancel_event.set()

    def close(self):
        with self._lock:
            llm = self._llm
            self._llm = None
            self._chat_handler = None
            if llm is not None:
                close_fn = getattr(llm, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                    except Exception:
                        pass
                else:
                    del llm


# ── Worker classes ────────────────────────────────────────────────────────────

class _BaseBatchCaptionWorker(QThread):
    """Batch-caption a list of images with a given CaptionBackend, writing
    results via file_manager.save_caption() (NOT the tag methods -- captions
    and tags are mutually exclusive per image and share the same sidecar
    .txt). Mirrors ai_tagger.py's _BaseBatchTaggerWorker progress/finished
    signal shapes so the UI layer can reuse the same wiring.

    Unlike the tagger, there is no GPU-fails -> reload-on-CPU retry loop: CPU
    inference of a ~27B-class VLM is not practical, so a backend/model
    failure stops the batch (it's likely to recur for every remaining image,
    e.g. a CUDA OOM won't clear itself) instead of being silently retried on
    CPU or skip-looped through. A bad *image* (ImageReadError), by contrast,
    is unrelated to the model and is safe to skip and continue past.
    """
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int, int, str)

    def __init__(self, file_manager, image_paths, backend, instruction):
        super().__init__()
        self.file_manager = file_manager
        self.image_paths = image_paths
        self.backend = backend
        self.instruction = instruction

    def run(self):
        total = len(self.image_paths)

        try:
            self.backend.load(progress_cb=lambda msg: self.progress.emit(0, total, msg))
        except Exception as e:
            traceback.print_exc()
            self.finished.emit(0, total, str(e))
            return

        def _cancel_check():
            return self.isInterruptionRequested()

        success_count = 0
        for i, img_path in enumerate(self.image_paths):
            if self.isInterruptionRequested():
                break
            self.progress.emit(i, total, os.path.basename(img_path))
            try:
                caption_text = self.backend.caption(img_path, self.instruction, cancel_check=_cancel_check)
            except _Cancelled:
                break
            except ImageReadError as e:
                print(f"[Captioner] {e}; skipped", file=sys.stderr)
                continue
            except Exception as e:
                # A backend/model failure (CUDA OOM, a generation error,
                # GGUFBackend's retry-exhausted RuntimeError, ...) is likely
                # to recur for every remaining image, so stop the batch and
                # surface it -- rather than skip-looping through the rest of
                # a large folder and burying a real, batch-wide problem under
                # N per-image log lines.
                traceback.print_exc()
                print(f"[Captioner] {os.path.basename(img_path)}: "
                      f"{type(e).__name__}: {e}; stopping batch", file=sys.stderr)
                self.finished.emit(success_count, total, str(e))
                return

            # backend.caption() may have completed normally in the same
            # instant cancellation was requested, or -- for a backend like
            # GGUFBackend whose cancellation is pre-flight-only -- may not be
            # able to detect a cancellation request mid-call at all. Re-check
            # here, before writing anything to disk, rather than relying only
            # on the check at the top of the next loop iteration.
            if self.isInterruptionRequested():
                break

            if not self.file_manager.save_caption(img_path, caption_text):
                print(f"[Captioner] {os.path.basename(img_path)}: failed to save caption "
                      f"({self.file_manager.last_error}); skipped", file=sys.stderr)
                continue
            success_count += 1

        self.finished.emit(success_count, total, "")


class _BaseSingleCaptionWorker(QThread):
    """Caption a single image with a given CaptionBackend. Unlike the
    tagger's finished signal (list, str), this emits (str, str) -- caption
    text, error message -- since a caption is one block of free text, not a
    list of tags."""
    finished = pyqtSignal(str, str)
    progress = pyqtSignal(str)

    def __init__(self, image_path, backend, instruction):
        super().__init__()
        self.image_path = image_path
        self.backend = backend
        self.instruction = instruction

    def run(self):
        try:
            self.progress.emit("Loading captioning model...")
            self.backend.load(progress_cb=self.progress.emit)
            self.progress.emit("Generating caption...")

            def _cancel_check():
                return self.isInterruptionRequested()

            try:
                caption_text = self.backend.caption(self.image_path, self.instruction, cancel_check=_cancel_check)
            except _Cancelled:
                self.finished.emit("", "Cancelled")
                return

            # Same reasoning as the batch worker: caption() may finish right
            # as cancellation is requested, or the backend may only be able
            # to detect cancellation pre-flight -- re-check before reporting
            # success.
            if self.isInterruptionRequested():
                self.finished.emit("", "Cancelled")
                return

            self.finished.emit(caption_text, "")
        except ImageReadError as e:
            self.finished.emit("", str(e))
        except Exception as e:
            traceback.print_exc()
            self.finished.emit("", str(e))
