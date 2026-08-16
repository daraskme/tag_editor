import os
import glob
import threading

SUPPORTED_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}

class FileManager:
    def __init__(self):
        self.folder_path = ""
        self.all_image_files = []
        self.image_files = []
        self.current_index = -1
        self._tag_counts_cache = None  # None means dirty
        # Per-file *raw text* cache, shared by the tag (comma-split) and
        # caption (free text) read/write paths, and with the batch-tagger/
        # batch-captioner QThread workers (via ai_tagger._merge_tags /
        # save_caption), hence the lock. A single cache -- not two separate
        # ones -- because tags and captions are mutually exclusive per image
        # and share the same .txt file; two independent caches could desync
        # when one feature overwrites what the other last wrote. Reflects
        # disk state only as of the last read/save/load_folder call for that
        # path -- editing a .txt outside the app requires reopening the
        # folder.
        self._text_cache = {}
        self._cache_lock = threading.Lock()
        self.last_error = None

    def load_folder(self, path):
        self.folder_path = path
        self.all_image_files = []
        self._tag_counts_cache = None
        with self._cache_lock:
            self._text_cache = {}

        if not os.path.exists(path):
            return

        escaped_path = glob.escape(path)
        all_files = glob.glob(os.path.join(escaped_path, "*"))
        for file in all_files:
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTS:
                self.all_image_files.append(file)

        self.all_image_files.sort()
        self.image_files = list(self.all_image_files)

        if self.image_files:
            self.current_index = 0
        else:
            self.current_index = -1

    def apply_filter(self, query):
        if not query:
            self.image_files = list(self.all_image_files)
        else:
            query = query.lower().strip()
            filtered = []
            for img_path in self.all_image_files:
                tags = [t.lower() for t in self.read_tags(img_path)]
                if query in tags:
                    filtered.append(img_path)
            self.image_files = filtered

        if self.image_files:
            self.current_index = 0
        else:
            self.current_index = -1
        return len(self.image_files)

    def get_tag_counts(self):
        """Returns cached list of (tag, count) sorted by count desc then alphabetically."""
        if self._tag_counts_cache is not None:
            return self._tag_counts_cache
        counts = {}
        for img_path in self.all_image_files:
            for tag in self.read_tags(img_path):
                counts[tag] = counts.get(tag, 0) + 1
        self._tag_counts_cache = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        return self._tag_counts_cache

    def get_current_image_path(self):
        if 0 <= self.current_index < len(self.image_files):
            return self.image_files[self.current_index]
        return None

    def get_text_file_path(self, image_path):
        if not image_path:
            return None
        base, _ = os.path.splitext(image_path)
        return base + ".txt"

    def _read_raw_text(self, image_path):
        """Read the sidecar .txt's full content as one stripped string,
        regardless of whether it holds comma-separated tags or a free-text
        caption. Cached (see _text_cache); never silently returns "" for a
        non-empty file whose encoding is unrecognized (the next save would
        wipe it out) -- falls back to utf-8 with lossy replacement instead.
        """
        txt_path = self.get_text_file_path(image_path)
        if not txt_path:
            return ""

        with self._cache_lock:
            cached = self._text_cache.get(image_path)
            if cached is not None:
                return cached

        if not os.path.exists(txt_path):
            content = ""
        else:
            try:
                with open(txt_path, 'rb') as f:
                    raw = f.read()
            except OSError:
                return ""
            content = None
            for enc in ('utf-8-sig', 'cp932'):
                try:
                    content = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                content = raw.decode('utf-8', errors='replace')
            content = content.strip()

        with self._cache_lock:
            self._text_cache[image_path] = content
        return content

    def _write_raw_text(self, image_path, text):
        """Atomically write `text` as the sidecar .txt's full content (temp
        file + os.replace), so a crash or interruption mid-write can't
        corrupt/truncate a file that an AI feature is about to overwrite
        wholesale."""
        txt_path = self.get_text_file_path(image_path)
        if not txt_path or os.path.isdir(txt_path):
            self.last_error = f"{txt_path}: is a directory" if txt_path else "no image path"
            return False
        tmp_path = txt_path + ".tmp"
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(text)
            os.replace(tmp_path, txt_path)
        except Exception as e:
            self.last_error = f"{txt_path}: {e}"
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            return False
        self.last_error = None
        with self._cache_lock:
            self._text_cache[image_path] = text
        return True

    def read_tags(self, image_path):
        content = self._read_raw_text(image_path)
        return [tag.strip() for tag in content.split(',') if tag.strip()] if content else []

    def read_caption(self, image_path):
        """Read the sidecar .txt as a free-text caption (no comma-splitting).
        Tags and captions are mutually exclusive per image and share the same
        file, so this simply returns whatever text is currently there."""
        return self._read_raw_text(image_path)

    def save_tags(self, image_path, tags):
        if not self._write_raw_text(image_path, ", ".join(tags)):
            return False
        self._tag_counts_cache = None  # invalidate cache
        return True

    def save_caption(self, image_path, text):
        """Write a free-text caption, replacing whatever tags/caption were
        previously in the sidecar .txt for this image."""
        if not self._write_raw_text(image_path, text.strip()):
            return False
        self._tag_counts_cache = None  # invalidate cache: old tag counts may
        # have included tags from this file's previous (non-caption) content
        return True

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            return True
        return False

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def add_tag_to_all(self, tag, position="end"):
        count = 0
        for img_path in self.all_image_files:
            tags = self.read_tags(img_path)
            if tag not in tags:
                if position == "start":
                    tags.insert(0, tag)
                else:
                    tags.append(tag)
                if self.save_tags(img_path, tags):
                    count += 1
        return count

    def remove_tag_from_all(self, tag):
        count = 0
        for img_path in self.all_image_files:
            tags = self.read_tags(img_path)
            if tag in tags:
                tags.remove(tag)
                if self.save_tags(img_path, tags):
                    count += 1
        return count

    def remove_tags_from_all(self, tags):
        """Remove every tag in `tags` from all files in a single pass.
        Returns the number of files actually updated."""
        tag_set = {t for t in tags if t}
        if not tag_set:
            return 0
        updated = 0
        for img_path in self.all_image_files:
            old_tags = self.read_tags(img_path)
            new_tags = [t for t in old_tags if t not in tag_set]
            if len(new_tags) != len(old_tags) and self.save_tags(img_path, new_tags):
                updated += 1
        return updated
