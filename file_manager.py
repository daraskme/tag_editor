import os
import glob

SUPPORTED_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}

class FileManager:
    def __init__(self):
        self.folder_path = ""
        self.all_image_files = []
        self.image_files = [] # This will hold filtered files
        self.current_index = -1

    def load_folder(self, path):
        self.folder_path = path
        self.all_image_files = []
        
        if not os.path.exists(path):
            return

        # Escape path for glob to handle special characters like '[' or ']'
        escaped_path = glob.escape(path)
        
        # Get all files and filter by supported extensions strictly
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
        """Filter images by tag query (case-insensitive)"""
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

    def get_all_unique_tags(self):
        """Aggregate all tags from all files in the current folder for autocomplete"""
        unique_tags = set()
        for img_path in self.all_image_files:
            tags = self.read_tags(img_path)
            for tag in tags:
                unique_tags.add(tag)
        return sorted(list(unique_tags))

    def get_current_image_path(self):
        if 0 <= self.current_index < len(self.image_files):
            return self.image_files[self.current_index]
        return None

    def get_text_file_path(self, image_path):
        if not image_path:
            return None
        base, _ = os.path.splitext(image_path)
        return base + ".txt"

    def read_tags(self, image_path):
        txt_path = self.get_text_file_path(image_path)
        if not txt_path or not os.path.exists(txt_path):
            return []
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return []
                tags = [tag.strip() for tag in content.split(',') if tag.strip()]
                return tags
        except Exception:
            return []

    def save_tags(self, image_path, tags):
        txt_path = self.get_text_file_path(image_path)
        if not txt_path or os.path.isdir(txt_path):
            return False
            
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(", ".join(tags))
            return True
        except Exception:
            return False

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
                self.save_tags(img_path, tags)
                count += 1
        return count

    def remove_tag_from_all(self, tag):
        count = 0
        for img_path in self.all_image_files:
            tags = self.read_tags(img_path)
            if tag in tags:
                tags.remove(tag)
                self.save_tags(img_path, tags)
                count += 1
        return count

    def replace_tag_in_all(self, old_tag, new_tag):
        count = 0
        new_tag = new_tag.strip() if new_tag else ""
        for img_path in self.all_image_files:
            tags = self.read_tags(img_path)
            if old_tag in tags:
                idx = tags.index(old_tag)
                if not new_tag or new_tag in tags:
                    tags.pop(idx)
                else:
                    tags[idx] = new_tag
                self.save_tags(img_path, tags)
                count += 1
        return count
