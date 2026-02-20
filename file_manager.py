import os
import glob

SUPPORTED_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}

class FileManager:
    def __init__(self):
        self.folder_path = ""
        self.image_files = []
        self.current_index = -1

    def load_folder(self, path):
        self.folder_path = path
        self.image_files = []
        
        if not os.path.exists(path):
            return

        for ext in SUPPORTED_IMAGE_EXTS:
            self.image_files.extend(glob.glob(os.path.join(path, f"*{ext}")))
            self.image_files.extend(glob.glob(os.path.join(path, f"*{ext.upper()}")))
        
        self.image_files.sort()
        if self.image_files:
            self.current_index = 0
        else:
            self.current_index = -1

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
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return []
            tags = [tag.strip() for tag in content.split(',') if tag.strip()]
            return tags

    def save_tags(self, image_path, tags):
        txt_path = self.get_text_file_path(image_path)
        if not txt_path:
            return False
            
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(", ".join(tags))
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
        for img_path in self.image_files:
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
        for img_path in self.image_files:
            tags = self.read_tags(img_path)
            if tag in tags:
                tags.remove(tag)
                self.save_tags(img_path, tags)
                count += 1
        return count
