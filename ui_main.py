# pyre-ignore-all-errors[21]
import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QSplitter, QScrollArea, QLineEdit, QFileDialog, QMessageBox, QMenuBar, QInputDialog, QSizePolicy, QComboBox
)
from PyQt6.QtGui import QPixmap, QAction
from PyQt6.QtCore import Qt, QSize
from ui_components import FlowLayout, TagButton, ClickableImageLabel, FlowContainer
from file_manager import FileManager
from ai_tagger import WDTaggerWorker, Florence2Worker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Tag Editor")
        self.resize(1200, 800)
        self.file_manager = FileManager()
        
        self.setup_ui()
        self.setup_menu()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Splitter to divide image area and tag area
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)

        # Left Side (Image Viewer)
        self.left_widget = QWidget()
        left_layout = QVBoxLayout(self.left_widget)
        
        self.image_label = ClickableImageLabel()
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("background-color: #2c3e50; border-radius: 5px;")
        
        self.filename_label = QLabel("No image loaded")
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filename_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        
        left_layout.addWidget(self.image_label, stretch=1)
        left_layout.addWidget(self.filename_label)
        left_layout.addLayout(nav_layout)
        
        # Right Side (Tags Editor)
        self.right_widget = QWidget()
        right_layout = QVBoxLayout(self.right_widget)
        
        tags_header = QLabel("Image Tags")
        tags_header.setStyleSheet("font-size: 18px; font-weight: bold;")
        right_layout.addWidget(tags_header)
        
        # Tags Area (Scrollable Flow Layout)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tags_container = FlowContainer()
        # Crucial for FlowLayout wrapping in QScrollArea: the container must expand horizontally, but have a fixed minimum to let it shrink properly
        self.tags_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.tags_layout = FlowLayout()
        self.tags_container.setLayout(self.tags_layout)
        self.scroll_area.setWidget(self.tags_container)
        right_layout.addWidget(self.scroll_area, stretch=1)
        
        # AI Integration Area
        ai_layout = QHBoxLayout()
        self.wd_btn = QPushButton("Run WD Tagger")
        self.wd_btn.setStyleSheet("background-color: #9b59b6; color: white;")
        self.wd_btn.clicked.connect(self.run_wd_tagger)
        
        self.florence_btn = QPushButton("Run Florence-2")
        self.florence_btn.setStyleSheet("background-color: #e67e22; color: white;")
        self.florence_btn.clicked.connect(self.run_florence2)
        
        ai_layout.addWidget(self.wd_btn)
        ai_layout.addWidget(self.florence_btn)
        right_layout.addLayout(ai_layout)
        
        # Tag Input Area
        input_layout = QHBoxLayout()
        self.tag_input = QLineEdit()
        self.tag_input.setPlaceholderText("Enter new tag...")
        self.tag_input.returnPressed.connect(self.add_tag)
        self.add_btn = QPushButton("Add Tag")
        self.add_btn.clicked.connect(self.add_tag)
        input_layout.addWidget(self.tag_input)
        input_layout.addWidget(self.add_btn)
        right_layout.addLayout(input_layout)
        
        # Batch Operations
        batch_layout = QHBoxLayout()
        self.add_all_btn = QPushButton("Add to All")
        self.position_combo = QComboBox()
        self.position_combo.addItems(["Add to End", "Add to Start"])
        self.remove_all_btn = QPushButton("Remove from All")
        self.add_all_btn.clicked.connect(self.add_tag_to_all)
        self.remove_all_btn.clicked.connect(self.remove_tag_from_all)
        batch_layout.addWidget(self.add_all_btn)
        batch_layout.addWidget(self.position_combo)
        batch_layout.addWidget(self.remove_all_btn)
        right_layout.addLayout(batch_layout)
        
        self.splitter.addWidget(self.left_widget)
        self.splitter.addWidget(self.right_widget)
        self.splitter.setSizes([800, 400])

    def setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Folder", self)
        open_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_action)

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            self.file_manager.load_folder(folder_path)
            self.update_ui()

    def update_ui(self):
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            self.image_label.clear()
            self.filename_label.setText("No image loaded")
            self.clear_tags()
            return
            
        self.filename_label.setText(os.path.basename(img_path))
        self.load_image_pixmap(img_path)
        self.load_tags()

    def load_image_pixmap(self, img_path=None):
        if img_path is None:
            img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
            
        # Load and scale image
        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            # Scale to fit label while keeping aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def load_tags(self):
        self.clear_tags()
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
            
        tags = self.file_manager.read_tags(img_path)
        for tag in tags:
            btn = TagButton(tag)
            btn.deleted.connect(self.remove_tag)
            btn.edit_requested.connect(self.edit_tag)
            self.tags_layout.addWidget(btn)

    def clear_tags(self):
        for i in reversed(range(self.tags_layout.count())):
            item = self.tags_layout.takeAt(i)
            if item and item.widget():
                item.widget().deleteLater()

    def add_tag(self):
        tag = self.tag_input.text().strip()
        if not tag:
            return
            
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
            
        tags = self.file_manager.read_tags(img_path)
        if tag not in tags:
            tags.append(tag)
            self.file_manager.save_tags(img_path, tags)
            self.tag_input.clear()
            self.load_tags()

    def edit_tag(self, old_tag):
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return

        new_tag, ok = QInputDialog.getText(self, "Edit Tag", "Enter new tag:", QLineEdit.EchoMode.Normal, old_tag)
        if ok and new_tag:
            new_tag = new_tag.strip()
            if not new_tag or new_tag == old_tag:
                return
            
            tags = self.file_manager.read_tags(img_path)
            if old_tag in tags:
                # Replace the old tag with the new one at the same position
                idx = tags.index(old_tag)
                
                # Check if the new tag already exists elsewhere
                if new_tag in tags and tags.index(new_tag) != idx:
                    QMessageBox.warning(self, "Warning", "This tag already exists.")
                    return
                    
                tags[idx] = new_tag
                self.file_manager.save_tags(img_path, tags)
                self.load_tags()

    def remove_tag(self, tag):
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
            
        tags = self.file_manager.read_tags(img_path)
        if tag in tags:
            tags.remove(tag)
            self.file_manager.save_tags(img_path, tags)
            # TagButton deletes itself via deleteLater in its on_click method
            # We just need to remove it from layout visually or let layout handle it
            # To be clean, reloading tags ensures correctness layout
            self.load_tags()

    def next_image(self):
        if self.file_manager.next_image():
            self.update_ui()

    def prev_image(self):
        if self.file_manager.prev_image():
            self.update_ui()

    def add_tag_to_all(self):
        tag = self.tag_input.text().strip()
        if not tag:
            QMessageBox.warning(self, "Warning", "Please enter a tag to add to all files.")
            return
            
        position = "start" if self.position_combo.currentIndex() == 1 else "end"
        action_text = "at the beginning of" if position == "start" else "to the end of"
        reply = QMessageBox.question(self, 'Confirm', f"Add '{tag}' {action_text} all text files in this folder?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            count = self.file_manager.add_tag_to_all(tag, position)
            QMessageBox.information(self, "Success", f"Added '{tag}' to {count} files.")
            self.tag_input.clear()
            self.load_tags()

    def remove_tag_from_all(self):
        tag = self.tag_input.text().strip()
        if not tag:
            QMessageBox.warning(self, "Warning", "Please enter a tag to remove from all files.")
            return
            
        reply = QMessageBox.question(self, 'Confirm', f"Remove '{tag}' from all text files in this folder?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            count = self.file_manager.remove_tag_from_all(tag)
            QMessageBox.information(self, "Success", f"Removed '{tag}' from {count} files.")
            self.tag_input.clear()
            self.load_tags()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-scale image on resize without reloading tags
        self.load_image_pixmap()

    def run_wd_tagger(self):
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
            
        self.wd_btn.setEnabled(False)
        self.florence_btn.setEnabled(False)
        self.statusBar().showMessage("Initializing WD Tagger...")
        
        self.wd_worker = WDTaggerWorker(img_path)
        self.wd_worker.progress.connect(self.update_status)
        self.wd_worker.finished.connect(self.on_ai_finished)
        self.wd_worker.start()

    def run_florence2(self):
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
            
        self.wd_btn.setEnabled(False)
        self.florence_btn.setEnabled(False)
        self.statusBar().showMessage("Initializing Florence-2...")
        
        self.flo_worker = Florence2Worker(img_path)
        self.flo_worker.progress.connect(self.update_status)
        self.flo_worker.finished.connect(self.on_ai_finished)
        self.flo_worker.start()

    def update_status(self, msg):
        self.statusBar().showMessage(msg)

    def on_ai_finished(self, new_tags, error_msg):
        self.wd_btn.setEnabled(True)
        self.florence_btn.setEnabled(True)
        self.statusBar().clearMessage()
        
        if error_msg:
            QMessageBox.critical(self, "AI Error", error_msg)
            return
            
        if not new_tags:
            QMessageBox.information(self, "AI Tagger", "No tags resulted from model.")
            return
            
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
            
        current_tags = self.file_manager.read_tags(img_path)
        added_count = 0
        
        for tag in new_tags:
            tag = str(tag).strip()
            if tag and tag not in current_tags:
                current_tags.append(tag)
                added_count += 1
                
        if added_count > 0:
            self.file_manager.save_tags(img_path, current_tags)
            self.load_tags()
            self.statusBar().showMessage(f"Added {added_count} new tags.", 3000)
        else:
            self.statusBar().showMessage("No new unique tags identified.", 3000)
