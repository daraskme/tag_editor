import sys
import os
import traceback
from PyQt6.QtGui import QPixmap, QAction, QIntValidator, QGuiApplication
from PyQt6.QtCore import Qt, QSize, QStringListModel
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSplitter, QScrollArea, QLineEdit, QFileDialog, QMessageBox,
    QMenuBar, QInputDialog, QSizePolicy, QComboBox, QProgressBar,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
)
from ui_components import FlowLayout, TagButton, ClickableImageLabel, FlowContainer
from file_manager import FileManager
from ai_tagger import PixAITaggerWorker, Florence2Worker, BatchPixAITaggerWorker, BatchFlorence2Worker

COLORS = {
    "bg": "#1e1e1e",
    "sidebar": "#252526",
    "primary": "#007acc",
    "primary_hover": "#1e90ff",
    "accent": "#9b59b6",
    "accent_hover": "#8e44ad",
    "orange": "#e67e22",
    "orange_hover": "#d35400",
    "danger": "#c0392b",
    "danger_hover": "#a93226",
    "text": "#cccccc",
    "inactive": "#3e3e42",
}


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically instead of lexicographically."""
    def __lt__(self, other):
        try:
            return int(self.text()) < int(other.text())
        except ValueError:
            return super().__lt__(other)



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PixAI Tag Editor Premium")
        self.resize(1200, 800)
        self.file_manager = FileManager()
        self.tag_clipboard = []

        self.setup_ui()
        self.apply_dark_theme()
        self.setup_menu()
        self.setAcceptDrops(True)

    def apply_dark_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {COLORS['bg']};
                color: {COLORS['text']};
                font-family: 'Segoe UI', sans-serif;
            }}
            QPushButton {{
                background-color: {COLORS['inactive']};
                border: none;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary']};
            }}
            QLineEdit, QComboBox, QSpinBox {{
                background-color: {COLORS['inactive']};
                border: 1px solid #333;
                border-radius: 3px;
                padding: 5px;
                color: white;
            }}
            QLabel {{
                color: {COLORS['text']};
            }}
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QSplitter::handle {{
                background-color: #333;
            }}
            QProgressBar {{
                border: 1px solid #333;
                border-radius: 4px;
                text-align: center;
                background-color: {COLORS['inactive']};
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
            }}
            QTabWidget::pane {{
                border: 1px solid #333;
                background-color: {COLORS['bg']};
            }}
            QTabBar::tab {{
                background-color: {COLORS['inactive']};
                color: {COLORS['text']};
                padding: 6px 14px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['primary']};
                color: white;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: #4e4e52;
            }}
            QTableWidget {{
                background-color: {COLORS['sidebar']};
                gridline-color: #333;
                border: none;
            }}
            QTableWidget::item {{
                padding: 4px;
            }}
            QTableWidget::item:selected {{
                background-color: {COLORS['primary']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['inactive']};
                color: {COLORS['text']};
                padding: 5px;
                border: none;
                border-right: 1px solid #333;
                border-bottom: 1px solid #333;
            }}
        """)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)

        # ── Left Side: Image Viewer ──────────────────────────────────────────
        self.left_widget = QWidget()
        self.left_widget.setStyleSheet(f"background-color: {COLORS['sidebar']};")
        left_layout = QVBoxLayout(self.left_widget)

        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("🔍 Filter:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by tag...")
        self.search_input.textChanged.connect(self.filter_images)
        search_layout.addWidget(self.search_input)
        left_layout.addLayout(search_layout)

        self.image_label = ClickableImageLabel()
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("background-color: #111; border: 1px solid #333; border-radius: 8px;")

        info_layout = QHBoxLayout()
        self.filename_label = QLabel("No image loaded")
        self.filename_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #aaa;")
        self.counter_label = QLabel("0 / 0")
        self.counter_label.setStyleSheet("color: #007acc; font-weight: bold;")
        info_layout.addWidget(self.filename_label)
        info_layout.addStretch()
        info_layout.addWidget(self.counter_label)

        jump_layout = QHBoxLayout()
        jump_layout.addWidget(QLabel("Jump to:"))
        self.jump_input = QLineEdit()
        self.jump_input.setFixedWidth(50)
        self.jump_input.setValidator(QIntValidator(1, 999999))
        self.jump_input.returnPressed.connect(self.jump_to_image)
        jump_layout.addWidget(self.jump_input)

        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Previous")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.setShortcut("Left")
        self.next_btn.setShortcut("Right")
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addLayout(jump_layout)
        nav_layout.addWidget(self.next_btn)

        left_layout.addLayout(info_layout)
        left_layout.addWidget(self.image_label, stretch=1)
        left_layout.addLayout(nav_layout)

        # ── Right Side: Tabbed Tag Editor ────────────────────────────────────
        self.right_widget = QWidget()
        right_layout = QVBoxLayout(self.right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.tags_tab_widget = QTabWidget()
        self.tags_tab_widget.currentChanged.connect(self._on_tab_changed)
        right_layout.addWidget(self.tags_tab_widget)

        # ── Tab 1: Image Tags ────────────────────────────────────────────────
        image_tab = QWidget()
        image_tab_layout = QVBoxLayout(image_tab)
        image_tab_layout.setContentsMargins(4, 6, 4, 4)

        tags_header = QLabel("Image Tags")
        tags_header.setStyleSheet("font-size: 16px; font-weight: bold;")
        image_tab_layout.addWidget(tags_header)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tags_container = FlowContainer()
        self.tags_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        self.tags_layout = FlowLayout()
        self.tags_container.setLayout(self.tags_layout)
        self.scroll_area.setWidget(self.tags_container)
        image_tab_layout.addWidget(self.scroll_area, stretch=1)

        # AI controls
        ai_layout = QVBoxLayout()
        ai_single_layout = QHBoxLayout()
        self.pixai_btn = QPushButton("Run PixAI Tagger")
        self.pixai_btn.setStyleSheet("background-color: #9b59b6; color: white; padding: 5px;")
        self.pixai_btn.clicked.connect(self.run_pixai_tagger)
        self.florence_btn = QPushButton("Run Florence-2")
        self.florence_btn.setStyleSheet("background-color: #e67e22; color: white; padding: 5px;")
        self.florence_btn.clicked.connect(self.run_florence2)
        ai_single_layout.addWidget(self.pixai_btn)
        ai_single_layout.addWidget(self.florence_btn)
        ai_layout.addLayout(ai_single_layout)

        flo_task_layout = QHBoxLayout()
        flo_task_layout.addWidget(QLabel("Florence-2 Task:"))
        self.flo_task_combo = QComboBox()
        self.flo_task_combo.addItems([
            "<DETAILED_CAPTION>", "<CAPTION>", "<OCR>",
            "<OCR_WITH_REGION>", "<REGION_PROPOSAL>",
        ])
        flo_task_layout.addWidget(self.flo_task_combo)
        ai_layout.addLayout(flo_task_layout)

        batch_ai_layout = QHBoxLayout()
        self.batch_pixai_btn = QPushButton("Batch Tag All (PixAI)")
        self.batch_pixai_btn.setStyleSheet("background-color: #8e44ad; color: white; padding: 5px;")
        self.batch_pixai_btn.clicked.connect(self.run_batch_pixai)
        self.batch_florence_btn = QPushButton("Batch Caption All (Florence-2)")
        self.batch_florence_btn.setStyleSheet("background-color: #d35400; color: white; padding: 5px;")
        self.batch_florence_btn.clicked.connect(self.run_batch_florence)
        batch_ai_layout.addWidget(self.batch_pixai_btn)
        batch_ai_layout.addWidget(self.batch_florence_btn)
        ai_layout.addLayout(batch_ai_layout)
        image_tab_layout.addLayout(ai_layout)

        # Batch progress (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        image_tab_layout.addWidget(self.progress_bar)

        self.batch_status_label = QLabel()
        self.batch_status_label.setVisible(False)
        image_tab_layout.addWidget(self.batch_status_label)

        self.cancel_batch_btn = QPushButton("🛑 Cancel Batch")
        self.cancel_batch_btn.setStyleSheet(f"background-color: {COLORS['danger']}; color: white; font-weight: bold;")
        self.cancel_batch_btn.setVisible(False)
        self.cancel_batch_btn.clicked.connect(self.cancel_batch)
        image_tab_layout.addWidget(self.cancel_batch_btn)

        # Tag input
        input_layout = QHBoxLayout()
        self.tag_input = QLineEdit()
        self.tag_input.setPlaceholderText("Enter new tag...")
        self.tag_input.returnPressed.connect(self.add_tag)
        self.add_btn = QPushButton("Add Tag")
        self.add_btn.clicked.connect(self.add_tag)
        input_layout.addWidget(self.tag_input)
        input_layout.addWidget(self.add_btn)
        image_tab_layout.addLayout(input_layout)

        # Copy/Paste
        copy_paste_layout = QHBoxLayout()
        self.copy_btn = QPushButton("📋 Copy Tags")
        self.paste_btn = QPushButton("📋 Paste Tags")
        self.copy_btn.clicked.connect(self.copy_tags)
        self.paste_btn.clicked.connect(self.paste_tags)
        copy_paste_layout.addWidget(self.copy_btn)
        copy_paste_layout.addWidget(self.paste_btn)
        image_tab_layout.addLayout(copy_paste_layout)

        # Batch tag operations
        add_rem_layout = QHBoxLayout()
        self.add_all_btn = QPushButton("Add to All")
        self.position_combo = QComboBox()
        self.position_combo.addItems(["Add to End", "Add to Start"])
        self.remove_all_btn = QPushButton("Remove from All")
        self.add_all_btn.clicked.connect(self.add_tag_to_all)
        self.remove_all_btn.clicked.connect(self.remove_tag_from_all)
        add_rem_layout.addWidget(self.add_all_btn)
        add_rem_layout.addWidget(self.position_combo)
        add_rem_layout.addWidget(self.remove_all_btn)
        image_tab_layout.addLayout(add_rem_layout)

        self.clear_all_btn = QPushButton("Clear Current Image Tags")
        self.clear_all_btn.setStyleSheet(f"background-color: {COLORS['danger']}; color: white;")
        self.clear_all_btn.clicked.connect(self.clear_current_tags)
        image_tab_layout.addWidget(self.clear_all_btn)

        self.tags_tab_widget.addTab(image_tab, "Image Tags")

        # ── Tab 2: All Tags ──────────────────────────────────────────────────
        all_tab = QWidget()
        all_tab_layout = QVBoxLayout(all_tab)
        all_tab_layout.setContentsMargins(4, 6, 4, 4)

        all_header = QLabel("All Tags in Folder")
        all_header.setStyleSheet("font-size: 16px; font-weight: bold;")
        all_tab_layout.addWidget(all_header)

        all_search_layout = QHBoxLayout()
        all_search_layout.addWidget(QLabel("🔍"))
        self.all_tags_search = QLineEdit()
        self.all_tags_search.setPlaceholderText("Filter tags...")
        self.all_tags_search.textChanged.connect(self._filter_all_tags_table)
        all_search_layout.addWidget(self.all_tags_search)

        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(self.refresh_all_tags_table)
        all_search_layout.addWidget(refresh_btn)
        all_tab_layout.addLayout(all_search_layout)

        self.all_tags_table = QTableWidget()
        self.all_tags_table.setColumnCount(2)
        self.all_tags_table.setHorizontalHeaderLabels(["Tag", "Images"])
        self.all_tags_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.all_tags_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.all_tags_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.all_tags_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.all_tags_table.setSortingEnabled(True)
        self.all_tags_table.verticalHeader().setVisible(False)
        self.all_tags_table.setAlternatingRowColors(True)
        self.all_tags_table.setStyleSheet(
            "QTableWidget { alternate-background-color: #252526; }"
        )
        all_tab_layout.addWidget(self.all_tags_table, stretch=1)

        self.remove_selected_btn = QPushButton("🗑 Remove Selected Tag from All Files")
        self.remove_selected_btn.setStyleSheet(
            f"background-color: {COLORS['danger']}; color: white; font-weight: bold; padding: 7px;"
        )
        self.remove_selected_btn.clicked.connect(self._remove_selected_tags)
        all_tab_layout.addWidget(self.remove_selected_btn)

        self.tags_tab_widget.addTab(all_tab, "All Tags")

        self.splitter.addWidget(self.left_widget)
        self.splitter.addWidget(self.right_widget)
        self.splitter.setSizes([750, 450])

    # ── Menu ─────────────────────────────────────────────────────────────────

    def setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open Folder", self)
        open_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_action)

    # ── Folder / Navigation ───────────────────────────────────────────────────

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            self.file_manager.load_folder(folder_path)
            self.update_ui()

    def filter_images(self, text):
        count = self.file_manager.apply_filter(text)
        self.update_ui()
        if text:
            self.statusBar().showMessage(f"Found {count} images with tag '{text}'", 2000)

    def update_ui(self):
        img_path = self.file_manager.get_current_image_path()
        total = len(self.file_manager.image_files)
        if not img_path:
            self.image_label.clear()
            self.filename_label.setText("No image loaded")
            self.counter_label.setText("0 / 0")
            self.clear_tags()
            return

        current = self.file_manager.current_index + 1
        self.filename_label.setText(os.path.basename(img_path))
        self.counter_label.setText(f"{current} / {total}")
        self.load_image_pixmap(img_path)
        self.load_tags()

    def load_image_pixmap(self, img_path=None):
        if img_path is None:
            img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)

    # ── Image Tags tab ────────────────────────────────────────────────────────

    def load_tags(self):
        self.clear_tags()
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
        for tag in self.file_manager.read_tags(img_path):
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
                idx = tags.index(old_tag)
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
            self.load_tags()

    def next_image(self):
        if self.file_manager.next_image():
            self.update_ui()

    def prev_image(self):
        if self.file_manager.prev_image():
            self.update_ui()

    def jump_to_image(self):
        text = self.jump_input.text()
        if not text:
            return
        idx = int(text) - 1
        if 0 <= idx < len(self.file_manager.image_files):
            self.file_manager.current_index = idx
            self.update_ui()
            self.jump_input.clear()
        else:
            QMessageBox.warning(self, "Invalid Index", f"Please enter a number between 1 and {len(self.file_manager.image_files)}.")

    def clear_current_tags(self):
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
        reply = QMessageBox.question(self, "Confirm Clear", "Clear all tags for this image?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.file_manager.save_tags(img_path, [])
            self.load_tags()

    def copy_tags(self):
        img_path = self.file_manager.get_current_image_path()
        if img_path:
            self.tag_clipboard = self.file_manager.read_tags(img_path)
            self.statusBar().showMessage(f"Copied {len(self.tag_clipboard)} tags", 2000)

    def paste_tags(self):
        img_path = self.file_manager.get_current_image_path()
        if img_path and self.tag_clipboard:
            current_tags = self.file_manager.read_tags(img_path)
            added = False
            for tag in self.tag_clipboard:
                if tag not in current_tags:
                    current_tags.append(tag)
                    added = True
            if added:
                self.file_manager.save_tags(img_path, current_tags)
                self.load_tags()
                self.statusBar().showMessage(f"Pasted {len(self.tag_clipboard)} tags", 2000)

    def add_tag_to_all(self):
        tag = self.tag_input.text().strip()
        if not tag:
            QMessageBox.warning(self, "Warning", "Please enter a tag to add to all files.")
            return
        position = "start" if self.position_combo.currentIndex() == 1 else "end"
        action_text = "at the beginning of" if position == "start" else "to the end of"
        reply = QMessageBox.question(self, "Confirm",
                                     f"Add '{tag}' {action_text} all text files?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
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
        reply = QMessageBox.question(self, "Confirm",
                                     f"Remove '{tag}' from all text files?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            count = self.file_manager.remove_tag_from_all(tag)
            QMessageBox.information(self, "Success", f"Removed '{tag}' from {count} files.")
            self.tag_input.clear()
            self.load_tags()

    # ── All Tags tab ──────────────────────────────────────────────────────────

    def _on_tab_changed(self, index: int):
        if index == 1:
            self.refresh_all_tags_table()

    def refresh_all_tags_table(self):
        filter_text = self.all_tags_search.text()
        tag_counts = self.file_manager.get_tag_counts()
        self._populate_all_tags_table(tag_counts, filter_text)

    def _filter_all_tags_table(self, text: str):
        tag_counts = self.file_manager.get_tag_counts()
        self._populate_all_tags_table(tag_counts, text)

    def _populate_all_tags_table(self, tag_counts: list, filter_text: str = ""):
        self.all_tags_table.setSortingEnabled(False)
        self.all_tags_table.setRowCount(0)

        f = filter_text.lower()
        for tag, count in tag_counts:
            if f and f not in tag.lower():
                continue
            row = self.all_tags_table.rowCount()
            self.all_tags_table.insertRow(row)

            tag_item = QTableWidgetItem(tag)
            self.all_tags_table.setItem(row, 0, tag_item)

            count_item = NumericTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.all_tags_table.setItem(row, 1, count_item)

        self.all_tags_table.setSortingEnabled(True)
        self.all_tags_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)

    def _remove_selected_tags(self):
        selected_rows = self.all_tags_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Warning", "削除するタグを選択してください。")
            return

        tags = [self.all_tags_table.item(idx.row(), 0).text() for idx in selected_rows]
        tag_list = ", ".join(f"'{t}'" for t in tags)
        reply = QMessageBox.question(
            self, "Confirm",
            f"全ファイルから {tag_list} を削除しますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        total_removed = 0
        for tag in tags:
            total_removed += self.file_manager.remove_tag_from_all(tag)

        self.statusBar().showMessage(f"{len(tags)}個のタグを削除しました。({total_removed}件のファイルを更新)", 3000)
        self.load_tags()

        # Remove deleted rows from table without full rebuild
        self.all_tags_table.setSortingEnabled(False)
        for tag in tags:
            for r in range(self.all_tags_table.rowCount() - 1, -1, -1):
                item = self.all_tags_table.item(r, 0)
                if item and item.text() == tag:
                    self.all_tags_table.removeRow(r)
                    break
        self.all_tags_table.setSortingEnabled(True)

    # ── AI Tagging ────────────────────────────────────────────────────────────

    def set_ai_buttons_enabled(self, enabled: bool):
        self.pixai_btn.setEnabled(enabled)
        self.florence_btn.setEnabled(enabled)
        self.batch_pixai_btn.setEnabled(enabled)
        self.batch_florence_btn.setEnabled(enabled)
        self.add_all_btn.setEnabled(enabled)
        self.remove_all_btn.setEnabled(enabled)

    def run_pixai_tagger(self):
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
        self.set_ai_buttons_enabled(False)
        self.statusBar().showMessage("Initializing PixAI Tagger...")
        self.pixai_worker = PixAITaggerWorker(img_path)
        self.pixai_worker.progress.connect(self.update_status)
        self.pixai_worker.finished.connect(self.on_ai_finished)
        self.pixai_worker.start()

    def run_batch_pixai(self):
        if not self.file_manager.image_files:
            QMessageBox.warning(self, "Warning", "No images loaded in the folder.")
            return
        reply = QMessageBox.question(self, "Confirm",
                                     f"Run PixAI Tagger on all {len(self.file_manager.image_files)} images?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.set_ai_buttons_enabled(False)
            self.progress_bar.setVisible(True)
            self.batch_status_label.setVisible(True)
            self.cancel_batch_btn.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(len(self.file_manager.image_files))
            self.batch_pixai_worker = BatchPixAITaggerWorker(self.file_manager, self.file_manager.image_files)
            self.batch_pixai_worker.progress.connect(self.update_batch_progress)
            self.batch_pixai_worker.finished.connect(self.on_batch_finished)
            self.batch_pixai_worker.start()

    def run_florence2(self):
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
        task_prompt = self.flo_task_combo.currentText()
        self.set_ai_buttons_enabled(False)
        self.statusBar().showMessage(f"Initializing Florence-2 ({task_prompt})...")
        self.flo_worker = Florence2Worker(img_path, task_prompt=task_prompt)
        self.flo_worker.progress.connect(self.update_status)
        self.flo_worker.finished.connect(self.on_ai_finished)
        self.flo_worker.start()

    def run_batch_florence(self):
        if not self.file_manager.image_files:
            QMessageBox.warning(self, "Warning", "No images loaded in the folder.")
            return
        task_prompt = self.flo_task_combo.currentText()
        reply = QMessageBox.question(self, "Confirm",
                                     f"Run Florence-2 ({task_prompt}) on all {len(self.file_manager.image_files)} images?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.set_ai_buttons_enabled(False)
            self.progress_bar.setVisible(True)
            self.batch_status_label.setVisible(True)
            self.cancel_batch_btn.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(len(self.file_manager.image_files))
            self.batch_flo_worker = BatchFlorence2Worker(
                self.file_manager, self.file_manager.image_files, task_prompt=task_prompt
            )
            self.batch_flo_worker.progress.connect(self.update_batch_progress)
            self.batch_flo_worker.finished.connect(self.on_batch_finished)
            self.batch_flo_worker.start()

    def update_status(self, msg: str):
        self.statusBar().showMessage(msg)

    def cancel_batch(self):
        if hasattr(self, "batch_pixai_worker") and self.batch_pixai_worker.isRunning():
            self.batch_pixai_worker.requestInterruption()
            self.statusBar().showMessage("Cancelling PixAI batch...")
        if hasattr(self, "batch_flo_worker") and self.batch_flo_worker.isRunning():
            self.batch_flo_worker.requestInterruption()
            self.statusBar().showMessage("Cancelling Florence-2 batch...")
        self.cancel_batch_btn.setEnabled(False)

    def update_batch_progress(self, current: int, total: int, filename: str):
        self.progress_bar.setValue(current)
        self.batch_status_label.setText(f"Processing {current + 1}/{total}: {filename}")
        self.statusBar().showMessage(f"Batch Processing: {current + 1}/{total}...")

    def on_batch_finished(self, success_count: int, total: int, error_msg: str):
        self.set_ai_buttons_enabled(True)
        self.progress_bar.setVisible(False)
        self.batch_status_label.setVisible(False)
        self.cancel_batch_btn.setVisible(False)
        self.cancel_batch_btn.setEnabled(True)
        self.statusBar().clearMessage()
        if error_msg:
            QMessageBox.critical(self, "Batch Error", error_msg)
        else:
            QMessageBox.information(self, "Batch Complete",
                                    f"Successfully processed {success_count} out of {total} images.")
        self.load_tags()

    def on_ai_finished(self, new_tags: list, error_msg: str):
        self.set_ai_buttons_enabled(True)
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

    # ── Resize / Drag-and-Drop ────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.load_image_pixmap()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        try:
            urls = event.mimeData().urls()
            if not urls:
                return
            file_path = urls[0].toLocalFile()
            if not file_path:
                return
            file_path = os.path.normpath(file_path)
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    self.statusBar().showMessage(f"Loading folder: {file_path}")
                    self.file_manager.load_folder(file_path)
                    self.update_ui()
                elif os.path.isfile(file_path):
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in {'.png', '.jpg', '.jpeg', '.webp'}:
                        folder_path = os.path.dirname(file_path)
                        self.statusBar().showMessage(f"Loading image from folder: {folder_path}")
                        self.file_manager.load_folder(folder_path)
                        for i, img in enumerate(self.file_manager.image_files):
                            if os.path.normpath(img) == file_path:
                                self.file_manager.current_index = i
                                break
                        self.update_ui()
            event.acceptProposedAction()
        except Exception as e:
            error_msg = f"Drag and Drop Error: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(self, "Crash Prevented", error_msg)
