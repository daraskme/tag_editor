import os
import traceback
from PyQt6.QtGui import QPixmap, QAction, QIntValidator, QGuiApplication
from PyQt6.QtCore import Qt, QTimer, QLocale, QSettings, QSignalBlocker
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSplitter, QScrollArea, QLineEdit, QFileDialog, QMessageBox,
    QInputDialog, QSizePolicy, QComboBox, QProgressBar,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox,
    QTextEdit,
)
from ui_components import FlowLayout, TagButton, ClickableImageLabel, FlowContainer
from file_manager import FileManager
from ai_tagger import OppaiOracleWorker, BatchOppaiOracleWorker, oppai_model_info
from ai_captioner import (
    BF16Backend, GGUFBackend, BF16_VRAM_TIER, CAPTION_VRAM_TIERS,
    caption_model_info, default_caption_vram_tier,
    _BaseBatchCaptionWorker, _BaseSingleCaptionWorker,
)
from download_utils import _cache_dir

COLORS = {
    "bg": "#1e1e1e",
    "sidebar": "#252526",
    "primary": "#007acc",
    "primary_hover": "#1e90ff",
    "orange": "#e67e22",
    "orange_hover": "#d35400",
    "danger": "#c0392b",
    "danger_hover": "#a93226",
    "text": "#cccccc",
    "inactive": "#3e3e42",
}

CHARACTER_LORA_PROMPT = (
    "Describe this image in natural, flowing sentences for LoRA training. Focus on pose, "
    "facial expression, clothing, accessories, actions, camera angle/framing, background, and "
    "lighting. Do NOT describe the character's inherent physical traits (hair color/style, eye "
    "color, face shape, body type, species/race) -- assume those are already known and should "
    "not be re-described. Keep it concise and objective, 1-3 sentences."
)
STYLE_LORA_PROMPT = (
    "Describe this image in natural, flowing sentences for LoRA training. Focus on the subject "
    "matter and content: what characters/objects/scenery are present, and their pose, action, "
    "and composition. Do NOT describe the artistic style itself (art style, rendering "
    "technique, line quality, color palette, shading, medium) -- assume the style is already "
    "known and should not be re-described. Keep it concise and objective, 1-3 sentences."
)


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
        self.setWindowTitle("AI Tag Editor")
        self.resize(1200, 800)
        self.file_manager = FileManager()
        self.tag_clipboard = []
        self._src_pixmap = QPixmap()
        self._src_pixmap_path = None
        self._all_tags_sort = (1, Qt.SortOrder.DescendingOrder)
        self._settings = QSettings(
            os.path.join(_cache_dir(""), "ui_settings.ini"), QSettings.Format.IniFormat,
        )

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
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(300)
        self._filter_timer.timeout.connect(lambda: self.filter_images(self.search_input.text()))
        self.search_input.textChanged.connect(lambda _: self._filter_timer.start())
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
        validator = QIntValidator(1, 999999)
        loc = QLocale()
        loc.setNumberOptions(loc.numberOptions() | QLocale.NumberOption.RejectGroupSeparator)
        validator.setLocale(loc)
        self.jump_input.setValidator(validator)
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

        # ── Caption mode: tags (OppaiOracle) vs natural language (Qwen3.8) ──
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("キャプションモード:"))
        self.caption_mode_combo = QComboBox()
        self.caption_mode_combo.addItems([
            "タグキャプション (OppaiOracle)",
            "自然言語キャプション (Qwen3.8)",
        ])
        mode_layout.addWidget(self.caption_mode_combo, stretch=1)
        image_tab_layout.addLayout(mode_layout)

        # Tag mode panel (OppaiOracle)
        self.tag_mode_widget = QWidget()
        ai_layout = QVBoxLayout(self.tag_mode_widget)
        ai_layout.setContentsMargins(0, 0, 0, 0)

        run_btn_layout = QHBoxLayout()
        self.oppai_btn = QPushButton("タグキャプションを実行 (OppaiOracle)")
        self.oppai_btn.setStyleSheet("background-color: #e67e22; color: white; padding: 5px;")
        self.oppai_btn.clicked.connect(self.run_oppai_oracle)
        run_btn_layout.addWidget(self.oppai_btn)
        ai_layout.addLayout(run_btn_layout)

        oppai_opt_layout = QHBoxLayout()
        oppai_opt_layout.addWidget(QLabel("OppaiOracle Model:"))
        self.oppai_model_combo = QComboBox()
        self.oppai_model_combo.addItems(["V1.1", "V1"])
        oppai_opt_layout.addWidget(self.oppai_model_combo)
        oppai_opt_layout.addWidget(QLabel("Threshold:"))
        self.oppai_threshold_edit = QLineEdit("0.4")
        self.oppai_threshold_edit.setFixedWidth(60)
        oppai_opt_layout.addWidget(self.oppai_threshold_edit)
        oppai_opt_layout.addStretch(1)
        ai_layout.addLayout(oppai_opt_layout)

        self.oppai_status_label = QLabel()
        self.oppai_status_label.setWordWrap(True)
        self.oppai_status_label.setStyleSheet("color: #999; font-size: 11px;")
        ai_layout.addWidget(self.oppai_status_label)

        batch_btn_layout = QHBoxLayout()
        self.batch_oppai_btn = QPushButton("一括タグキャプション (OppaiOracle)")
        self.batch_oppai_btn.setStyleSheet("background-color: #d35400; color: white; padding: 5px;")
        self.batch_oppai_btn.clicked.connect(self.run_batch_oppai)
        batch_btn_layout.addWidget(self.batch_oppai_btn)
        ai_layout.addLayout(batch_btn_layout)

        image_tab_layout.addWidget(self.tag_mode_widget)

        # Natural-language mode panel (Qwen3.8)
        self.nl_mode_widget = QWidget()
        caption_layout = QVBoxLayout(self.nl_mode_widget)
        caption_layout.setContentsMargins(0, 0, 0, 0)

        caption_vram_layout = QHBoxLayout()
        caption_vram_layout.addWidget(QLabel("VRAM:"))
        self.caption_vram_tier_combo = QComboBox()
        self.caption_vram_tier_combo.addItems(list(CAPTION_VRAM_TIERS))
        caption_vram_layout.addWidget(self.caption_vram_tier_combo)
        caption_vram_layout.addStretch(1)
        caption_layout.addLayout(caption_vram_layout)

        self.caption_backend_status_label = QLabel()
        self.caption_backend_status_label.setWordWrap(True)
        self.caption_backend_status_label.setTextFormat(Qt.TextFormat.RichText)
        self.caption_backend_status_label.setOpenExternalLinks(True)
        self.caption_backend_status_label.setStyleSheet("color: #999; font-size: 11px;")
        caption_layout.addWidget(self.caption_backend_status_label)

        caption_style_layout = QHBoxLayout()
        caption_style_layout.addWidget(QLabel("Caption Style:"))
        self.caption_style_combo = QComboBox()
        self.caption_style_combo.addItems(["Character LoRA", "Style LoRA", "Custom"])
        caption_style_layout.addWidget(self.caption_style_combo)
        caption_style_layout.addStretch(1)
        caption_layout.addLayout(caption_style_layout)

        self.caption_instruction_edit = QTextEdit()
        self.caption_instruction_edit.setMaximumHeight(70)
        self.caption_instruction_edit.setPlainText(CHARACTER_LORA_PROMPT)
        caption_layout.addWidget(self.caption_instruction_edit)

        caption_run_btn_layout = QHBoxLayout()
        self.caption_btn = QPushButton("自然言語キャプションを実行 (Qwen3.8)")
        self.caption_btn.setStyleSheet(f"background-color: {COLORS['primary']}; color: white; padding: 5px;")
        self.caption_btn.clicked.connect(self.run_caption)
        caption_run_btn_layout.addWidget(self.caption_btn)
        caption_layout.addLayout(caption_run_btn_layout)

        caption_batch_btn_layout = QHBoxLayout()
        self.batch_caption_btn = QPushButton("一括自然言語キャプション (Qwen3.8)")
        self.batch_caption_btn.setStyleSheet(f"background-color: {COLORS['danger']}; color: white; padding: 5px;")
        self.batch_caption_btn.clicked.connect(self.run_batch_caption)
        caption_batch_btn_layout.addWidget(self.batch_caption_btn)
        caption_layout.addLayout(caption_batch_btn_layout)

        image_tab_layout.addWidget(self.nl_mode_widget)

        raw_preview_label = QLabel("Raw .txt Preview:")
        image_tab_layout.addWidget(raw_preview_label)
        self.raw_text_preview = QTextEdit()
        self.raw_text_preview.setReadOnly(True)
        self.raw_text_preview.setMaximumHeight(70)
        image_tab_layout.addWidget(self.raw_text_preview)

        self.caption_mode_combo.currentIndexChanged.connect(self._on_caption_mode_changed)
        self.caption_mode_combo.currentIndexChanged.connect(self._save_caption_settings)
        self.caption_vram_tier_combo.currentIndexChanged.connect(self._on_caption_vram_changed)
        self.caption_vram_tier_combo.currentIndexChanged.connect(self._save_caption_settings)
        self.oppai_model_combo.currentIndexChanged.connect(self._refresh_oppai_status_label)
        self.caption_style_combo.currentTextChanged.connect(self._on_caption_style_changed)
        self.caption_style_combo.currentTextChanged.connect(self._save_caption_settings)
        # Debounced, unlike the combo boxes above (whose changes are already
        # infrequent, deliberate clicks): textChanged fires per keystroke, and
        # _save_caption_settings ends with an explicit QSettings.sync() disk
        # flush, so connecting it directly would sync once per character
        # typed into a multi-sentence prompt. Reuses the same debounce
        # pattern as self._filter_timer (search box) elsewhere in this file.
        self._caption_settings_timer = QTimer(self)
        self._caption_settings_timer.setSingleShot(True)
        self._caption_settings_timer.setInterval(500)
        self._caption_settings_timer.timeout.connect(self._save_caption_settings)
        self.caption_instruction_edit.textChanged.connect(
            lambda: self._caption_settings_timer.start()
        )

        self._load_caption_settings()
        self._on_caption_mode_changed()

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

        # Copy tags that appear in N or more images
        copy_freq_layout = QHBoxLayout()
        copy_freq_layout.addWidget(QLabel("Min images:"))
        self.copy_min_count_spin = QSpinBox()
        self.copy_min_count_spin.setRange(1, 999999)
        self.copy_min_count_spin.setValue(2)
        self.copy_min_count_spin.setFixedWidth(80)
        copy_freq_layout.addWidget(self.copy_min_count_spin)
        self.copy_frequent_btn = QPushButton("📋 Copy Tags ≥ N Images")
        self.copy_frequent_btn.setStyleSheet(
            f"background-color: {COLORS['primary']}; color: white; font-weight: bold;"
        )
        self.copy_frequent_btn.clicked.connect(self._copy_frequent_tags)
        copy_freq_layout.addWidget(self.copy_frequent_btn)
        copy_freq_layout.addStretch(1)
        all_tab_layout.addLayout(copy_freq_layout)

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
        self.all_tags_table.horizontalHeader().sortIndicatorChanged.connect(
            self._on_all_tags_sort_changed
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
            self.raw_text_preview.clear()
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
        if img_path != self._src_pixmap_path:
            self._src_pixmap = QPixmap(img_path)
            self._src_pixmap_path = img_path if not self._src_pixmap.isNull() else None
        if not self._src_pixmap.isNull():
            scaled = self._src_pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)
        else:
            self.image_label.clear()
            self.image_label.setText("Cannot load image")

    # ── Image Tags tab ────────────────────────────────────────────────────────

    def load_tags(self):
        self.clear_tags()
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            self.raw_text_preview.clear()
            return
        for tag in self.file_manager.read_tags(img_path):
            btn = TagButton(tag)
            btn.deleted.connect(self.remove_tag)
            btn.edit_requested.connect(self.edit_tag)
            self.tags_layout.addWidget(btn)
        self.raw_text_preview.setPlainText(self.file_manager.read_caption(img_path))

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
            if not self.file_manager.save_tags(img_path, tags):
                QMessageBox.warning(self, "Warning", self.file_manager.last_error or "Failed to save tag.")
                return
            self.tag_input.clear()
            btn = TagButton(tag)
            btn.deleted.connect(self.remove_tag)
            btn.edit_requested.connect(self.edit_tag)
            self.tags_layout.addWidget(btn)

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
                if not self.file_manager.save_tags(img_path, tags):
                    QMessageBox.warning(self, "Warning", self.file_manager.last_error or "Failed to save tag.")
                    return
                self.load_tags()

    def remove_tag(self, tag):
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
        tags = self.file_manager.read_tags(img_path)
        if tag in tags:
            tags.remove(tag)
            # The TagButton that emitted this signal already deleteLater()s
            # itself regardless of what happens below; Qt's ChildRemoved
            # handling removes it from FlowLayout and triggers reflow
            # automatically, so no rebuild is needed on success. On failure
            # the button is still gone from the UI (nothing to roll back to)
            # but the warning at least tells the user the file itself wasn't
            # actually updated, instead of silently disagreeing with disk.
            if not self.file_manager.save_tags(img_path, tags):
                QMessageBox.warning(self, "Warning", self.file_manager.last_error or "Failed to save tag.")

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
        total = len(self.file_manager.image_files)
        if total == 0:
            QMessageBox.warning(self, "No Images", "No folder is loaded yet.")
            return
        value, ok = QLocale().toInt(text)
        if not ok:
            QMessageBox.warning(self, "Invalid Index", "Please enter a valid number.")
            return
        idx = value - 1
        if 0 <= idx < total:
            self.file_manager.current_index = idx
            self.update_ui()
            self.jump_input.clear()
        else:
            QMessageBox.warning(self, "Invalid Index", f"Please enter a number between 1 and {total}.")

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
                if not self.file_manager.save_tags(img_path, current_tags):
                    QMessageBox.warning(self, "Warning", self.file_manager.last_error or "Failed to save tags.")
                    return
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
            msg = f"Added '{tag}' to {count} files."
            if self.file_manager.last_error:
                msg += f"\n\nSome files failed to save: {self.file_manager.last_error}"
            QMessageBox.information(self, "Success", msg)
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
            msg = f"Removed '{tag}' from {count} files."
            if self.file_manager.last_error:
                msg += f"\n\nSome files failed to save: {self.file_manager.last_error}"
            QMessageBox.information(self, "Success", msg)
            self.tag_input.clear()
            self.load_tags()

    # ── All Tags tab ──────────────────────────────────────────────────────────

    def _on_tab_changed(self, index: int):
        if index == 1:
            self.refresh_all_tags_table()

    def refresh_all_tags_table(self):
        tag_counts = self.file_manager.get_tag_counts()
        self._populate_all_tags_table(tag_counts)
        self._filter_all_tags_table(self.all_tags_search.text())

    def _on_all_tags_sort_changed(self, col, order):
        self._all_tags_sort = (col, order)

    def _copy_frequent_tags(self):
        """Copy every tag that appears in at least N images to the clipboard.

        Tags are placed on the system clipboard as a comma-separated, Danbooru
        style string and also stored in the in-app tag clipboard so the
        'Paste Tags' button works for them too. Order follows get_tag_counts()
        (most frequent first)."""
        min_count = self.copy_min_count_spin.value()
        tags = [tag for tag, count in self.file_manager.get_tag_counts()
                if count >= min_count]
        if not tags:
            QMessageBox.information(
                self, "Copy Tags",
                f"{min_count}枚以上の画像に付いているタグはありません。",
            )
            return
        QGuiApplication.clipboard().setText(", ".join(tags))
        self.tag_clipboard = list(tags)
        self.statusBar().showMessage(
            f"{min_count}枚以上に出現する {len(tags)}個のタグをコピーしました。", 3000,
        )

    def _filter_all_tags_table(self, text: str):
        f = text.lower()
        table = self.all_tags_table
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            table.setRowHidden(row, bool(f) and (item is None or f not in item.text().lower()))

    def _populate_all_tags_table(self, tag_counts: list):
        table = self.all_tags_table
        table.setSortingEnabled(False)
        table.setRowCount(0)

        for tag, count in tag_counts:
            row = table.rowCount()
            table.insertRow(row)

            tag_item = QTableWidgetItem(tag)
            table.setItem(row, 0, tag_item)

            count_item = NumericTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row, 1, count_item)

        table.setSortingEnabled(True)
        sort_col, sort_order = self._all_tags_sort
        table.sortByColumn(sort_col, sort_order)

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

        total_removed = self.file_manager.remove_tags_from_all(tags)
        if self.file_manager.last_error:
            QMessageBox.warning(
                self, "Warning",
                f"一部のファイルの保存に失敗しました: {self.file_manager.last_error}",
            )

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
        # Includes every control that read-modify-writes the shared .txt
        # sidecar, so the GUI thread can't race a worker thread writing to it
        # -- the tagger (via ai_tagger._merge_tags) and the captioner (via
        # file_manager.save_caption) both read-modify-write the SAME file per
        # image, so an AI operation of either kind running must block the
        # other kind from starting too, not just another instance of itself.
        for widget in (self.oppai_btn, self.batch_oppai_btn,
                       self.add_all_btn, self.remove_all_btn,
                       self.tag_input, self.add_btn, self.paste_btn,
                       self.clear_all_btn, self.remove_selected_btn,
                       self.tags_container,
                       self.caption_btn, self.batch_caption_btn,
                       self.caption_mode_combo, self.caption_style_combo,
                       self.caption_instruction_edit,
                       self.oppai_model_combo, self.oppai_threshold_edit):
            widget.setEnabled(enabled)
        if enabled:
            self.caption_vram_tier_combo.setEnabled(self._is_natural_caption_mode())
        else:
            self.caption_vram_tier_combo.setEnabled(False)

    @staticmethod
    def _parse_threshold(text: str, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(text.strip())))
        except ValueError:
            return default

    def _get_oppai_threshold(self) -> float:
        return self._parse_threshold(self.oppai_threshold_edit.text(), 0.4)

    def _start_single_tagger(self, worker, label: str):
        self.set_ai_buttons_enabled(False)
        self._show_cancel_button()
        self.statusBar().showMessage(f"Initializing {label}...")
        self._active_worker = worker  # keep a reference so it isn't GC'd
        worker.progress.connect(self.update_status)
        worker.finished.connect(self.on_ai_finished)
        worker.start()

    def _start_batch_tagger(self, worker, label: str):
        if not self.file_manager.image_files:
            QMessageBox.warning(self, "Warning", "No images loaded in the folder.")
            return False

        self.set_ai_buttons_enabled(False)
        self.progress_bar.setVisible(True)
        self.batch_status_label.setVisible(True)
        self._show_cancel_button()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.file_manager.image_files))
        self._active_batch_worker = worker
        worker.progress.connect(self.update_batch_progress)
        worker.finished.connect(self.on_batch_finished)
        worker.start()
        return True

    def run_oppai_oracle(self):
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
        variant = self.oppai_model_combo.currentText()
        if not self._confirm_model_download(
            oppai_model_info(variant),
            action_title="この画像にタグキャプション (OppaiOracle) を実行します。",
        ):
            return
        worker = OppaiOracleWorker(
            img_path,
            threshold=self._get_oppai_threshold(),
            model_variant=variant,
        )
        self._start_single_tagger(worker, f"OppaiOracle ({variant})")

    def run_batch_oppai(self):
        if not self.file_manager.image_files:
            QMessageBox.warning(self, "Warning", "No images loaded in the folder.")
            return
        variant = self.oppai_model_combo.currentText()
        n = len(self.file_manager.image_files)
        if not self._confirm_model_download(
            oppai_model_info(variant),
            action_title=f"フォルダ内の全 {n} 枚にタグキャプション (OppaiOracle) を実行します。",
        ):
            return
        worker = BatchOppaiOracleWorker(
            self.file_manager, self.file_manager.image_files,
            threshold=self._get_oppai_threshold(),
            model_variant=variant,
        )
        self._start_batch_tagger(worker, f"OppaiOracle ({variant})")

    def update_status(self, msg: str):
        self.statusBar().showMessage(msg)

    def _show_cancel_button(self):
        self.cancel_batch_btn.setText("🛑 Cancel")
        self.cancel_batch_btn.setVisible(True)
        self.cancel_batch_btn.setEnabled(True)

    def _hide_cancel_button(self):
        self.cancel_batch_btn.setVisible(False)
        self.cancel_batch_btn.setEnabled(True)

    def cancel_batch(self):
        interrupted = False
        for attr in ("_active_batch_worker", "_active_worker", "_active_caption_worker"):
            worker = getattr(self, attr, None)
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                interrupted = True
        backend = getattr(self, "_caption_backend", None)
        if backend is not None:
            try:
                backend.cancel()
            except Exception:
                pass
        if interrupted:
            self.statusBar().showMessage("Cancelling...")
        self.cancel_batch_btn.setEnabled(False)

    def update_batch_progress(self, current: int, total: int, filename: str):
        self.progress_bar.setValue(current)
        self.batch_status_label.setText(f"Processing {current + 1}/{total}: {filename}")
        self.statusBar().showMessage(f"Batch Processing: {current + 1}/{total}...")

    def on_batch_finished(self, success_count: int, total: int, error_msg: str):
        self.set_ai_buttons_enabled(True)
        self.progress_bar.setVisible(False)
        self.batch_status_label.setVisible(False)
        self._hide_cancel_button()
        self.statusBar().clearMessage()
        if error_msg:
            QMessageBox.critical(self, "Batch Error", error_msg)
        else:
            QMessageBox.information(self, "Batch Complete",
                                    f"Successfully processed {success_count} out of {total} images.")
        self.load_tags()

    def on_ai_finished(self, new_tags: list, error_msg: str):
        self.set_ai_buttons_enabled(True)
        self._hide_cancel_button()
        self.statusBar().clearMessage()
        # Resolve the target from the worker, not the current view: the user
        # may have navigated to a different image while inference ran (which
        # can take minutes on the first, model-downloading run).
        worker = getattr(self, "_active_worker", None)
        img_path = getattr(worker, "image_path", None)
        if error_msg:
            QMessageBox.critical(self, "AI Error", error_msg)
            return
        if not img_path or not os.path.exists(img_path):
            return
        if not new_tags:
            QMessageBox.information(self, "AI Tagger", "No tags resulted from model.")
            return
        current_tags = self.file_manager.read_tags(img_path)
        added_count = 0
        for tag in new_tags:
            tag = str(tag).strip()
            if tag and tag not in current_tags:
                current_tags.append(tag)
                added_count += 1
        if added_count == 0:
            self.statusBar().showMessage("No new unique tags identified.", 3000)
            return
        self.file_manager.save_tags(img_path, current_tags)
        if img_path == self.file_manager.get_current_image_path():
            self.load_tags()
            self.statusBar().showMessage(f"Added {added_count} new tags.", 3000)
        else:
            self.statusBar().showMessage(
                f"Added {added_count} tags to {os.path.basename(img_path)}.", 4000
            )

    # ── AI Captioning ─────────────────────────────────────────────────────────

    def _is_natural_caption_mode(self):
        return self.caption_mode_combo.currentIndex() == 1

    def _on_caption_mode_changed(self, _index=None):
        is_nl = self._is_natural_caption_mode()
        self.tag_mode_widget.setVisible(not is_nl)
        self.nl_mode_widget.setVisible(is_nl)
        self.caption_vram_tier_combo.setEnabled(is_nl)
        if is_nl:
            self._on_caption_vram_changed()
        else:
            self._refresh_oppai_status_label()

    def _refresh_oppai_status_label(self, _index=None):
        info = oppai_model_info(self.oppai_model_combo.currentText())
        cached = "ダウンロード済み" if info["cached"] else "未ダウンロード（初回は長時間かかります）"
        self.oppai_status_label.setText(
            f"{info['label']}\n{info['repo']}\n"
            f"サイズ目安: {info['size_hint']}\n状態: {cached}"
        )

    def _on_caption_vram_changed(self, _index=None):
        info = caption_model_info(self.caption_vram_tier_combo.currentText())
        cached = "ダウンロード済み" if info["cached"] else "未ダウンロード（初回は長時間かかります）"
        if info.get("runtime_ready", True):
            runtime = "導入済み"
        else:
            missing = "、".join(info.get("runtime_missing") or [])
            runtime = (
                f"未導入（モデルと同時にダウンロード / {info.get('runtime_size_hint', '')}）"
                f"<br>不足: {missing}"
            )
        self.caption_backend_status_label.setText(
            f'{info["label"]}<br>'
            f'<a href="{info["url"]}">{info["repo"]}</a><br>'
            f'モデルサイズ目安: {info["size_hint"]}<br>'
            f'モデル: {cached}<br>'
            f'推論ランタイム: {runtime}'
        )

    def _confirm_model_download(self, info, *, action_title, extra_warning=""):
        cached = info["cached"]
        runtime_ready = info.get("runtime_ready", True)
        needs_download = info.get("needs_download", not cached)
        details = [
            action_title,
            "",
            f"モデル: {info['label']}",
            f"リポジトリ: {info['repo']}",
            f"サイズ目安: {info['size_hint']}",
        ]
        if cached:
            details += ["", "モデルはダウンロード済みです。この実行では再ダウンロードしません。"]
        else:
            details += [
                "",
                "初回は Hugging Face からモデルをダウンロードします。",
                "ダウンロードには長時間かかります。",
            ]
        if not runtime_ready:
            missing = "、".join(info.get("runtime_missing") or [])
            details += [
                "",
                "推論用パッケージもモデルと同時にダウンロードします。",
                f"不足: {missing}",
                f"サイズ目安: {info.get('runtime_size_hint', '')}",
                "こちらも長時間かかる場合があります。",
            ]
        if extra_warning:
            details += ["", extra_warning]

        box = QMessageBox(self)
        box.setWindowTitle("DL選択（長時間かかります）")
        box.setIcon(QMessageBox.Icon.Warning if needs_download else QMessageBox.Icon.Information)
        box.setText("DL選択（長時間かかります）")
        box.setInformativeText("\n".join(details))
        accept = box.addButton(
            "ダウンロードして実行" if needs_download else "実行",
            QMessageBox.ButtonRole.AcceptRole,
        )
        box.addButton("キャンセル", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(accept)
        box.exec()
        return box.clickedButton() == accept

    def _on_caption_style_changed(self, text):
        if text == "Character LoRA":
            self.caption_instruction_edit.setPlainText(CHARACTER_LORA_PROMPT)
        elif text == "Style LoRA":
            self.caption_instruction_edit.setPlainText(STYLE_LORA_PROMPT)

    def _save_caption_settings(self):
        self._settings.setValue(
            "caption/mode", "natural" if self._is_natural_caption_mode() else "tags",
        )
        self._settings.setValue("caption/vram_tier", self.caption_vram_tier_combo.currentText())
        self._settings.setValue("caption/style", self.caption_style_combo.currentText())
        self._settings.setValue("caption/instruction", self.caption_instruction_edit.toPlainText())
        # Force an immediate disk write (QSettings otherwise batches/delays
        # this) so a crash right after a change doesn't lose it -- this is
        # the save-on-change path's whole point vs. relying on closeEvent
        # alone.
        self._settings.sync()

    def _load_caption_settings(self):
        # Read all values up front, before touching any widget: setting
        # one combo below would otherwise fire its currentIndexChanged ->
        # _save_caption_settings and re-save the *other* settings using
        # their still-default widget values, clobbering the saved values this
        # method hasn't applied yet. Signals are also blocked below as a
        # second layer of protection (and to stop _on_caption_style_changed
        # from stomping a saved "Custom" instruction on the way in).
        mode = self._settings.value("caption/mode", None)
        backend_index = self._settings.value("caption/backend_index", None)
        vram_tier = self._settings.value("caption/vram_tier", None)
        style = self._settings.value("caption/style", None)
        instruction = self._settings.value("caption/instruction", None)

        with QSignalBlocker(self.caption_mode_combo):
            if mode == "natural":
                self.caption_mode_combo.setCurrentIndex(1)
            elif mode == "tags":
                self.caption_mode_combo.setCurrentIndex(0)

        with QSignalBlocker(self.caption_vram_tier_combo):
            selected = None
            if vram_tier in CAPTION_VRAM_TIERS:
                selected = vram_tier
            elif backend_index is not None:
                # Older builds stored 0=BF16 / 1=GGUF instead of a 96GB tier.
                try:
                    if int(backend_index) == 0:
                        selected = BF16_VRAM_TIER
                except (TypeError, ValueError):
                    pass
            if selected is None:
                selected = default_caption_vram_tier()
            idx = self.caption_vram_tier_combo.findText(selected)
            if idx >= 0:
                self.caption_vram_tier_combo.setCurrentIndex(idx)

        with QSignalBlocker(self.caption_style_combo):
            if style:
                idx = self.caption_style_combo.findText(style)
                if idx >= 0:
                    self.caption_style_combo.setCurrentIndex(idx)

        with QSignalBlocker(self.caption_instruction_edit):
            # is not None, not a truthiness check: an intentionally-cleared
            # (empty-string) instruction is a real saved value and must be
            # restored as empty, not silently fall back to the widget's
            # construction-time CHARACTER_LORA_PROMPT default.
            if instruction is not None:
                self.caption_instruction_edit.setPlainText(instruction)

    def _get_caption_backend(self):
        vram_tier = self.caption_vram_tier_combo.currentText()
        key = vram_tier
        if getattr(self, "_caption_backend_key", None) == key and getattr(self, "_caption_backend", None) is not None:
            return self._caption_backend

        old_backend = getattr(self, "_caption_backend", None)
        if old_backend is not None:
            try:
                old_backend.close()
            except Exception:
                pass

        if vram_tier == BF16_VRAM_TIER:
            backend = BF16Backend()
        else:
            backend = GGUFBackend(vram_tier=vram_tier)
        self._caption_backend = backend
        self._caption_backend_key = key
        return backend

    def _get_caption_instruction(self) -> str:
        text = self.caption_instruction_edit.toPlainText().strip()
        return text if text else CHARACTER_LORA_PROMPT

    def _start_single_captioner(self, worker):
        self.set_ai_buttons_enabled(False)
        self._show_cancel_button()
        self.statusBar().showMessage("Initializing captioner...")
        self._active_caption_worker = worker  # keep a reference so it isn't GC'd
        worker.progress.connect(self.update_status)
        worker.finished.connect(self.on_caption_finished)
        worker.start()

    def _start_batch_captioner(self, worker):
        if not self.file_manager.image_files:
            QMessageBox.warning(self, "Warning", "No images loaded in the folder.")
            return False

        self.set_ai_buttons_enabled(False)
        self.progress_bar.setVisible(True)
        self.batch_status_label.setVisible(True)
        self._show_cancel_button()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.file_manager.image_files))
        self._active_batch_worker = worker
        worker.progress.connect(self.update_batch_progress)
        worker.finished.connect(self.on_caption_batch_finished)
        worker.start()
        return True

    def run_caption(self):
        img_path = self.file_manager.get_current_image_path()
        if not img_path:
            return
        info = caption_model_info(self.caption_vram_tier_combo.currentText())
        if not self._confirm_model_download(
            info,
            action_title="この画像に自然言語キャプション (Qwen3.8) を実行します。",
            extra_warning="この画像の既存タグは自然言語キャプションで上書きされます。",
        ):
            return
        try:
            backend = self._get_caption_backend()
        except Exception as e:
            QMessageBox.critical(self, "Captioning Error", str(e))
            return
        worker = _BaseSingleCaptionWorker(img_path, backend, self._get_caption_instruction())
        self._start_single_captioner(worker)

    def run_batch_caption(self):
        if not self.file_manager.image_files:
            QMessageBox.warning(self, "Warning", "No images loaded in the folder.")
            return
        n = len(self.file_manager.image_files)
        info = caption_model_info(self.caption_vram_tier_combo.currentText())
        if not self._confirm_model_download(
            info,
            action_title=f"フォルダ内の全 {n} 枚に自然言語キャプション (Qwen3.8) を実行します。",
            extra_warning="各画像の既存タグ／キャプションは上書きされます。元に戻せません。",
        ):
            return
        try:
            backend = self._get_caption_backend()
        except Exception as e:
            QMessageBox.critical(self, "Captioning Error", str(e))
            return
        worker = _BaseBatchCaptionWorker(
            self.file_manager, self.file_manager.image_files, backend,
            self._get_caption_instruction(),
        )
        self._start_batch_captioner(worker)

    def on_caption_finished(self, caption_text: str, error_msg: str):
        self.set_ai_buttons_enabled(True)
        self._hide_cancel_button()
        self._on_caption_vram_changed()
        self.statusBar().clearMessage()
        # Resolve the target from the worker, not the current view: the user
        # may have navigated to a different image while generation ran.
        worker = getattr(self, "_active_caption_worker", None)
        img_path = getattr(worker, "image_path", None)
        if error_msg:
            if error_msg == "Cancelled":
                QMessageBox.information(self, "AI Caption", "Captioning cancelled.")
            else:
                QMessageBox.critical(self, "AI Caption Error", error_msg)
            return
        if not img_path or not os.path.exists(img_path):
            return
        if not self.file_manager.save_caption(img_path, caption_text):
            QMessageBox.critical(
                self, "AI Caption Error",
                f"Failed to save caption: {self.file_manager.last_error}",
            )
            return
        if img_path == self.file_manager.get_current_image_path():
            self.load_tags()
            self.statusBar().showMessage("Caption saved.", 3000)
        else:
            self.statusBar().showMessage(
                f"Caption saved for {os.path.basename(img_path)}.", 4000
            )

    def on_caption_batch_finished(self, success_count: int, total: int, error_msg: str):
        self.set_ai_buttons_enabled(True)
        self.progress_bar.setVisible(False)
        self.batch_status_label.setVisible(False)
        self._hide_cancel_button()
        self._on_caption_vram_changed()
        self.statusBar().clearMessage()
        if error_msg:
            QMessageBox.critical(self, "Batch Caption Error", error_msg)
        else:
            QMessageBox.information(self, "Batch Caption Complete",
                                    f"Successfully captioned {success_count} out of {total} images.")
        self.load_tags()

    # ── Close / Resize / Drag-and-Drop ───────────────────────────────────────

    def closeEvent(self, event):
        self._save_caption_settings()
        for worker in (getattr(self, "_active_worker", None),
                       getattr(self, "_active_batch_worker", None),
                       getattr(self, "_active_caption_worker", None)):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.wait(10000)
        event.accept()

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
