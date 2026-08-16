import sys
import os
import traceback
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from ui_main import MainWindow


def _relax_console_encoding_errors():
    """Running from source (not frozen) on a non-UTF-8 Windows console (e.g.
    cp932 on Japanese Windows) leaves sys.stdout/stderr unable to encode
    characters outside that codepage. A worker thread's own diagnostic
    print(..., file=sys.stderr) call hitting an unencodable character (far
    more likely with LLM-generated caption prose -- em/en dashes, curly
    quotes -- than the short tag words this app used to only ever print)
    would then raise UnicodeEncodeError *inside* a QThread.run(), which Qt
    swallows silently: the thread just stops without ever emitting its
    finished signal, leaving the UI stuck in "processing" state with no
    error shown. Reconfiguring both streams to degrade to an escaped
    representation instead of raising closes that hang for every print in
    the app, not just newly-added ones."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="backslashreplace")
            except Exception:
                pass


def _setup_frozen_logging():
    """A PyInstaller --windowed build has no console, so sys.stdout/stderr
    are None; any stray print() or traceback would raise AttributeError
    instead of just being lost. Redirect both to a log file instead."""
    if not getattr(sys, "frozen", False):
        return
    log_dir = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "tag_editor", "logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    log = open(os.path.join(log_dir, "tag_editor.log"), "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log
    sys.excepthook = lambda exc_type, exc, tb: traceback.print_exception(exc_type, exc, tb, file=log)


def main():
    _setup_frozen_logging()
    _relax_console_encoding_errors()
    app = QApplication(sys.argv)

    # Apply basic styling
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    if "--smoke-test" in sys.argv:
        # CI packaging check: boot the GUI and exit cleanly (used with
        # QT_QPA_PLATFORM=offscreen) without running any real inference.
        QTimer.singleShot(0, app.quit)

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
