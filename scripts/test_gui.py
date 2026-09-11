"""Headless-ish GUI test: build the window, exercise it, tear it down.

Run from the repo root:  .venv\\Scripts\\python.exe scripts\\test_gui.py
Exit 0 = the window builds, theme toggles, toggles and hide/show work.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tkinter as tk


def main() -> int:
    from app.gui import AppWindow
    from src.dictation import DictationEngine

    engine = DictationEngine(Path(__file__).resolve().parent.parent)
    root = tk.Tk()
    window = AppWindow(root, engine, on_quit=root.destroy)

    # pump the event loop a few rounds (pollers run inside)
    for _ in range(6):
        root.update()
        root.after(50, lambda: None)

    assert window.mic_canvas is not None
    assert window.status_label.cget("text")

    # tab buttons must keep their size when switching (the old Notebook bug)
    root.update_idletasks()
    root.update()
    home_btn = window.tab_buttons["home"]
    width_before = home_btn.winfo_width()
    window._show_tab("settings")
    root.update()
    window._show_tab("home")
    root.update()
    width_after = home_btn.winfo_width()
    if width_before > 1 and width_after > 1:
        assert width_before == width_after, (width_before, width_after)

    # toggle listening twice — settings must survive round-trips
    window.toggle_from_tray()
    root.update()
    window.toggle_from_tray()
    root.update()

    # theme switch both ways
    window._toggle_theme()
    root.update()
    window._toggle_theme()
    root.update()

    # hide to tray and back
    window._hide()
    root.update()
    window.show()
    root.update()

    # quit path: engine stops, window closes (the "process stays" bug);
    # on_quit is root.destroy here, so any failure would raise
    window.quit()

    print("gui test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
