#!/usr/bin/env python3
"""DeskGUI basit başlatma script'i.

Gelişmiş kullanım için run_gui.py veya run_all.py tercih edin.
"""
from __future__ import annotations
import sys
import argparse
import multiprocessing
import traceback
from PyQt5.QtWidgets import QApplication
from modules.gui.desk_gui_app import DeskGUI


def build_arg_parser():
    parser = argparse.ArgumentParser(description="SentryBOT DeskGUI Başlatıcı")
    parser.add_argument("--robot-ip", default="192.168.137.52", help="Robot IP adresi")
    parser.add_argument("--video-port", type=int, default=8000, help="Video akış portu")
    parser.add_argument("--command-port", type=int, default=8090, help="Komut portu")
    parser.add_argument("--encodings-file", default="encodings.pickle", help="Yüz kodları dosyası")
    parser.add_argument("--theme", default="auto", choices=["auto", "light", "dark", "red"], help="Tema")
    parser.add_argument("--debug", action="store_true", help="Debug modu")
    parser.add_argument("--ollama-url", default="http://localhost:11435", help="Ollama URL (varsayılan port 11435)")
    parser.add_argument("--ollama-model", default="SentryBOT:4b", help="Ollama modeli")
    parser.add_argument("--gemini-api-key", default=None, help="Gemini API anahtarı")
    return parser


def launch_gui(**kwargs):
    try:
        if sys.platform.startswith("win"):
            multiprocessing.freeze_support()
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        gui = DeskGUI(**kwargs)
        gui.show()
        return app.exec_()
    except Exception as exc:  # noqa: BLE001
        print(f"DeskGUI başlatma hatası: {exc}")
        traceback.print_exc()
        return 1


def main():
    args = build_arg_parser().parse_args()
    sys.exit(launch_gui(**vars(args)))


if __name__ == "__main__":  # pragma: no cover
    main()
