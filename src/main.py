#!/usr/bin/env python3
"""
DeskGUI Unified Entry Point
"""
import sys
import argparse
import multiprocessing
import traceback
from PyQt5.QtWidgets import QApplication
import src.config as config
from src.gui.desk_gui_app import DeskGUI

def build_arg_parser():
    parser = argparse.ArgumentParser(description="SentryBOT DeskGUI Launcher")
    parser.add_argument("--robot-ip", default="192.168.137.52", help="Robot IP Address")
    parser.add_argument("--video-port", type=int, default=8000, help="Video Stream Port")
    parser.add_argument("--command-port", type=int, default=8090, help="Command Port")
    parser.add_argument("--encodings-file", default=str(config.ENCODINGS_FILE), help="Face Encodings File")
    parser.add_argument("--theme", default="auto", choices=["auto", "light", "dark", "red"], help="Theme")
    parser.add_argument("--debug", action="store_true", help="Debug Mode")
    parser.add_argument("--ollama-url", default="http://localhost:11435/api", help="Ollama URL")
    parser.add_argument("--ollama-model", default="SentryBOT:4b", help="Ollama Model")
    parser.add_argument("--gemini-api-key", default=None, help="Gemini API Key")
    return parser

def main():
    args = build_arg_parser().parse_args()
    
    try:
        if sys.platform.startswith("win"):
            multiprocessing.freeze_support()
            
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        
        gui = DeskGUI(**vars(args))
        gui.show()
        
        sys.exit(app.exec_())
    except Exception as exc:
        print(f"Error starting DeskGUI: {exc}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
