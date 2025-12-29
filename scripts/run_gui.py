#!/usr/bin/env python3
"""
SentryBOT Desktop GUI with Bluetooth Audio Support
"""

import argparse
import sys
import os
import traceback

# Print startup message immediately for debugging
print("Starting SentryBOT GUI launcher...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Make sure we can import from parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
print(f"Added parent directory to path: {parent_dir}")

try:
    print("Importing desk_gui module...")
    from desk_gui import launch_gui
    print("Import successful")
except ImportError as e:
    print(f"ERROR: Failed to import from desk_gui - {e}")
    print("Make sure you're running this script from the DeskGUI directory")
    print("and all dependencies are installed.")
    sys.exit(1)

def main():
    print("Parsing command-line arguments...")
    parser = argparse.ArgumentParser(description='Run SentryBOT Desktop GUI')
    
    parser.add_argument('--robot-ip', default='192.168.137.52', help='IP address of the robot')
    parser.add_argument('--video-port', type=int, default=8000, help='Video streaming port')
    parser.add_argument('--command-port', type=int, default=8090, help='Command port')
    parser.add_argument('--ollama-url', default='http://localhost:11435', help='Ollama API base URL (port 11435, without /api/generate)')
    parser.add_argument('--ollama-model', default='SentryBOT:4b', help='Ollama model to use')
    parser.add_argument('--encodings-file', default='encodings.pickle', help='Face encodings file')
    parser.add_argument('--bluetooth-server', default='192.168.1.100', help='Bluetooth audio server IP')
    parser.add_argument('--enable-fastapi', action='store_true', help='Enable FastAPI support')
    
    parser.add_argument('--retry-on-error', action='store_true', help='Automatically retry on error')
    parser.add_argument('--log-file', default='sentry_gui.log', help='Log file for errors')
    
    parser.add_argument('--debug', action='store_true', help='Show debug information')
    
    try:
        args = parser.parse_args()
        print(f"Arguments parsed: {args}")
        
        if args.debug:
            import logging
            logging.basicConfig(
                filename=args.log_file,
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logging.info("Starting SentryBOT GUI")
            
        config = {
            'robot_ip': args.robot_ip,
            'video_port': args.video_port,
            'command_port': args.command_port,
            'ollama_url': args.ollama_url,
            'ollama_model': args.ollama_model,
            'encodings_file': args.encodings_file,
            'bluetooth_server': args.bluetooth_server,
            'enable_fastapi': args.enable_fastapi,
        }
        
        print("Starting SentryBOT Desktop GUI")
        print(f"Robot IP: {args.robot_ip}")
        print(f"Bluetooth Audio Server: {args.bluetooth_server}")
        
        # Check if PyQt5 is available
        try:
            from PyQt5 import QtWidgets
            print("PyQt5 modules loaded successfully")
        except ImportError:
            print("ERROR: PyQt5 is not installed. Please run: pip install PyQt5")
            sys.exit(1)
            
        # Global exception handler
        def exception_handler(exctype, value, tb):
            error_msg = f"Unhandled exception: {value}"
            print(error_msg)
            traceback.print_exception(exctype, value, tb)
            
            if args.debug and 'logging' in sys.modules:
                logging.error(error_msg, exc_info=(exctype, value, tb))
                
            if args.retry_on_error and exctype not in [SystemExit, KeyboardInterrupt]:
                print("Attempting to restart the application...")
                sys.exc_info = lambda: (None, None, None)
                launch_gui(**config)
                
        sys.excepthook = exception_handler
            
        # Launch GUI with error handling
        launch_gui(**config)
        
    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        print(traceback.format_exc())
        sys.exit(1)
