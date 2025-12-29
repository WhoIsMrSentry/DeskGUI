#!/usr/bin/env python3
"""
Integrated launcher for SentryBOT GUI and Bluetooth Audio Server
This script runs both components on your laptop to manage the robot
"""

import argparse
import sys
import os
import threading
import time
import traceback

# Print startup message immediately for debugging
print("Starting SentryBOT integrated launcher...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Make sure we can import from parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
print(f"Added parent directory to path: {parent_dir}")

try:
    print("Importing required modules...")
    from desk_gui import launch_gui
    print("Imports successful")
except ImportError as e:
    print(f"ERROR: Import failed - {e}")
    print("Make sure you're running this script from the DeskGUI directory")
    print("and all dependencies are installed.")
    sys.exit(1)

def main():
    print("Parsing command-line arguments...")
    parser = argparse.ArgumentParser(description='Run SentryBOT Desktop GUI with Bluetooth Audio Support')
    
    # GUI options
    parser.add_argument('--robot-ip', default='192.168.137.52', help='IP address of the robot')
    parser.add_argument('--video-port', type=int, default=8000, help='Video streaming port')
    parser.add_argument('--command-port', type=int, default=8090, help='Command port')
    parser.add_argument('--ollama-url', default='http://localhost:11435/api', help='Ollama API URL (port 11435)')
    parser.add_argument('--encodings-file', default='encodings.pickle', help='Face encodings file')
    parser.add_argument('--debug', action='store_true', help='Show debug information')
    parser.add_argument('--theme', default='auto', choices=['light', 'dark', 'auto'], 
                        help='Application theme (light, dark, auto)')
    parser.add_argument('--xtts', action='store_true', help='XTTS API sunucusunu ayrı terminalde başlat (Windows)')
    try:
        args = parser.parse_args()
        print(f"Arguments parsed: {args}")
        
        # XTTS API sunucusunu başlat
        if getattr(args, 'xtts', False):
            xtts_bat = r'C:\Users\emirh\xTTS\start_xtts_api.bat'
            xtts_cwd = r'C:\Users\emirh\xTTS'
            if os.path.exists(xtts_bat):
                print(f"XTTS API sunucusu başlatılıyor: {xtts_bat}")
                import subprocess
                try:
                    subprocess.Popen(['start', '', xtts_bat], shell=True, cwd=xtts_cwd)
                except Exception as e:
                    print(f"subprocess ile başlatma başarısız oldu: {e}")
                print("XTTS API sunucusu bağımsız olarak başlatıldı.")
            else:
                print(f"HATA: {xtts_bat} bulunamadı!")
        
        # Configure local IP
        local_ip = "127.0.0.1"
        try:
            print("Determining local IP address...")
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            print(f"Local IP address: {local_ip}")
        except Exception as e:
            print(f"WARNING: Failed to determine local IP - {e}")
        
        # Start the GUI
        gui_config = {
            'robot_ip': args.robot_ip,
            'video_port': args.video_port,
            'command_port': args.command_port,
            'ollama_url': args.ollama_url,
            'encodings_file': args.encodings_file,
            'bluetooth_server': local_ip,  # Eski kod yapısıyla uyumluluk için tutuldu
            'debug': args.debug,
            'theme': args.theme,  # Tema parametresini ekledik
        }
        
        print("Starting SentryBOT Desktop GUI...")
        print(f"Robot IP: {args.robot_ip}")
        print(f"Theme: {args.theme}")

        launch_gui(**gui_config)
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nERROR: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        print(traceback.format_exc())
        sys.exit(1)