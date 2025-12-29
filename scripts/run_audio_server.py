#!/usr/bin/env python3
"""
Bluetooth Audio Server for SentryBOT
Run this on your laptop to provide audio services for the robot.
"""

import argparse
import os
import sys

# Make sure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.audio.bluetooth_audio import BluetoothAudioServer
import time

def main():
    parser = argparse.ArgumentParser(description='Run Bluetooth Audio Server for SentryBOT')
    
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--tts-port', type=int, default=8095, help='Port for TTS service')
    parser.add_argument('--speech-port', type=int, default=8096, help='Port for speech recognition service')
    parser.add_argument('--fastapi-port', type=int, default=8098, help='Port for FastAPI WebSocket server')
    parser.add_argument('--use-fastapi', action='store_true', help='Use FastAPI for improved performance')
    parser.add_argument('--device-name', help='Microphone device name')
    parser.add_argument('--device-index', type=int, help='Microphone device index (alternative to device name)')
    parser.add_argument('--list-devices', action='store_true', help='List available microphone devices')
    parser.add_argument('--voice-idx', type=int, default=0, help='Voice index for TTS')
    parser.add_argument('--auto-start-speech', action='store_true', help='Automatically start speech recognition on launch')
    parser.add_argument('--language', default='en-US', help='Language for speech recognition (e.g., en-US, fr-FR, de-DE)')
    parser.add_argument('--service-name', default='SentryBOT_Audio', help='Service name for zeroconf registration')
    parser.add_argument('--no-register-service', action='store_true', help='Disable zeroconf service registration')
    parser.add_argument('--test-audio', action='store_true', help='Test audio output on startup')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        import speech_recognition as sr
        print("Available microphone devices:")
        for i, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"{i}: {name}")
        
        # Also list TTS voices
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            print("\nAvailable TTS voices:")
            for i, voice in enumerate(voices):
                print(f"{i}: {voice.name} ({voice.id}), Language: {voice.languages[0] if len(voice.languages) > 0 else 'Unknown'}")
            engine.stop()
        except Exception as e:
            print(f"Failed to list TTS voices: {e}")
        
        return
    
    # Process service registration flag
    register_service = not args.no_register_service
    
    # Start server
    try:
        # Try to get primary sound device if not specified
        if not args.device_name:
            try:
                import speech_recognition as sr
                mic_names = sr.Microphone.list_microphone_names()
                
                print("Available microphones:")
                for idx, name in enumerate(mic_names):
                    print(f"{idx}: {name}")
                
                if mic_names:
                    default_device = None
                    # Look for likely default microphones - with refined search criteria
                    default_keywords = ['default', 'mic', 'primary', 'input', 'microphone', 'array', 'built-in']
                    
                    for idx, name in enumerate(mic_names):
                        name_lower = name.lower()
                        if any(keyword in name_lower for keyword in default_keywords):
                            default_device = name
                            device_index = idx
                            print(f"Auto-selecting microphone: {default_device} (index {device_index})")
                            args.device_name = default_device
                            break
                    
                    # If no preferred device found, use the first one
                    if not default_device and len(mic_names) > 0:
                        default_device = mic_names[0]
                        print(f"Auto-selecting first available microphone: {default_device}")
                        args.device_name = default_device
            except Exception as e:
                print(f"Error auto-selecting microphone: {e}")
                import traceback
                traceback.print_exc()
        
        # If device index is specified, use it instead of device name
        if args.device_index is not None:
            print(f"Using device index: {args.device_index}")
            args.device_name = None  # Clear device name to use index
        
        print(f"Using microphone device: {args.device_name or 'Default system microphone'}")
        
        # Get TTS voices for better selection
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            print("\nAvailable TTS voices:")
            for i, voice in enumerate(voices):
                print(f"{i}: {voice.name} ({voice.id})")
            engine.stop()
            del engine
        except Exception as e:
            print(f"Error listing TTS voices: {e}")
        
        server = BluetoothAudioServer(
            host=args.host,
            tts_port=args.tts_port,
            speech_port=args.speech_port,
            use_fastapi=args.use_fastapi,
            fastapi_port=args.fastapi_port,
            device_name=args.device_name,
            device_index=args.device_index if args.device_index is not None else None,
            voice_idx=args.voice_idx,
            auto_start_speech=args.auto_start_speech,
            language=args.language,
            register_service=register_service,
            service_name=args.service_name,
            verbose=args.verbose
        )
        
        # Print startup message
        print("=" * 60)
        print("SentryBOT Bluetooth Audio Server")
        print("=" * 60)
        print(f"TTS Service: {args.host}:{args.tts_port}")
        print(f"Speech Recognition Service: {args.host}:{args.speech_port}")
        print(f"Recognition Language: {args.language}")
        print(f"Using microphone: {args.device_name or 'Default system microphone'}")
        print(f"Using voice index: {args.voice_idx}")
        
        if args.use_fastapi:
            print(f"FastAPI WebSocket Service: {args.host}:{args.fastapi_port}")
        if args.auto_start_speech:
            print("Speech recognition will start automatically")
        print(f"Service registration: {'Enabled' if register_service else 'Disabled'}")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60)
        
        server.start()
        
        # Test audio if requested
        if args.test_audio:
            print("Testing audio output...")
            server._process_tts("This is a test of the audio output. If you can hear this, audio is working correctly.")
        
        # Give feedback that server is running
        print("\nServer is running. Use Ctrl+C to stop.")
        
        # Keep the main thread running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error running server: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'server' in locals():
            server.stop()

if __name__ == "__main__":
    main()
