import socket
import json
from PyQt5.QtCore import QThread, pyqtSignal

class RobotDataListener(QThread):
    # Signals to update GUI safely
    robot_status_updated = pyqtSignal(dict)
    robot_log_received = pyqtSignal(str)
    robot_disconnected = pyqtSignal()
    robot_connected = pyqtSignal(str) # Parameter: address
    robot_personality_list_updated = pyqtSignal(list) # Kişilik listesi için sinyal

    def __init__(self, listen_port=8091, parent=None):
        super().__init__(parent)
        self.listen_port = listen_port
        self.running = False
        self.server_socket = None
        self.client_socket = None
        self.client_address = None

    def run(self):
        self.running = True
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.listen_port))
            self.server_socket.listen(1) # Listen for one connection from the robot
            print(f"[RobotListener] Listening for robot data on port {self.listen_port}")

            while self.running:
                self.client_socket = None
                try:
                    self.server_socket.settimeout(1.0) # Check self.running flag periodically
                    self.client_socket, self.client_address = self.server_socket.accept()
                    print(f"[RobotListener] Robot connected from {self.client_address}")
                    self.robot_connected.emit(str(self.client_address)) # Signal connection

                    # Handle data from this client
                    self.handle_robot_data()

                except socket.timeout:
                    continue # No connection attempt, check running flag
                except OSError as e:
                     if self.running:
                         print(f"[RobotListener] Socket error during accept: {e}")
                     break # Stop listening on error
                except Exception as e:
                    if self.running:
                        print(f"[RobotListener] Unexpected error during accept: {e}")
                    break

                # After client disconnects or error, signal disconnection
                if self.client_address:
                    print(f"[RobotListener] Robot {self.client_address} disconnected.")
                    self.robot_disconnected.emit()
                    self.client_address = None # Reset for next connection

        except Exception as e:
            print(f"[RobotListener] Failed to start listener on port {self.listen_port}: {e}")
        finally:
            self.stop_listening()
            print("[RobotListener] Listener thread finished.")

    def handle_robot_data(self):
        """Receives and processes data from the connected robot."""
        buffer_size = 4096
        full_data = b""
        separator = b'\n'
        self.client_socket.settimeout(10.0) # Timeout for receiving data

        while self.running and self.client_socket:
            try:
                chunk = self.client_socket.recv(buffer_size)
                if not chunk:
                    break # Robot disconnected

                full_data += chunk
                while separator in full_data:
                    message_part, full_data = full_data.split(separator, 1)
                    if message_part:
                        self.process_robot_message(message_part.decode('utf-8'))

            except socket.timeout:
                 # Send keepalive? Or just assume connection is potentially stale
                 try:
                     self.client_socket.send(b'') # Check connection
                 except:
                     print("[RobotListener] Connection check failed after timeout.")
                     break # Disconnected
                 continue # Continue listening
            except ConnectionResetError:
                 break # Robot disconnected abruptly
            except Exception as e:
                 print(f"[RobotListener] Error receiving data: {e}")
                 break # Stop handling on error

        # Cleanup when loop exits
        if self.client_socket:
            try: self.client_socket.close()
            except: pass
            self.client_socket = None


    def process_robot_message(self, message_str):
        """Parses JSON message and emits signals."""
        try:
            data = json.loads(message_str)
            command = data.get('command')
            params = data.get('params', {})

            if command == 'update_status':
                # Durum güncellemesinde kişilik listesi YOK artık
                self.robot_status_updated.emit(params)
            elif command == 'update_personality_list': # YENİ KOMUT
                personalities = params.get('personalities', [])
                self.robot_personality_list_updated.emit(personalities) # Yeni sinyali yayınla
            elif command == 'log':
                log_msg = params.get('message', 'No message')
                level = params.get('level', 'info')
                self.robot_log_received.emit(f"[Robot {level.upper()}] {log_msg}")
            elif command == 'hello_from_robot':
                 # Sadece loglayabilir veya GUI'de bir bildirim gösterebilirsiniz
                 hello_msg = params.get('message', 'Robot connected')
                 self.robot_log_received.emit(f"[Robot INFO] {hello_msg}") # Loga ekle
                 # print(f"[RobotListener] Received hello: {params}") # Veya sadece print          
            elif command == 'ping':
                 # Could respond with pong if needed, but usually just for keepalive
                 pass
            else:
                print(f"[RobotListener] Received unknown command from robot: {command}")

        except json.JSONDecodeError:
            print(f"[RobotListener] Received non-JSON message: {message_str[:100]}")
        except Exception as e:
            print(f"[RobotListener] Error processing message: {e}")

    def stop_listening(self):
        """Stops the listener thread."""
        self.running = False
        if self.client_socket:
             try: self.client_socket.close()
             except: pass
        if self.server_socket:
            try: self.server_socket.close()
            except: pass