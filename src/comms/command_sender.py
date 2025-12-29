import socket
import time
import json
import threading
from pubsub import pub

class CommandSender:
    def __init__(self, robot_ip, port=8090):
        self.robot_ip = robot_ip
        self.port = port
        self.socket = None
        self.connected = False
        self.max_retries = 3
        self.retry_delay = 2
        self.timeout = 3  # Zaman aşımını azalt (önceki değer 5 saniye)
        self._lock = threading.Lock()  # Thread güvenliği için kilit ekle
        
    def connect(self):
        if self.connected and self.socket:
            return True
            
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                # Önceki soketi tamamen kapatıp temizleme
                if self.socket:
                    try:
                        self.socket.shutdown(socket.SHUT_RDWR)
                    except:
                        pass
                    try:
                        self.socket.close()
                    except:
                        pass
                    self.socket = None
                
                # Yeni soket oluşturup bağlanma
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(self.timeout)
                self.socket.connect((self.robot_ip, self.port))
                self.connected = True
                return True
                
            except (socket.timeout, ConnectionRefusedError) as e:
                pub.sendMessage('log', msg=f"Connection attempt {attempt+1} failed: {e}")
                time.sleep(self.retry_delay)
            except Exception as e:
                pub.sendMessage('log:error', msg=f"Unexpected error connecting to {self.robot_ip}:{self.port} - {e}")
                time.sleep(self.retry_delay)
        
        self.connected = False
        return False
    
    def send_command(self, command_type, params=None):
        if params is None:
            params = {}
    
        force_update = False  # Define force_update with a default value
        if force_update:
            params['force_publish'] = True    
    
        with self._lock:  # Thread güvenliği için kilit kullan
            if not self.connected and not self.connect():
                pub.sendMessage('log:error', msg=f"Cannot send command '{command_type}': Not connected.")
                return False
    
            # Add retry logic for sending commands
            max_send_retries = 2
            for send_attempt in range(max_send_retries):
                try:
                    command = {
                        'command': command_type,
                        'params': params
                    }
    
                    # Soket kontrolleri
                    if not self.socket:
                        if not self.connect():
                            return False
    
                    # >>> BU LOG ÇOK ÖNEMLİ <<<
                    pub.sendMessage('log:debug', msg=f"[CommandSender] DEBUG: >>> ATTEMPTING socket.sendall FOR: {command_type}")
                    # >>> BU LOG ÇOK ÖNEMLİ <<<
                    json_data = json.dumps(command).encode('utf-8') + b'\n'
                    self.socket.sendall(json_data)  # <-- Asıl gönderme işlemi
                    # >>> BU LOG ÇOK ÖNEMLİ <<<
                    pub.sendMessage('log:debug', msg=f"[CommandSender] DEBUG: >>> socket.sendall COMPLETED for: {command_type}")
                    # >>> BU LOG ÇOK ÖNEMLİ <<<
    
                    # Yanıt bekleme
                    response_data = b""
                    start_time = time.time()
    
                    # CommandReceiver'ın yanıt vermesi için parça parça okuma
                    while time.time() - start_time < self.timeout:
                        try:
                            chunk = self.socket.recv(1024)
                            if not chunk:  # Bağlantı kapandıysa
                                raise ConnectionResetError("Connection closed by remote host")
    
                            response_data += chunk
                            if b'\n' in response_data:  # Tam bir yanıt alındı
                                break
                        except socket.timeout:
                            continue  # Zamanı dolmadıysa devam et
    
                    if not response_data:
                        raise socket.timeout("No response received")
    
                    # Yanıtı işle
                    response_str = response_data.decode('utf-8').strip()
                    result = json.loads(response_str)
                    pub.sendMessage('log', msg=f"[CommandSender] Received response: {result}")
                    return result
    
                except (socket.timeout, ConnectionResetError, BrokenPipeError, OSError) as e:  # BrokenPipeError ve OSError eklendi
                    # >>> BU LOG ÇOK ÖNEMLİ <<<
                    pub.sendMessage('log:warning', msg=f"[CommandSender] DEBUG: >>> SOCKET ERROR during send/recv for {command_type}: {e}")
                    # >>> BU LOG ÇOK ÖNEMLİ <<<
                    self.connected = False
                    if self.socket:
                        try:
                            self.socket.close()
                        except:
                            pass
                        self.socket = None
    
                    if send_attempt < max_send_retries - 1:
                        if self.connect():
                            pub.sendMessage('log', msg=f"[CommandSender] Reconnected, retrying command")
                        else:
                            break
                except Exception as e:
                    # >>> BU LOG ÇOK ÖNEMLİ <<<
                    pub.sendMessage('log:error', msg=f"[CommandSender] DEBUG: >>> UNEXPECTED EXCEPTION in send_command for {command_type}: {e}")
                    # >>> BU LOG ÇOK ÖNEMLİ <<<
                    self.connected = False
                    if self.socket:
                        try:
                            self.socket.close()
                        except:
                            pass
                        self.socket = None
                    break
    
        return {"status": "error", "message": f"Command '{command_type}' failed after all retries"}
        
    def close(self):
        with self._lock:
            if self.socket:
                try:
                    self.socket.shutdown(socket.SHUT_RDWR)
                except:
                    pass
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
            self.connected = False