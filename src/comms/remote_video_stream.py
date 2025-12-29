import threading
import time
import cv2
from pubsub import pub

class RemoteVideoStream:
    def __init__(self, robot_ip, port=8000, **kwargs):
        self.robot_ip = robot_ip
        self.port = port
        self.socket = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        
        # Bağlantı hatalarına dayanıklılık için ek parametreler
        self.reconnect_attempts = kwargs.get('reconnect_attempts', 3)
        self.reconnect_delay = kwargs.get('reconnect_delay', 2)
        self.connection_timeout = kwargs.get('connection_timeout', 10)
        
    def start(self):
        self.running = True
        threading.Thread(target=self._stream_receiver).start()
        return self
        
    def _stream_receiver(self):
        attempt = 0
        last_error_time = 0
        
        # Use OpenCV for MJPEG streaming - much more reliable than socket connection
        stream_url = f"http://{self.robot_ip}:{self.port}/stream.mjpg"
        pub.sendMessage('log', msg=f"[RemoteVideoStream] Connecting to stream: {stream_url}")
        
        while self.running:
            try:
                if attempt > 0:
                    pub.sendMessage('log', msg=f"[RemoteVideoStream] Reconnection attempt {attempt}/{self.reconnect_attempts}")
                
                # Open video capture from MJPEG stream
                self.cap = cv2.VideoCapture(stream_url)
                if not self.cap.isOpened():
                    raise ConnectionError(f"Could not open stream at {stream_url}")
                
                # Reset attempt counters on successful connection
                attempt = 0
                last_error_time = 0
                
                # Main frame reading loop
                while self.running:
                    ret, frame = self.cap.read()
                    if not ret:
                        pub.sendMessage('log', msg="[RemoteVideoStream] Failed to read frame, reconnecting...")
                        break
                    
                    # Store frame safely
                    with self.lock:
                        self.frame = frame
                    
                    # Small delay to prevent CPU overload
                    time.sleep(0.01)
                    
            except Exception as e:
                pub.sendMessage('log', msg=f"[RemoteVideoStream] Connection error: {str(e)}")
                
                # Reset retry attempts if enough time has passed
                current_time = time.time()
                if current_time - last_error_time > 60:
                    attempt = 0
                
                # Update error time
                last_error_time = current_time
                
                # Close capture if exists
                if hasattr(self, 'cap') and self.cap:
                    try:
                        self.cap.release()
                    except:
                        pass
                
                # Check if we should retry
                attempt += 1
                if not self.running or attempt > self.reconnect_attempts:
                    break
                    
                # Wait before retrying
                time.sleep(self.reconnect_delay)
                
        # Ensure resources are cleaned up
        self.stop()
    
    def read(self):
        with self.lock:
            return self.frame
            
    def stop(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()