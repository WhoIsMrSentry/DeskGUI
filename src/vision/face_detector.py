import cv2
import threading
import time
import os
import json
import numpy as np
import face_recognition
import pickle
from queue import Queue, Empty
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pubsub import pub
import src.config as config

class FaceDetector:
    def __init__(self, encodings_file=None):
        # Still keep cascade for backup/fallback
        cascade_path = str(config.HAARCASCADE_FILE)
        self.cascade = cv2.CascadeClassifier(cascade_path)
        # Priority animations
        self.priority_animations = {}  # Kişi adı -> animasyon eşleştirmesi
        self.load_priority_animations()
        # --- YENİ EKLENENLER ---
        self.last_valid_result_time = 0.0 # En son geçerli (yüz içeren) sonucun zamanı
        self.result_timeout = 1.5 # saniye cinsinden zaman aşımı (ayarlayabilirsiniz)
        # --- YENİ EKLENENLER SONU ---

        # --- YENİ KONTROL ---
        if self.cascade.empty():
            error_msg = f"FATAL: Could not load cascade classifier from {cascade_path}"
            pub.sendMessage('log:error', msg=error_msg)
            # GUI'ye de bildirim gönderebilir veya programı durdurabiliriz.
            # Şimdilik sadece loglayalım ve None yapalım ki sonraki kullanımlarda hata versin.
            print(f"!!! {error_msg} !!!") # Konsola da yazdır
            # Belki burada bir Exception raise etmek daha doğru olur?
            # raise IOError(error_msg) # Programın başlamasını engeller
            self.cascade = None # Cascade kullanılamaz durumda
        else:
             pub.sendMessage('log', msg=f"Cascade classifier loaded successfully from {cascade_path}")
        # --- KONTROL SONU ---

        # Face recognition data
        self.data = None
        self.encodings_file = encodings_file or str(config.ENCODINGS_FILE)
        self.load_encodings()
        
        # Priority persons configuration with ordering
        self.priority_persons = []
        self.priority_order = {}  # Dictionary to store priority order
        self.priority_file = str(config.PRIORITY_PERSONS_FILE)
        self.load_priority_persons()
        
        # Optimization parameters
        self.scale_factor = 1.2
        self.min_neighbors = 5
        self.min_size = (30, 30)
        self.last_frame_time = time.time()
        self.detection_interval = 0.1
        
        # Recognition parameters for fine-tuning
        self.recognition_tolerance = 0.6
        self.min_face_size_for_recognition = 60
        
        # Debug counters for diagnostics
        self.face_detections = 0
        self.face_recognitions = 0
        self.recognition_failures = 0
        
        # Known face encodings and names caches for faster processing
        self.known_face_encodings = []
        self.known_face_names = []
        self.prepare_cached_encodings()
        
        # Create processing queue
        self.frame_queue = Queue(maxsize=1)  # Only keep latest frame
        self.result_queue = Queue(maxsize=2)  # Keep a few results
        
        # Create processing thread
        self.processing_active = False
        self.processing_thread = None
        self.start_processing_thread()
        
        # Multiscale processing
        self.process_every_nth_frame = 5  # Process only every Nth frame fully
        self.scale_factor = 0.5  # Scale factor for resizing frames
        self.frame_count = 0
        self.last_detection_result = ([], [], [])

        self.detection_interval = 0.2  # Interval between detections in seconds
        self.use_hog_always = True  # Use HOG for face detection
        
        # Processing pool for face recognition
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Current results storage
        self.current_faces = []
        self.current_names = []
        self.current_priority_faces = []
        self.results_lock = threading.Lock()
        self.last_recognition_time = time.time()

        # --- YENİ: Son tespit edilen yüzleri ve zamanı sakla ---
        self.last_faces = []
        self.last_names = []
        self.last_priority_faces = []
        self.last_faces_time = 0
        self.max_persist_time = 1.5  # saniye, yüz kaybolduktan sonra kutu ne kadar ekranda kalsın
    
    def start_processing_thread(self):
        """Start background thread for face detection/recognition using multiprocessing."""
        if self.processing_thread is not None and self.processing_thread.is_alive():
            # Already running, consider it a success for idempotency
            pub.sendMessage('log', msg="Face recognition background thread already running.")
            return True # <--- ADDED: Return True if already running

        self.processing_active = True
        try:
            self.processing_thread = threading.Thread(
                target=self._background_processing,
                daemon=True
            )
            self.processing_thread.start()
            pub.sendMessage('log', msg="Face recognition background processing started")
            # Optional: Add a small delay and check is_alive() for extra safety
            # time.sleep(0.1)
            # if not self.processing_thread.is_alive():
            #     pub.sendMessage('log:error', msg="Face processing thread failed to start!")
            #     return False
            return True # <--- ADDED: Explicitly return True on success
        except Exception as e:
            pub.sendMessage('log:error', msg=f"Failed to start face processing thread: {e}")
            self.processing_active = False # Reset flag on failure
            return False # <--- ADDED: Return False on exception

    def load_priority_animations(self):
        """Kişi-animasyon eşleştirmelerini yükle"""
        try:
            priority_file = str(config.PRIORITY_ANIMATIONS_FILE)
            if os.path.exists(priority_file):
                with open(priority_file, 'r') as f:
                    self.priority_animations = json.load(f)
                pub.sendMessage('log', msg=f"Loaded {len(self.priority_animations)} priority animations")
            else:
                # Varsayılan eşleştirmeler, dosya yoksa bunlar kullanılacak
                self.priority_animations = {
                    "SentryCoderDev": "wave",
                    # Diğer varsayılan eşleştirmeler buraya eklenebilir
                }
                # Varsayılan eşleştirmeleri kaydet
                self.save_priority_animations()
        except Exception as e:
            pub.sendMessage('log', msg=f"Error loading priority animations: {e}")
            self.priority_animations = {}

    def save_priority_animations(self):
        """Kişi-animasyon eşleştirmelerini kaydet"""
        try:
            priority_file = str(config.PRIORITY_ANIMATIONS_FILE)
            with open(priority_file, 'w') as f:
                json.dump(self.priority_animations, f, indent=2)
            pub.sendMessage('log', msg=f"Saved {len(self.priority_animations)} priority animations")
        except Exception as e:
            pub.sendMessage('log', msg=f"Error saving priority animations: {e}")

    def get_animation_for_person(self, name):
        """Belirli bir kişi için tanımlı animasyonu döndür"""
        return self.priority_animations.get(name, None)

    def reload_data(self):
        """Reload face recognition data (after training or restoring)."""
        self.log_signal.emit("Reloading face encodings...") # Log için sinyal kullanabiliriz
        # Arka plan işleme thread'ini durdurmaya gerek YOK, sadece veriyi güncelliyoruz.
        # Eğer çalışıyorsa, bir sonraki _process_frame_internal çağrısında yeni veriyi kullanacak.
        success = self.load_encodings() # Bu metot zaten I/O yapar ve self.data'yı günceller
        if not success:
             self.log_signal.emit("Failed to load encodings during reload.")
        return success

    def stop_processing_thread(self):
        """Stop the background processing thread"""
        self.processing_active = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        # --- YENİ EKLENECEK METODLAR ---
    def start(self, mode='recognize'):
        """Starts the face detection/recognition processing."""
        pub.sendMessage('log', msg=f"FaceDetector starting in mode: {mode}")
        self.current_mode = mode
        try:
            # Arka plan thread'ini başlatmayı dene
            success = self.start_processing_thread() # Bu metodun True döndürmesi bekleniyor
            if success:
                pub.sendMessage('log', msg=f"FaceDetector thread started/verified successfully for mode {mode}.")
                return True # <<< BAŞARILI DÖN
            else:
                # Bu durum normalde olmamalı ama olursa logla
                pub.sendMessage('log:error', msg="start_processing_thread returned False unexpectedly.")
                return False # <<< BAŞARISIZ DÖN
        except Exception as e:
            # Thread başlatma sırasında bir hata oluşursa
            pub.sendMessage('log:error', msg=f"Exception during FaceDetector start/thread start: {e}")
            import traceback
            print(traceback.format_exc()) # Konsola tam hatayı yazdır
            return False # <<< BAŞARISIZ DÖN

    def stop(self):
        """Stops the face detection/recognition processing."""
        pub.sendMessage('log', msg="FaceDetector stopping...")
        self.current_mode = 'none' # Modu sıfırla
        self.stop_processing_thread()
    # --- YENİ EKLENECEK METODLAR SONU ---


    def handle_face_detection(self, detected_names):
        """Yüz tanıma olayında animasyon gönderir."""
        for name in detected_names:
            if name in self.face_detector.priority_animations:
                animation = self.face_detector.get_animation_for_person(name)
                self.send_animation(animation)
            
    def _background_processing(self):
        """Background thread using a multiprocessing pool for faster face detection."""
        while self.processing_active:
            try:
                try:
                    frame = self.frame_queue.get(block=False)
                except Empty:
                    time.sleep(0.005) # Kuyruk boşsa CPU yormamak için kısa bekleme
                    continue
                    
                # Yüz işleme fonksiyonunu ayrı bir işlemde çalıştır
                future = self.executor.submit(self._process_frame_internal, frame.copy())
                
                try:
                    # Sonucu al (timeout ile bekleme sınırı koymak iyi olur)
                    faces, names, priority_faces = future.result(timeout=0.5) # Örn: 0.5 saniye timeout

                    # --- DEĞİŞİKLİK: Zaman Damgasını Güncelle ---
                    # Sadece gerçekten yüz bulunduysa zaman damgasını güncelle
                    if faces:
                        self.last_valid_result_time = time.time()
                    # --- DEĞİŞİKLİK SONU ---

                except concurrent.futures.TimeoutError:
                    pub.sendMessage('log:warning', msg="Face processing task timed out.")
                    continue # Zaman aşımı olursa bu frame'i atla
                except Exception as e:
                    # İşlem sırasında hata olursa logla ve devam et
                    pub.sendMessage('log:error', msg=f"Error getting result from face processing task: {e}")
                    # Hata durumunda belki önceki sonuçları temizlemek iyi olabilir? (Opsiyonel)
                    # with self.results_lock:
                    #     self.current_faces, self.current_names, self.current_priority_faces = [], [], []
                    continue

                # Sonucu kuyruğa koy (GUI thread'i buradan alacak)
                if not self.result_queue.full():
                    self.result_queue.put((faces, names, priority_faces))
                
                # Ana önbelleği her zaman güncelle (sonucu kuyruğa koyduktan sonra)
                # Bu, result_queue'dan alınamasa bile en son işlenmiş sonucu tutar
                with self.results_lock:
                    self.current_faces = faces.copy() if faces else []
                    self.current_names = names.copy() if names else []
                    self.current_priority_faces = priority_faces.copy() if priority_faces else []
                    # self.last_recognition_time = time.time() # Bu satırın önemi azaldı, kaldırılabilir.

                # CPU kullanımını düşürmek için çok kısa bir bekleme
                time.sleep(0.005)

            except Empty: # frame_queue.get için Empty hatası (zaten yukarıda handle edildi ama yine de)
                time.sleep(0.005)
                continue
            except Exception as e:
                # Beklenmeyen genel hataları logla
                pub.sendMessage('log:error', msg=f"Exception in background processing loop: {e}")
                # Hata durumunda döngüyü kırmamak için biraz bekle
                time.sleep(0.01)
    
    def _process_frame_internal(self, frame):
        """Internal frame processing function (called in background thread)"""
        if frame is None:
            return [], [], []
        
        # Yüz tanıma için frame sayısı kontrolü ve hız optimizasyonu
        self.frame_count += 1
        
        # Her kareyi işleme => Arada kareleri atla ve sadece belirli aralıklarla işle
        if self.frame_count % self.process_every_nth_frame != 0:
            # Ara kareler için son sonuçları kullan
            return self.last_detection_results
        
        # Resimleri küçülterek performansı artır (daha küçük = daha hızlı)
        scale_percent = int(self.scale_factor * 100)  # 0.5 = %50
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        small_frame = cv2.resize(frame, (width, height))
        
        # RGB'ye çevir (face_recognition için)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Yüz tanıma başarısızsa cascade'e geçmeden önce bir kontrol
        if self.use_hog_always:
            # Daha hızlı olan HOG kullan (model=hog)
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        else:
            # Daha doğru olan CNN kullan
            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        
        # If no faces found with face_recognition, try with cascade as fallback
        if len(face_locations) == 0:
            # Try with cascade
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            cascade_faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )
            # İşlenmiş yüzleri ve sonuçları kaydet
            self.last_detection_results = (faces, names, priority_faces)
            return faces, names, priority_faces    
            # Convert cascade format to face_recognition format
            face_locations = []
            for (x, y, w, h) in cascade_faces:
                face_locations.append((y, x + w, y + h, x))  # top, right, bottom, left
        
        # Update detection counter for diagnostics
        self.face_detections += len(face_locations)
        
        # Recognize faces if we have encodings data
        names = []
        priority_faces = []  # List to track which faces are priority persons
        
        if self.data is not None and len(face_locations) > 0 and len(self.known_face_encodings) > 0:
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Match each face encoding to known faces
            for face_encoding in face_encodings:
                try:
                    # Check for NaN values which can break comparison
                    if np.isnan(face_encoding).any():
                        names.append("Unknown")
                        priority_faces.append(False)
                        continue
                    
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings,
                        face_encoding,
                        tolerance=self.recognition_tolerance
                    )
                    
                    # Find closest match using face distances
                    name = "Unknown"
                    is_priority = False
                    
                    # If any match found, find the best one
                    if True in matches:
                        # Calculate face distances for better matching
                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings,
                            face_encoding
                        )
                        
                        # Get the index of the closest match
                        best_match_index = np.argmin(face_distances)
                        
                        # Check if it's actually a match
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            is_priority = self.is_priority_person(name)
                            self.face_recognitions += 1
                        else:
                            name = "Unknown"
                            self.recognition_failures += 1
                    else:
                        self.recognition_failures += 1
                    
                    names.append(name)
                    priority_faces.append(is_priority)
                except Exception as e:
                    pub.sendMessage('log:error', msg=f"Error recognizing face: {e}")
                    names.append("Unknown")
                    priority_faces.append(False)
        
        # Convert face_locations back to (x,y,w,h) format (scaled back to original frame)
        faces = []
        for (top, right, bottom, left) in face_locations:
            # Scale coordinates back to original frame size
            x = int(left * 100 / scale_percent)
            y = int(top * 100 / scale_percent)
            w = int((right - left) * 100 / scale_percent)
            h = int((bottom - top) * 100 / scale_percent)
            faces.append((x, y, w, h))
            
        return faces, names, priority_faces
    
    def prepare_cached_encodings(self):
        """Prepare cached encodings from the loaded data for faster processing"""
        if self.data is None or "encodings" not in self.data or "names" not in self.data:
            self.known_face_encodings = []
            self.known_face_names = []
            return
            
        self.known_face_encodings = self.data["encodings"]
        self.known_face_names = self.data["names"]
        pub.sendMessage('log', msg=f"Prepared {len(self.known_face_encodings)} face encodings for recognition")
    
    def load_encodings(self):
        """Load face encodings from file with error handling"""
        if not self.encodings_file or not os.path.exists(self.encodings_file):
            pub.sendMessage('log:error', msg=f"Encodings file not found: {self.encodings_file}")
            self.data = None
            return False
            
        try:
            pub.sendMessage('log', msg=f"Loading encodings from {self.encodings_file}")
            with open(self.encodings_file, 'rb') as f:
                self.data = pickle.load(f)
                
            # Validate the loaded data
            if not isinstance(self.data, dict) or 'encodings' not in self.data or 'names' not in self.data:
                pub.sendMessage('log:error', msg=f"Invalid encodings file format")
                self.data = None
                return False
                
            # Print diagnostics
            encoding_count = len(self.data["encodings"]) if self.data and "encodings" in self.data else 0
            names_count = len(self.data["names"]) if self.data and "names" in self.data else 0
            unique_names = set(self.data["names"]) if self.data and "names" in self.data else set()
            
            # Verify encoding shapes are correct (should be 128-dimensional vectors)
            valid_encodings = 0
            invalid_encodings = 0
            nan_encodings = 0
            
            if encoding_count > 0:
                for encoding in self.data["encodings"]:
                    if len(encoding) != 128:
                        invalid_encodings += 1
                    elif np.isnan(encoding).any():
                        nan_encodings += 1
                    else:
                        valid_encodings += 1
            
            pub.sendMessage('log', msg=f"Loaded {encoding_count} encodings for {len(unique_names)} unique persons: {', '.join(unique_names)}")
            self.prepare_cached_encodings()
            return encoding_count > 0 and valid_encodings > 0
        except Exception as e:
            pub.sendMessage('log:error', msg=f"Error loading encodings: {e}")
            self.data = None
            return False
    
    def load_priority_persons(self):
        """Load priority persons from JSON file with order information"""
        try:
            if os.path.exists(self.priority_file):
                with open(self.priority_file, 'r') as f:
                    data = json.load(f)
                    self.priority_persons = data.get("priority_persons", [])
                    
                    # Load order if available, otherwise assign default order
                    self.priority_order = data.get("priority_order", {})
                    
                    # Make sure all priority persons have an order
                    for i, person in enumerate(self.priority_persons):
                        if person not in self.priority_order:
                            self.priority_order[person] = i + 1  # Start from 1
                            
                pub.sendMessage('log', msg=f"Loaded {len(self.priority_persons)} priority persons with order information")
            else:
                self.priority_persons = []
                self.priority_order = {}
                pub.sendMessage('log', msg="No priority persons file found, starting with empty list")
        except Exception as e:
            pub.sendMessage('log:error', msg=f"Error loading priority persons: {e}")
            self.priority_persons = []
            self.priority_order = {}
    
    def save_priority_persons(self):
        """Save priority persons to JSON file with order information"""
        try:
            with open(self.priority_file, 'w') as f:
                json.dump({
                    "priority_persons": self.priority_persons,
                    "priority_order": self.priority_order
                }, f, indent=2)
            pub.sendMessage('log', msg=f"Saved {len(self.priority_persons)} priority persons with order information")
            return True
        except Exception as e:
            pub.sendMessage('log:error', msg=f"Error saving priority persons: {e}")
            return False
    
    def add_priority_person(self, name, order=None):
        """Add a person to the priority list with optional order"""
        if name in self.priority_persons:
            # If already in list but order is provided, update the order
            if order is not None:
                self.priority_order[name] = order
                self.save_priority_persons()
                return True
            return False
            
        self.priority_persons.append(name)
        
        # Assign order if provided, otherwise use the next available number
        if order is not None:
            self.priority_order[name] = order
        else:
            max_order = 0
            for _, existing_order in self.priority_order.items():
                max_order = max(max_order, existing_order)
            self.priority_order[name] = max_order + 1
            
        self.save_priority_persons()
        return True
    
    def remove_priority_person(self, name):
        """Remove a person from the priority list"""
        if name not in self.priority_persons:
            return False
        self.priority_persons.remove(name)
        if name in self.priority_order:
            del self.priority_order[name]
        self.save_priority_persons()
        return True
    
    def is_priority_person(self, name):
        """Check if a person is in the priority list"""
        return name in self.priority_persons
    
    def get_priority_order(self, name):
        """Get the priority order of a person (lower number = higher priority)"""
        if name in self.priority_order:
            return self.priority_order[name]
        return float('inf')  # Return infinity for non-priority persons
    
    def set_priority_order(self, name, order):
        """Set the priority order for a person"""
        if name in self.priority_persons:
            self.priority_order[name] = order
            self.save_priority_persons()
            return True
        return False
    
    def get_highest_priority_person(self, names):
        """Get the highest priority person from a list of names"""
        highest_priority = float('inf')
        highest_priority_person = None
        
        for name in names:
            if name == "Unknown":
                continue
                
            priority = self.get_priority_order(name)
            if priority < highest_priority:
                highest_priority = priority
                highest_priority_person = name
                
        return highest_priority_person

    def detect_faces(self, frame):
        """Queue a frame for processing and return the most recent results with staleness check.
        Yüz kaybolduktan sonra kutuları bir süre daha gösterir."""
        if frame is None:
            return [], [], []

        # --- Frame Kuyruğa Ekleme ---
        if self.frame_queue.empty():
            try:
                self.frame_queue.put(frame.copy(), block=False)
            except Queue.Full:
                pass

        local_faces, local_names, local_priority_faces = [], [], []
        new_result_processed = False

        try:
            if not self.result_queue.empty():
                fetched_faces, fetched_names, fetched_priority_faces = self.result_queue.get(block=False)
                new_result_processed = True
                with self.results_lock:
                    self.current_faces = fetched_faces
                    self.current_names = fetched_names
                    self.current_priority_faces = fetched_priority_faces
                local_faces = fetched_faces
                local_names = fetched_names
                local_priority_faces = fetched_priority_faces
        except Empty:
            pass
        except Exception as e:
            pub.sendMessage('log:error', msg=f"Error getting result from result_queue: {e}")

        if not new_result_processed:
            with self.results_lock:
                local_faces = self.current_faces
                local_names = self.current_names
                local_priority_faces = self.current_priority_faces

        # --- YENİ: Son yüzleri ve zamanı güncelle ---
        now = time.time()
        if local_faces and len(local_faces) > 0:
            self.last_faces = local_faces
            self.last_names = local_names
            self.last_priority_faces = local_priority_faces
            self.last_faces_time = now
        else:
            # Eğer yeni yüz yoksa, eski yüzleri belirli bir süre daha göster
            if self.last_faces and (now - self.last_faces_time < self.max_persist_time):
                local_faces = self.last_faces
                local_names = self.last_names
                local_priority_faces = self.last_priority_faces
            else:
                # Süre dolduysa kutuları temizle
                self.last_faces = []
                self.last_names = []
                self.last_priority_faces = []
                self.last_faces_time = 0
                local_faces = []
                local_names = []
                local_priority_faces = []

        return local_faces, local_names, local_priority_faces

    def process_frame(self, frame):
        """İyileştirilmiş ve optimize edilmiş yüz tanıma işlemi"""
        if frame is None:
            return frame, [], [], []
        
        # Her kareyi işleme (çok yoğun CPU kullanımı) - her N kare yerine sadece belirli kareleri işle
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
            self.last_faces = []
            self.last_names = []
            self.last_priority_faces = []
        
        self.frame_count += 1
        
        # Her 4 karede bir yüz tanıma yap (CPU'yu rahatlatır)
        if self.frame_count % 4 != 0:
            # Son bilinen sonuçları kullan, tekrar hesaplama
            return self._draw_results_on_frame(frame.copy(), 
                                              self.last_faces, 
                                              self.last_names, 
                                              self.last_priority_faces), \
                   self.last_faces, self.last_names, self.last_priority_faces
        
        # Performans için resmi küçült - 1/2 boyuta
        if frame.shape[0] > 300:
            scale_factor = 0.5
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        else:
            small_frame = frame
            scale_factor = 1.0
        
        # RGB çevir (face_recognition için)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Yüz konumlarını HOG ile bul (daha hızlı)
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        
        # Orijinal görüntüdeki konuma çevir
        if scale_factor != 1.0:
            face_locations = [(int(top/scale_factor), int(right/scale_factor), 
                              int(bottom/scale_factor), int(left/scale_factor)) 
                              for top, right, bottom, left in face_locations]
        
        # Sonuçları işle
        faces, names, priority_faces = self._process_detected_faces(frame, face_locations)
        
        # DeskGUI içindeki face detection kodu içinde
        if self.command_sender and self.command_sender.connected:
            # Öncelikli kişi varsa
            if priority_name:
                animation = self.priority_animations.get(priority_name, 'RAINBOW')
                self.command_sender.send_command("face_detected", {
                    "name": priority_name,
                    "is_priority": True,
                    "animation": animation
                })
            # Öncelikli kişi algılandığında robota bildir
             # Define is_priority and priority_name based on application logic
            is_priority = False  # Example default value
            priority_name = None  # Example default value

            # Example logic to set is_priority and priority_name
            if self.last_detected_names:
                priority_name = self.face_detector.get_highest_priority_person(self.last_detected_names)
                is_priority = priority_name is not None

            if is_priority and priority_name:
                animation = self.face_detector.priority_animations.get(priority_name, 'RAINBOW')
                clean_priority_name = priority_name.replace('â­ ', '').strip() # Yıldızı ve olası bozuk karakterleri temizle
                # >>> BU LOG ÇOK ÖNEMLİ <<<
                self.log(f"DEBUG: >>> INTENDING TO SEND 'face_detected' for PRIORITY: {clean_priority_name}, Anim: {animation}")
                # >>> BU LOG ÇOK ÖNEMLİ <<<
                self.command_sender.send_command("face_detected", {
                    "name": clean_priority_name,
                    "is_priority": True,
                    "animation": animation
                })
            elif names and len(names) > 0:
                # En azından ismi gönder
                self.command_sender.send_command("face_detected", {
                    "name": names[0],
                    "is_priority": False
                })
        
        # Sonuçları ön belleğe al
        self.last_faces = faces
        self.last_names = names
        self.last_priority_faces = priority_faces
        
        # Çizimleri yap ve görüntüyü döndür
        return self._draw_results_on_frame(frame.copy(), faces, names, priority_faces), \
               faces, names, priority_faces
            
    def reload_data(self):
        """Reload face recognition data (after training)"""
        self.load_encodings()
        
    def __del__(self):
        """Cleanup resources when object is destroyed"""
        self.stop_processing_thread()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
            
    def detect_faces_simple(self, frame):
        """Sadece Haar Cascade kullanarak yüzleri bulur ve çerçeve çizer."""
        if frame is None:
            return frame, [] # Değiştirilmiş kare ve boş yüz listesi döndür

        # Gri tonlamaya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Yüzleri bul (Haar Cascade ile)
        # scaleFactor, minNeighbors gibi değerler FaceDetector'ın __init__ kısmındaki
        # değerlerle aynı olabilir veya farklı ayarlanabilir.
        faces_rects = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # Örnekteki gibi veya self.scale_factor kullan
            minNeighbors=5,   # Örnekteki gibi veya self.min_neighbors kullan
            minSize=(30, 30)  # Örnekteki gibi veya self.min_size kullan
        )

        # Bulunan her yüz için dikdörtgen çiz
        output_frame = frame.copy() # Orijinal kareyi kopyala
        detected_faces_list = [] # Tespit edilen yüzlerin koordinat listesi
        for (x, y, w, h) in faces_rects:
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Yeşil çerçeve
            detected_faces_list.append((x, y, w, h))

        # İşlenmiş kareyi ve yüzlerin listesini döndür
        return output_frame, detected_faces_list