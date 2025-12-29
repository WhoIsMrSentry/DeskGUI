import cv2
import numpy as np
import os
import time
from pubsub import pub
from ..vision import get_model_path

class ObjectTracker:
    def __init__(self, command_sender=None):
        """Nesne takip modülü."""
        self.command_sender = command_sender
        self.processing_active = False
        
        # Tracker algoritmaları
        self.tracker_types = {
            'CSRT': cv2.legacy.TrackerCSRT_create,
            'KCF': cv2.legacy.TrackerKCF_create,
            'MOSSE': cv2.legacy.TrackerMOSSE_create
        }
        
        # Varsayılan tracker
        self.tracker_type = 'CSRT'
        self.tracker = None
        self.tracking_bbox = None
        self.tracking_success = False
        self.tracking_object_class = None
        
        # Takip çerçevesi ve geçmiş konumlar
        self.bbox_history = []
        self.max_history = 20  # Maksimum tarihçe sayısı
        self.tracking_started = False
        
        # Takip merkezleme bilgileri
        self.frame_center_x = 0
        self.frame_center_y = 0
        self.target_center_x = 0
        self.target_center_y = 0
        self.center_threshold = 50  # Merkezden sapma eşiği (piksel)
        
        # Servo kontrol parametreleri
        self.last_command_time = 0
        self.command_interval = 0.5  # Komut gönderme aralığı (saniye)
        
        self.log(f"ObjectTracker başlatıldı, varsayılan tracker: {self.tracker_type}")

    def start(self):
        """İşlemeyi başlat."""
        self.processing_active = True
        self.log("Nesne takibi başlatıldı")
        return True

    def stop(self):
        """İşlemeyi durdur."""
        self.processing_active = False
        self.reset_tracking()
        self.log("Nesne takibi durduruldu")

    def reset_tracking(self):
        """Takip durumunu sıfırla."""
        self.tracker = None
        self.tracking_bbox = None
        self.tracking_success = False
        self.tracking_object_class = None
        self.bbox_history.clear()
        self.tracking_started = False
        self.log("Takip sıfırlandı")

    def init_tracker(self, frame, bbox, object_class=None):
        """Tracker'ı başlat."""
        if frame is None or bbox is None:
            return False
            
        try:
            # Yeni bir tracker oluştur
            self.tracker = self.tracker_types[self.tracker_type]()
            
            # Tracker'ı ilk çerçeve ve kutu ile başlat
            x, y, w, h = [int(v) for v in bbox]
            init_bbox = (x, y, w, h)
            self.tracking_success = self.tracker.init(frame, init_bbox)
            
            if self.tracking_success:
                self.tracking_bbox = init_bbox
                self.tracking_object_class = object_class
                self.bbox_history.clear()
                self.bbox_history.append(init_bbox)
                self.tracking_started = True
                self.log(f"Tracker başlatıldı: {self.tracker_type} - Sınıf: {object_class}")
                return True
            else:
                self.log(f"Tracker başlatılamadı!")
                return False
                
        except Exception as e:
            self.log(f"Tracker başlatılırken hata: {e}")
            return False

    def process_frame(self, frame, detection_results=None):
        """Bir kareyi işle ve nesne takibi yap. Kutu sürekli görünür."""
        if not self.processing_active or frame is None:
            return frame, None

        processed_frame = frame.copy()
        frame_height, frame_width = processed_frame.shape[:2]
        self.frame_center_x = frame_width // 2
        self.frame_center_y = frame_height // 2

        # --- Otomatik tracker başlatma (aynı kalabilir) ---
        if detection_results and not self.tracking_started:
            try:
                detections = detection_results.get('detections', [])
                selected_target = None
                # ... (insan veya ilk nesneyi seçme mantığı) ...
                if selected_target:
                    success = self.init_tracker(
                        processed_frame,
                        selected_target['box'],
                        selected_target['label']
                    )
                    if success:
                        self.log(f"Yeni nesne takip edilmeye başlandı: {selected_target['label']}")
            except Exception as e:
                self.log(f"Otomatik tracker başlatma hatası: {e}")
        # --- Otomatik tracker başlatma sonu ---

        # --- Takip Güncelleme ve Çizim (DEĞİŞTİRİLDİ) ---
        current_bbox_to_draw = None # Bu karede çizilecek kutu
        object_label_to_show = None # Gösterilecek etiket

        if self.tracking_started and self.tracker:
            try:
                # Tracker'ı güncelle
                success, new_bbox = self.tracker.update(processed_frame)
                self.tracking_success = success # Başarı durumunu sakla (döndürmek için)

                if self.tracking_success:
                    # Başarılı: Yeni konumu kullan ve sakla
                    self.tracking_bbox = tuple(int(v) for v in new_bbox)
                    current_bbox_to_draw = self.tracking_bbox
                    # Başarı durumunda geçmişi güncelle
                    self.bbox_history.append(self.tracking_bbox)
                    if len(self.bbox_history) > self.max_history:
                        self.bbox_history.pop(0)
                elif self.tracking_bbox:
                    # Başarısız: Son bilinen konumu kullan (varsa)
                    current_bbox_to_draw = self.tracking_bbox
                    # self.log("Tracker update failed, using last known bbox for drawing.") # İsteğe bağlı log
                # else: Başarısız ve son bilinen konum da yoksa çizilecek bir şey yok

                # Eğer çizilecek bir kutu varsa (başarılı veya başarısız fark etmez)
                if current_bbox_to_draw:
                    x, y, w, h = current_bbox_to_draw

                    # Başarı durumuna göre renk belirle
                    draw_color = (0, 255, 0) if self.tracking_success else (0, 165, 255) # Yeşil (OK) / Turuncu (Kayıp)

                    # Kutuyu çiz
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), draw_color, 2)

                    # Etiketi oluştur (Her zaman takip edilen sınıfı göster)
                    if self.tracking_object_class:
                        status_text = "(Locked)" if self.tracking_success else "(Lost)"
                        object_label_to_show = f"{self.tracking_object_class} {status_text}"
                        cv2.putText(processed_frame, object_label_to_show, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

                    # Geçmişi ve hedef merkezini çizilecek kutuya göre hesapla
                    self._draw_tracking_history(processed_frame) # Geçmişi bbox_history'den çizer
                    self.target_center_x = x + w // 2
                    self.target_center_y = y + h // 2
                    self._calculate_servo_position(processed_frame) # Servo hedefini ayarla

                # Eğer güncelleme başarısız olduysa ve DeskGUI hemen kilidi bırakmıyorsa,
                # kullanıcıya takip kaybını bildiren bir mesaj gösterilebilir (Opsiyonel)
                # if not self.tracking_success:
                #     cv2.putText(processed_frame, "Takip kaybedildi!", (20, 60),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # NOT: DeskGUI artık hemen kilidi bırakacağı için bu mesaja gerek kalmayabilir.

            except Exception as e:
                self.log(f"Takip güncellenirken hata: {e}")
                self.reset_tracking() # Hata durumunda sıfırla
                # Hata durumunda None döndürerek DeskGUI'nin bilmesini sağla
                return processed_frame, None

        # Dönen değerler: İşlenmiş kare ve takip durumu sözlüğü
        # Sözlük, DeskGUI'nin takip durumunu anlamasına yardımcı olur
        return processed_frame, {
            'tracking': self.tracking_started,
            'success': self.tracking_success,       # Bu frame'deki GÜNCELLEME başarısı
            'bbox': self.tracking_bbox,            # En son BAŞARILI takip kutusu
            'object_class': self.tracking_object_class
        }
    
    def _draw_tracking_history(self, frame):
        """Takip geçmişini çiz (hareket izi)."""
        if len(self.bbox_history) < 2:
            return
            
        # Önceki merkez noktaları
        centers = []
        for box in self.bbox_history:
            x, y, w, h = box
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))
        
        # Noktaları çiz
        for i in range(len(centers) - 1):
            # Renk gradyanı oluştur (eskiden yeniye doğru)
            alpha = i / (len(centers) - 1)  # 0 ile 1 arası
            color = (
                int(255 * (1 - alpha)),  # B
                int(255 * alpha),        # G
                0                        # R
            )
            
            cv2.line(frame, centers[i], centers[i+1], color, 2)
            cv2.circle(frame, centers[i], 3, color, -1)
        
        # Son noktayı işaretle
        cv2.circle(frame, centers[-1], 5, (0, 255, 0), -1)

    def _calculate_servo_position(self, frame):
        """Servo pozisyonunu hesapla ve gerekirse komutu gönder."""
        if not self.tracking_success or not self.command_sender or not self.command_sender.connected:
            return
            
        # Ekranın ortasına hedef çiz
        cv2.circle(frame, (self.frame_center_x, self.frame_center_y), 5, (255, 0, 0), -1)
        cv2.line(frame, (self.frame_center_x - 10, self.frame_center_y), 
                (self.frame_center_x + 10, self.frame_center_y), (255, 0, 0), 2)
        cv2.line(frame, (self.frame_center_x, self.frame_center_y - 10),
                (self.frame_center_x, self.frame_center_y + 10), (255, 0, 0), 2)
        
        # Geçerli hedef nesnenin merkezine çizgi çiz
        cv2.line(frame, (self.frame_center_x, self.frame_center_y), 
                (self.target_center_x, self.target_center_y), (0, 255, 255), 2)
        
        # Hareket miktarını hesapla
        dx = self.target_center_x - self.frame_center_x
        dy = self.target_center_y - self.frame_center_y
        
        # Eğer merkezdeki sapma yeteri kadar büyükse ve komut gönderme aralığı geçtiyse
        current_time = time.time()
        if (abs(dx) > self.center_threshold or abs(dy) > self.center_threshold) and \
           (current_time - self.last_command_time > self.command_interval):
            
            # Pan (x-ekseni) ve tilt (y-ekseni) hareket yönlerini belirle
            pan_dir = -1 if dx > 0 else 1  # Sağa doğru ise sola pan (-1), sola doğru ise sağa pan (1)
            tilt_dir = -1 if dy < 0 else 1  # Aşağı doğru ise yukarı tilt (-1), yukarı doğru ise aşağı tilt (1)
            
            # Hareketi normalleştir (% olarak) - daha büyük sapma, daha büyük hareket
            pan_amount = min(100, abs(dx) // 10)  # 0-100 arası değer
            tilt_amount = min(100, abs(dy) // 10)  # 0-100 arası değer
            
            # Servo hareket bilgisini çiz
            direction_text = f"Pan: {pan_dir*pan_amount:+d}%, Tilt: {tilt_dir*tilt_amount:+d}%"
            cv2.putText(frame, direction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Robota hareket komutlarını gönder
            try:
                if abs(dx) > self.center_threshold:
                    self.command_sender.send_command('servo_move', {
                        'identifier': 'pan',
                        'percentage': int(pan_dir * pan_amount),
                        'absolute': False
                    })
                    
                if abs(dy) > self.center_threshold:
                    self.command_sender.send_command('servo_move', {
                        'identifier': 'tilt',
                        'percentage': int(tilt_dir * tilt_amount),
                        'absolute': False
                    })
                    
                self.last_command_time = current_time
                self.log(f"Servo hareketi: Pan={pan_dir*pan_amount:+d}%, Tilt={tilt_dir*tilt_amount:+d}%")
                
            except Exception as e:
                self.log(f"Servo komutu gönderilirken hata: {e}")

    def set_tracker_type(self, tracker_type):
        """Tracker tipini değiştir."""
        if tracker_type in self.tracker_types:
            self.tracker_type = tracker_type
            self.log(f"Tracker tipi değiştirildi: {tracker_type}")
            return True
        else:
            self.log(f"Geçersiz tracker tipi: {tracker_type}, mevcut tipler: {list(self.tracker_types.keys())}")
            return False

    def log(self, message):
        """Loglama işlemlerini yapar."""
        pub.sendMessage('log', msg=f"[ObjectTracker] {message}")
