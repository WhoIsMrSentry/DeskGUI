import cv2
import numpy as np

class MotionDetector:
    def __init__(self):
        # self.static_back = None # Bu yaklaşım sabit kamera varsayar, MOG2 daha iyi olabilir
        self.motion_threshold = 30 # Bu MOG2 için gerekmeyebilir
        self.min_area = 500 # Alan eşiği ayarlanabilir
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True) # MOG2 kullanmak daha robust
        self.processing_active = False # İşlem bayrağı

    # !!! detect_motion METODUNU process_frame OLARAK YENİDEN ADLANDIR VE GÜNCELLE !!!
    def process_frame(self, frame):
        """Processes a single frame for motion detection if active."""
        if not self.processing_active or frame is None:
            # Aktif değilse veya frame yoksa orijinal frame ve boş veri döndür
            return frame, {'detected': False, 'areas': []}

        motion_areas = []
        motion_detected = False
        processed_bgr_frame = frame.copy() # BGR üzerinde çalışalım

        try:
            # MOG2 kullanarak ön plan maskesini al
            fg_mask = self.background_subtractor.apply(processed_bgr_frame)

            # Gölge piksellerini griye çevir (isteğe bağlı)
            # fg_mask[fg_mask == 127] = 0 # Gölge piksellerini yok say

            # Gürültüyü azaltma ve boşlukları doldurma
            thresh_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1] # Eşiği ayarla
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            morph_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            morph_mask = cv2.dilate(morph_mask, kernel, iterations=2) # Biraz genişlet

            # Konturları bul
            contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Konturları filtrele ve çiz
            for contour in contours:
                if cv2.contourArea(contour) < self.min_area:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(processed_bgr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Yeşil dikdörtgen
                motion_areas.append((x, y, w, h))
                motion_detected = True

        except Exception as e:
             # print(f"Motion detection error in process_frame: {e}") # Çok fazla log olmaması için kapatılabilir
             # Hata durumunda orijinal frame ve boş veri döndür
             return frame, {'detected': False, 'areas': []}

        # Beklenen formatta sonucu döndür
        return processed_bgr_frame, {'detected': motion_detected, 'areas': motion_areas}

    def start(self):
        """Starts motion detection processing."""
        print("MotionDetector starting...")
        self.processing_active = True
        # MOG2'yi sıfırlamak için yeni bir örnek oluşturabiliriz (isteğe bağlı)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        return True

    def stop(self):
        """Stops motion detection processing."""
        print("MotionDetector stopping...")
        self.processing_active = False

    # Bu metod artık kullanılmıyor, process_frame içinde yapılıyor
    # def handle_motion_detection(self, motion_detected):
    #     """Hareket algılama olayında animasyon gönderir."""
    #     if motion_detected:
    #         self.send_animation("alert_red") # self.send_animation burada yok!
    #     else:
    #         self.send_animation("calm_blue") # Bu mantık DeskGUI içinde olmalı