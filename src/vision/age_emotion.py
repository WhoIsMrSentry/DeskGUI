# age_emotion.py
import cv2
import numpy as np
import os
import time
from pubsub import pub
from ..vision import get_model_path
import math

class AgeEmotionDetector:
    def __init__(self, command_sender=None):
        """Yaş ve duygu tespit modülü."""
        self.command_sender = command_sender
        self.processing_active = False
        self.log = lambda msg: pub.sendMessage('log', msg=f"[AgeEmotion] {msg}")  # Basit loglama

        # Model ortalamaları ve listeler
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.emotion_list = ['Notr', 'Mutlu', 'Saskin', 'Uzgun', 'Kizgin', 'Igrenmis', 'Korkmus', 'Kucumseme']

        # Yüz tespiti için DNN modelini yükle
        self.face_detector_model = get_model_path('face_detection_yunet_2023mar.onnx')
        if not os.path.exists(self.face_detector_model):
            self.log(f"UYARI: Yüz tespit modeli bulunamadı: {self.face_detector_model}")
            self.face_detector = None
        else:
            try:
                self.face_detector = cv2.FaceDetectorYN_create(self.face_detector_model, "", (0, 0))
                self.log("Yüz tespit modeli başarıyla yüklendi.")
            except Exception as e:
                self.log(f"HATA: Yüz tespit modeli yüklenemedi: {e}")
                self.face_detector = None

        # Yaş modeli
        self.age_model_path = get_model_path('age_net.caffemodel')
        self.age_proto_path = get_model_path('age_deploy.prototxt')
        if not os.path.exists(self.age_model_path) or not os.path.exists(self.age_proto_path):
            self.log(f"UYARI: Yaş tespit modeli dosyaları bulunamadı.")
            self.age_net = None
        else:
            try:
                self.age_net = cv2.dnn.readNet(self.age_model_path, self.age_proto_path)
                self.log("Yaş tespit modeli başarıyla yüklendi.")
            except Exception as e:
                self.log(f"HATA: Yaş tespit modeli yüklenemedi: {e}")
                self.age_net = None

        # Duygu modeli (Yeni model adı)
        self.emotion_model_path = get_model_path('emotion-ferplus-8.onnx')
        if not os.path.exists(self.emotion_model_path):
            self.log(f"UYARI: Duygu tespit modeli bulunamadı: {self.emotion_model_path}")
            self.emotion_net = None
        else:
            try:
                self.emotion_net = cv2.dnn.readNet(self.emotion_model_path)
                self.log("Duygu tespit modeli başarıyla yüklendi.")
            except Exception as e:
                self.log(f"HATA: Duygu tespit modeli yüklenemedi: {e}")
                self.emotion_net = None

        self.models_loaded = self.face_detector and self.age_net and self.emotion_net
        if not self.models_loaded:
            self.log("UYARI: Tüm modeller yüklenemediği için yaş/duygu tespiti çalışmayacak.")

        self.padding = 20
        self.last_detection_time = 0
        self.detection_interval = 0.5

        # --- YENİ: Son tespit edilen yüzleri ve zamanı sakla ---
        self.last_results = []
        self.last_results_time = 0
        self.max_persist_time = 1.5  # saniye, yüz kaybolduktan sonra kutu ne kadar ekranda kalsın

    def start(self):
        if not self.models_loaded:
            self.log("Modeller yüklenemediği için başlatılamıyor.")
            return False
        self.log("Yaş ve duygu tespiti başlatılıyor...")
        self.processing_active = True
        return True

    def stop(self):
        self.log("Yaş ve duygu tespiti durduruluyor...")
        self.processing_active = False

    def process_frame(self, frame):
        """Gelen çerçevede yaş ve duygu tespiti yapar ve (frame, detected_data) döndürür."""
        if not self.processing_active or not self.models_loaded or frame is None:
            # --- YENİ: Son tespit edilen yüzleri göster ---
            now = time.time()
            if self.last_results and (now - self.last_results_time < self.max_persist_time):
                processed_frame = frame.copy()
                for res in self.last_results:
                    x, y, w, h = res['bbox']
                    age = res['age']
                    emotion = res['emotion']
                    confidence = res.get('confidence', 0)
                    age_label = f"Yas: {age}"
                    emotion_label = f"Duygu: {emotion}"
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label_text = f"{age_label}, {emotion_label}"
                    cv2.putText(processed_frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                return processed_frame, self.last_results
            else:
                self.last_results = []
                return frame, []

        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            # --- YENİ: Son tespit edilen yüzleri göster ---
            if self.last_results and (current_time - self.last_results_time < self.max_persist_time):
                processed_frame = frame.copy()
                for res in self.last_results:
                    x, y, w, h = res['bbox']
                    age = res['age']
                    emotion = res['emotion']
                    confidence = res.get('confidence', 0)
                    age_label = f"Yas: {age}"
                    emotion_label = f"Duygu: {emotion}"
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label_text = f"{age_label}, {emotion_label}"
                    cv2.putText(processed_frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                return processed_frame, self.last_results
            else:
                self.last_results = []
                return frame, []

        self.last_detection_time = current_time
        processed_frame = frame.copy()
        frame_height, frame_width = processed_frame.shape[:2]
        detected_results_list = []

        if self.face_detector:
            self.face_detector.setInputSize((frame_width, frame_height))
            try:
                faces = self.face_detector.detect(processed_frame)
                if faces[1] is not None:
                    for face_data in faces[1]:
                        confidence = face_data[-1]
                        if confidence < 0.7:
                            continue
                        box = face_data[0:4].astype(np.int32)
                        x, y, w, h = box
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        x1 = max(0, x1 - self.padding)
                        y1 = max(0, y1 - self.padding)
                        x2 = min(frame_width - 1, x2 + self.padding)
                        y2 = min(frame_height - 1, y2 + self.padding)
                        face_roi_color = processed_frame[y1:y2, x1:x2]
                        if face_roi_color.size == 0:
                            continue
                        age = "Unknown"
                        emotion = "Unknown"
                        age_label = "Yas: ?"
                        emotion_label = "Duygu: ?"
                        if self.age_net:
                            try:
                                age_blob = cv2.dnn.blobFromImage(face_roi_color, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
                                self.age_net.setInput(age_blob)
                                age_preds = self.age_net.forward()
                                age = self.age_list[age_preds[0].argmax()]
                                age_label = f"Yas: {age}"
                            except Exception as age_e:
                                self.log(f"Yaş tespiti hatası: {age_e}")
                                age = "Error"
                        if self.emotion_net:
                            try:
                                face_roi_gray = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2GRAY)
                                emotion_roi = cv2.resize(face_roi_gray, (64, 64))
                                emotion_blob = cv2.dnn.blobFromImage(emotion_roi, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
                                self.emotion_net.setInput(emotion_blob)
                                emotion_preds = self.emotion_net.forward()
                                emotion_index = np.argmax(emotion_preds[0])
                                emotion = self.emotion_list[emotion_index]
                                emotion_label = f"Duygu: {emotion}"
                            except Exception as emotion_e:
                                self.log(f"Duygu tespiti hatası: {emotion_e}")
                                emotion = "Error"
                        detected_results_list.append({
                            'bbox': (x1, y1, x2 - x1, y2 - y1),
                            'age': age,
                            'emotion': emotion,
                            'confidence': float(confidence)
                        })
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label_text = f"{age_label}, {emotion_label}"
                        cv2.putText(processed_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                self.log(f"Çerçeve işlenirken hata oluştu: {e}")
                # --- YENİ: Son tespit edilen yüzleri göster ---
                if self.last_results and (current_time - self.last_results_time < self.max_persist_time):
                    processed_frame = frame.copy()
                    for res in self.last_results:
                        x, y, w, h = res['bbox']
                        age = res['age']
                        emotion = res['emotion']
                        confidence = res.get('confidence', 0)
                        age_label = f"Yas: {age}"
                        emotion_label = f"Duygu: {emotion}"
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label_text = f"{age_label}, {emotion_label}"
                        cv2.putText(processed_frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    return processed_frame, self.last_results
                else:
                    self.last_results = []
                    return processed_frame, []

        # --- YENİ: Sonuçları güncelle ---
        if detected_results_list:
            self.last_results = detected_results_list
            self.last_results_time = current_time
        else:
            # Eğer yeni tespit yoksa, eskiyi göster
            if self.last_results and (current_time - self.last_results_time < self.max_persist_time):
                processed_frame = frame.copy()
                for res in self.last_results:
                    x, y, w, h = res['bbox']
                    age = res['age']
                    emotion = res['emotion']
                    confidence = res.get('confidence', 0)
                    age_label = f"Yas: {age}"
                    emotion_label = f"Duygu: {emotion}"
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label_text = f"{age_label}, {emotion_label}"
                    cv2.putText(processed_frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                return processed_frame, self.last_results
            else:
                self.last_results = []
                return processed_frame, []

        return processed_frame, detected_results_list

    def __del__(self):
        self.stop()
        self.log("AgeEmotionDetector temizlendi.")