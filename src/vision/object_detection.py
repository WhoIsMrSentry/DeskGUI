import cv2
import numpy as np
import os
import time
from pubsub import pub
from ..vision import get_model_path

class ObjectDetector:
    def __init__(self, command_sender=None):
        """Nesne tespiti modülü."""
        self.command_sender = command_sender
        self.processing_active = False
        
        # Model yolları
        self.yolo_model_path = get_model_path('yolov8n.onnx')
        self.classes_path = get_model_path('coco.names')
        self.log(f"Attempting to load YOLO model from: {self.yolo_model_path}")
        self.log(f"Attempting to load classes from: {self.classes_path}")
        
        # COCO sınıf isimleri - classes.txt dosyası bulunamazsa varsayılan olarak kullan
        self.classes = self._load_classes()
        
        # YOLO modeli yükleme
        if not os.path.exists(self.yolo_model_path):
            self.log(f"HATA: YOLO modeli bulunamadı: {self.yolo_model_path}")
            self.model = None
        elif not os.path.isfile(self.yolo_model_path):
            self.log(f"HATA: YOLO modeli bir dosya değil: {self.yolo_model_path}")
            self.model = None
        elif os.path.getsize(self.yolo_model_path) == 0:
            self.log(f"HATA: YOLO modeli dosyası boş: {self.yolo_model_path}")
            self.model = None
        else:
            try:
                self.log(f"YOLO model dosya boyutu: {os.path.getsize(self.yolo_model_path)} bayt")
                self.model = cv2.dnn.readNet(self.yolo_model_path)
                self.log(f"YOLO modeli yüklendi: {self.yolo_model_path}")
            except Exception as e:
                import traceback
                self.log(f"Model yüklenirken hata: {e}")
                self.log(traceback.format_exc())
                self.model = None
        
        # Algılama parametreleri
        self.conf_threshold = 0.5  # Güven eşiği
        self.nms_threshold = 0.4   # Non-maximum suppression eşiği
        self.input_width = 640     # Model giriş genişliği
        self.input_height = 640    # Model giriş yüksekliği
        
        # Son tespit bilgileri
        self.last_detection_time = 0
        self.min_detection_interval = 1.0  # Saniye
        self.last_detected_objects = set()
        
        self.log("ObjectDetector başlatıldı")
        
    def _load_classes(self):
        """COCO sınıf isimlerini yükle."""
        try:
            if os.path.exists(self.classes_path):
                with open(self.classes_path, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
                self.log(f"{len(classes)} sınıf yüklendi")
                return classes
            else:
                self.log(f"Sınıf dosyası bulunamadı: {self.classes_path}, varsayılan liste kullanılacak")
                # COCO veri setindeki 80 sınıfı içeren varsayılan liste
                return ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        except Exception as e:
            self.log(f"Sınıf dosyası yüklenirken hata: {e}")
            return ["unknown"]  # Hata durumunda tek bir sınıf döndür

    def start(self):
        """İşlemeyi başlat."""
        if self.model is None:
            self.log("YOLO modeli yüklenemediği için başlatılamıyor")
            return False
            
        self.processing_active = True
        self.log("Nesne tespiti başlatıldı")
        return True

    def stop(self):
        """İşlemeyi durdur."""
        self.processing_active = False
        self.log("Nesne tespiti durduruldu")

    def process_frame(self, frame):
        """Bir kareyi işle ve nesne tespiti yap."""
        if not self.processing_active or frame is None or self.model is None:
            return frame, None
            
        try:
            # Kareyi kopyala
            processed_frame = frame.copy()
            
            # Blob oluştur
            blob = cv2.dnn.blobFromImage(processed_frame, 1/255.0, (self.input_width, self.input_height), 
                                        [0,0,0], swapRB=True, crop=False)
            self.model.setInput(blob)
            
            # Çıkış katmanı isimlerini al
            output_layer_names = self.model.getUnconnectedOutLayersNames()
            
            # Çıktıları al
            outputs = self.model.forward(output_layer_names)
            
            # YOLOv8 için post-processing
            class_ids, confidences, boxes = self._postprocess_yolov8(processed_frame, outputs[0])
            
            detected_objects = []
            detected_classes_set = set()
            
            # NMS uygula (iç içe geçen kutuları filtrele)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
            
            # Tespit edilen her nesne için
            for idx in indices:
                # idx hem liste hem numpy array olabiliyor, bu yüzden düzelt
                i = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
                box = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                
                # Sınıf adını al
                class_name = self.classes[class_id] if class_id < len(self.classes) else f"Sınıf {class_id}"
                
                # Kutu bilgilerini al
                x, y, w, h = box
                
                # Kutuyu çiz
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Etiketi çiz
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(processed_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
                # Tespit sonucunu listeye ekle
                detected_objects.append({
                    'label': class_name,
                    'confidence': confidence,
                    'box': box
                })
                
                # Tespit edilen sınıfı kaydet
                detected_classes_set.add(class_name)
            
            # Yeni tespit edilen nesneleri yayınla
            current_time = time.time()
            if current_time - self.last_detection_time > self.min_detection_interval:
                # Yeni tespit edilen nesneleri bul
                new_detections = detected_classes_set - self.last_detected_objects
                
                # Yeni tespitler varsa yayınla
                for obj in detected_objects:
                    if obj['label'] in new_detections and obj['confidence'] > 0.6:  # Yüksek güven için ek filtre
                        self.publish_object_detected(obj)
                
                # Son tespitleri güncelle
                self.last_detected_objects = detected_classes_set
                self.last_detection_time = current_time
            
            return processed_frame, {'detections': detected_objects}
            
        except Exception as e:
            self.log(f"Kare işlenirken hata: {e}")
            import traceback
            self.log(traceback.format_exc())
            return frame, None
    
    def _postprocess_yolov8(self, frame, output):
        """YOLOv8 için post-processing."""
        frame_height, frame_width = frame.shape[:2]
        
        boxes = []
        class_ids = []
        confidences = []
        
        rows = output.shape[0]
        
        x_scale = frame_width / self.input_width
        y_scale = frame_height / self.input_height
        
        def get_scalar(val):
            # Eğer array ve boyutu 1 ise, ilk elemana eriş
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    return float(val.flat[0])
                else:
                    return float(val[0])
            return float(val)
        
        for i in range(rows):
            row = output[i]
            conf = get_scalar(row[4])
            
            if conf < self.conf_threshold:
                continue
                
            classes_scores = row[5:]
            class_id = int(np.argmax(classes_scores))
            class_score = get_scalar(classes_scores[class_id])
            
            if class_score < self.conf_threshold:
                continue
                
            cx = get_scalar(row[0])
            cy = get_scalar(row[1])
            w = get_scalar(row[2])
            h = get_scalar(row[3])
            
            x = int((cx - w/2) * x_scale)
            y = int((cy - h/2) * y_scale)
            width = int(w * x_scale)
            height = int(h * y_scale)
            
            box = [x, y, width, height]
            boxes.append(box)
            confidences.append(conf)
            class_ids.append(class_id)
            
        return class_ids, confidences, boxes

    def publish_object_detected(self, object_info):
        """Tespit edilen nesneyi pubsub ile yayınlar ve isteğe bağlı olarak robota gönderir."""
        if not object_info:
            return
            
        self.log(f"Nesne tespit edildi: {object_info['label']} (güven: {object_info['confidence']:.2f})")
        
        # PubSub ile nesneyi yayınla
        pub.sendMessage('object_detected', object_info=object_info)
        
        # Komut gönderici varsa robota gönder
        if self.command_sender and self.command_sender.connected:
            try:
                self.command_sender.send_command('object_event', {
                    'label': object_info['label'],
                    'confidence': float(object_info['confidence'])
                })
            except Exception as e:
                self.log(f"Nesne bilgisi gönderilirken hata: {e}")

    def log(self, message):
        """Loglama işlemlerini yapar."""
        pub.sendMessage('log', msg=f"[ObjectDetector] {message}")
