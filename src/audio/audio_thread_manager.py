import threading
from concurrent.futures import ThreadPoolExecutor

class AudioThreadManager:
    """TTS ve ses işlemleri için thread yöneticisi - daha iyi kontrol için"""
    def __init__(self, max_workers=2):
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks = {}
        self.lock = threading.Lock()
        
    def submit_task(self, task_id, task_fn, *args, **kwargs):
        """İşlem gönder ve takip et"""
        with self.lock:
            # Eğer aynı ID'li aktif bir görev varsa iptal et
            if task_id in self.active_tasks:
                self.active_tasks[task_id].cancel()
            
            # Yeni görevi gönder
            future = self.pool.submit(task_fn, *args, **kwargs)
            self.active_tasks[task_id] = future
            return future
            
    def cancel_task(self, task_id):
        """Görevi iptal et"""
        with self.lock:
            if task_id in self.active_tasks:
                self.active_tasks[task_id].cancel()
                del self.active_tasks[task_id]
                return True
        return False
        
    def shutdown(self):
        """Tüm aktif görevleri iptal et ve havuzu kapat"""
        with self.lock:
            for task_id in list(self.active_tasks.keys()):
                self.active_tasks[task_id].cancel()
            self.active_tasks.clear()
        self.pool.shutdown(wait=False)