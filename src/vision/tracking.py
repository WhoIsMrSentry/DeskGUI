from pubsub import pub

class Tracking:
    TRACKING_THRESHOLD = 30
    TRACKING_MOVE_PERCENT = 10
    
    def __init__(self, command_sender=None):
        self.command_sender = command_sender
        self.active = False
        self.screen_dimensions = (640, 480)
        
    def set_active(self, active):
        """Takip özelliğini etkinleştir veya devre dışı bırak"""
        self.active = active
        
    def set_dimensions(self, dimensions):
        """Ekran boyutlarını ayarla"""
        self.screen_dimensions = dimensions
        
    def track_object(self, object_rect, is_priority=False):
        """Bir nesneyi takip et ve gerekli servo komutlarını gönder"""
        if not self.active or not self.command_sender or not object_rect:
            return False
            
        # Extract object coordinates
        x, y, w, h = object_rect
        
        # Calculate center points
        screen_w, screen_h = self.screen_dimensions
        screen_cx = screen_w / 2
        screen_cy = screen_h / 2
        
        obj_cx = x + (w / 2)
        obj_cy = y + (h / 2)
        
        # Adjust tracking sensitivity for priority persons
        threshold = self.TRACKING_THRESHOLD
        move_percent = self.TRACKING_MOVE_PERCENT
        
        if is_priority:
            # More sensitive tracking for priority persons
            threshold = self.TRACKING_THRESHOLD * 0.7
            move_percent = self.TRACKING_MOVE_PERCENT * 0.8
            pub.sendMessage('log', msg="Tracking priority person with increased sensitivity")
        
        # Calculate movement needed for pan (horizontal)
        if abs(screen_cx - obj_cx) > threshold:
            x_move = round((screen_cx - obj_cx) / move_percent)
            self.command_sender.send_command('servo_move', {
                'identifier': 'pan',
                'percentage': x_move
            })
            
        # Calculate movement needed for tilt (vertical)
        if abs(screen_cy - obj_cy) > threshold:
            y_move = round((obj_cy - screen_cy) / move_percent)
            self.command_sender.send_command('servo_move', {
                'identifier': 'tilt',
                'percentage': -y_move
            })
            
        return True