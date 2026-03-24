import cv2
from core.anti_spoofing import AntiSpoofing
from core.recognition import FaceRecognizer
from db_manager import log_event

class VideoPipeline:
    def __init__(self):
        self.anti_spoofing = AntiSpoofing()
        self.recognizer = FaceRecognizer()
        
        # State to prevent spamming logs
        self.last_logged_name = None
        self.last_logged_status = None
        self.frames_since_log = 0
        
        # Caching state for UI responsiveness
        self.frame_count = 0
        self.cached_name = "Unknown"
        self.cached_bbox = None
        
    def process_frame(self, frame):
        # 1. Anti-Spoofing (Fast enough to run every frame)
        is_real, _ = self.anti_spoofing.process_frame(frame)
        status_text = "Real" if is_real else "Spoof"
        
        # 2. Recognition (Expensive: Run every 5 frames)
        self.frame_count += 1
        if self.frame_count % 5 == 0 or self.cached_bbox is None:
            name, bbox = self.recognizer.recognize(frame)
            if bbox is not None:
                self.cached_bbox = bbox
                self.cached_name = name if name else "Unknown"
            else:
                self.cached_bbox = None
                self.cached_name = "Unknown"
            
        if self.cached_bbox is not None:
            x, y, w, h = self.cached_bbox
            # Draw bbox
            color = (0, 255, 0) if is_real else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Label
            label = f"{self.cached_name} - {status_text}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Log event if state changed or sufficient time passed
            self.frames_since_log += 1
            if (self.cached_name != self.last_logged_name or status_text != self.last_logged_status) or self.frames_since_log > 150:
                # Log if it's not a temporary unknown/spoof transition that oscillates
                log_event(self.cached_name, status_text)
                self.last_logged_name = self.cached_name
                self.last_logged_status = status_text
                self.frames_since_log = 0
                
        return frame
