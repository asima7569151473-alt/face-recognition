import cv2
import os
from core.hand_tracker import HandTracker

def main():
    if not os.path.exists('hand_landmarker.task'):
        print("Model file 'hand_landmarker.task' is missing! Please download it first.")
        return
        
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    
    print("Press ESC to exit the Hand Tracking window...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break
            
        # process_frame handles mirroring and annotating, returning RGB
        frame_rgb = tracker.process_frame(frame)
        
        # Convert back to BGR for native OpenCV presentation
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Hand Gesture Live", frame_bgr)
        
        if cv2.waitKey(1) & 0xFF == 27: # ESC key
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
