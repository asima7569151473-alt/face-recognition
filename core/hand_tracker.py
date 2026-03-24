import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, max_hands=1, model_path='hand_landmarker.task'):
        """
        Initialize the MediaPipe Hand Landmarker using the modern Tasks API.
        This handles the case where older 'solutions' module is deprecated.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Manually define hand connections for drawing
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), # thumb joints
            (0, 5), (5, 6), (6, 7), (7, 8), # index joints
            (5, 9), (9, 10), (10, 11), (11, 12), # middle joints
            (9, 13), (13, 14), (14, 15), (15, 16), # ring joints
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # pinky & palm joints
        ]

    def count_fingers(self, hand_landmarks, handedness):
        """
        Count total number of fingers shown (0-5) using a mathematical distance-based 
        approach that works regardless of the camera being flipped, rotated, or angled.
        """
        count = 0
        
        def dist(idx1, idx2):
            # Calculate 2D Euclidean distance between two landmarks
            return np.hypot(hand_landmarks[idx1].x - hand_landmarks[idx2].x,
                            hand_landmarks[idx1].y - hand_landmarks[idx2].y)

        # Fingers: Index, Middle, Ring, Pinky
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        # When a finger is extended, its Tip is further from the Wrist (0) than its PIP joint.
        # When curled into a fist, the Tip tucks into the palm and becomes closer to the Wrist than the protruding PIP joint.
        for tip, pip in zip(tips, pips):
            if dist(tip, 0) > dist(pip, 0):
                count += 1
                
        # Thumb Logic: 
        # When extended, the Thumb Tip (4) is further from the Pinky Base (17) than the Thumb IP joint (3) is.
        # When folded across the palm (like in a 3-finger or 4-finger gesture), it physically moves closer to the Pinky Base.
        if dist(4, 17) > dist(3, 17):
            count += 1
                
        return count

    def process_frame(self, frame):
        """
        Takes a BGR frame from OpenCV, runs the complete pipeline,
        and returns an RGB frame for Streamlit with drawings/counts applied.
        """
        # Create mirror view
        frame = cv2.flip(frame, 1)
        
        # OpenCV uses BGR natively, Streamlit and MediaPipe Tasks use RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Format the image for the newer Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Perform detection
        results = self.detector.detect(mp_image)
        
        count = 0
        h, w, _ = frame_rgb.shape
        
        # If any hands are found
        if results.hand_landmarks and results.handedness:
            landmarks = results.hand_landmarks[0]
            hand_info = results.handedness[0]
            
            # Draw connections (lines)
            for connection in self.connections:
                pt1_idx, pt2_idx = connection
                pt1 = (int(landmarks[pt1_idx].x * w), int(landmarks[pt1_idx].y * h))
                pt2 = (int(landmarks[pt2_idx].x * w), int(landmarks[pt2_idx].y * h))
                cv2.line(frame_rgb, pt1, pt2, (255, 255, 255), 2)
                
            # Draw dots over the joints
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame_rgb, (cx, cy), 5, (0, 0, 255), -1)
                
            # Execute counting logic
            count = self.count_fingers(landmarks, hand_info)
            
        # Draw the resulting text on the frame
        cv2.putText(frame_rgb, f"Fingers: {count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        return frame_rgb

def main():
    import os
    if not os.path.exists('hand_landmarker.task'):
        print("The model file 'hand_landmarker.task' is missing!")
        return
        
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    
    print("Press ESC to exit the tracking window...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = tracker.process_frame(frame)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Hand Tracking & Finger Counting", frame_bgr)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
