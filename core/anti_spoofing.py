import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class AntiSpoofing:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Eye landmarks based on MediaPipe Face Mesh
        # Right eye indices
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        # Left eye indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        
        self.EAR_THRESHOLD = 0.21  # Eye Aspect Ratio threshold for blink
        self.BLINK_CONSEC_FRAMES = 2 # Frames below threshold to count as blink
        
        self.blink_counter = 0
        self.total_blinks = 0
        
        self.is_real = False
        self.consecutive_real_frames = 0
        
    def _euclidean_distance(self, p1, p2):
        return math.dist(p1, p2)
        
    def _calculate_ear(self, face_landmarks, eye_indices, img_w, img_h):
        # Extract coordinates
        coords = []
        for idx in eye_indices:
            landmark = face_landmarks[idx]
            coords.append((int(landmark.x * img_w), int(landmark.y * img_h)))
            
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = self._euclidean_distance(coords[1], coords[5])
        B = self._euclidean_distance(coords[2], coords[4])
        
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = self._euclidean_distance(coords[0], coords[3])
        
        if C == 0:
            return 0.0
            
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def process_frame(self, frame):
        """
        Process the frame to detect liveness (blinks).
        Returns a boolean indicating if the face is "Real".
        """
        img_h, img_w, _ = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.face_landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            # No face detected
            self.is_real = False
            self.consecutive_real_frames = 0
            return False, frame
            
        face_landmarks = detection_result.face_landmarks[0]
        
        # Calculate EAR for both eyes
        left_ear = self._calculate_ear(face_landmarks, self.LEFT_EYE, img_w, img_h)
        right_ear = self._calculate_ear(face_landmarks, self.RIGHT_EYE, img_w, img_h)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Detect blink
        if avg_ear < self.EAR_THRESHOLD:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.BLINK_CONSEC_FRAMES:
                self.total_blinks += 1
                # If a blink is detected, we consider it a live real face
                self.is_real = True
                self.consecutive_real_frames = 30 # Keep "Real" status for next X frames to avoid flickering
            self.blink_counter = 0
            
        if self.consecutive_real_frames > 0:
            self.consecutive_real_frames -= 1
            if self.consecutive_real_frames == 0:
                self.is_real = False
                
            
        return self.is_real, frame
