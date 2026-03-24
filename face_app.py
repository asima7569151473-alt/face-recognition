import cv2
from core.recognition import FaceRecognizer

def main():
    cap = cv2.VideoCapture(0)
    recognizer = FaceRecognizer()
    
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 5
    last_name = None
    last_bbox = None
    
    print("Press ESC to exit the Face Recognition window...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break
            
        frame = cv2.flip(frame, 1) # Mirror view
        
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            name, bbox = recognizer.recognize(frame)
            if name is not None:
                last_name = name
                last_bbox = bbox
            else:
                last_name = None
                last_bbox = None
                
        if last_bbox is not None:
            x, y, w, h = last_bbox
            color = (0, 255, 0) if last_name != "Unknown" else (0, 0, 255) # BGR
            
            # Face Bounding Box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Overlay Details
            status = "Recognized" if last_name != "Unknown" else "Unknown"
            display_text = f"{last_name} ({status})"
            cv2.putText(frame, display_text, (x, max(10, y-10)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
                        
        cv2.imshow("Face Recognition Live", frame)
        
        if cv2.waitKey(1) & 0xFF == 27: # ESC key
            break
            
        frame_count += 1
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
