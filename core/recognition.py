import cv2
import numpy as np
from deepface import DeepFace
from db_manager import get_all_users

class FaceRecognizer:
    def __init__(self, model_name="Facenet", detector_backend="opencv"):
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.similarity_threshold = 0.4 # Cosine similarity threshold for Facenet
        self._load_users()
        
    def _load_users(self):
        """Loads all registered users and their embeddings into memory."""
        users_list = get_all_users()
        self.known_embeddings = []
        self.known_names = []
        
        for user in users_list:
            self.known_names.append(user["name"])
            self.known_embeddings.append(np.array(user["embedding"]))
            
    def reload_users(self):
        """Used to reload users if a new user was registered."""
        self._load_users()

    def get_embedding(self, frame, enforce_detection=True, enforce_single_face=False):
        """
        Extracts face embedding from the frame.
        Returns the embedding list, and bounding box coordinates if a face is found.
        """
        try:
            # We use DeepFace.represent to get the embeddings directly
            results = DeepFace.represent(
                img_path=frame, 
                model_name=self.model_name, 
                detector_backend=self.detector_backend, 
                enforce_detection=enforce_detection,
                align=True
            )
            
            if len(results) == 0:
                return None, None
                
            if enforce_single_face and len(results) > 1:
                return "multiple", None
            
            # Return the largest face for simplicity
            face_data = results[0]
            embedding = face_data["embedding"]
            facial_area = face_data["facial_area"] 
            # facial_area is dict: {'x': x, 'y': y, 'w': w, 'h': h}
            bbox = (facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h'])
            return embedding, bbox
        except ValueError:
            # No face detected
            return None, None
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None, None

    def recognize(self, frame):
        """
        Recognize faces in the frame.
        Returns the name of the recognized person, and face bounding box.
        """
        embedding, bbox = self.get_embedding(frame, enforce_detection=True)
        
        if embedding is None:
            return None, None
            
        embedding_arr = np.array(embedding)
        name = "Unknown"
        min_distance = float('inf')
        best_match_idx = -1
        
        # Compare with known embeddings
        for idx, known_emb in enumerate(self.known_embeddings):
            # Calculate Cosine Distance
            # Distance = 1 - Cosine Similarity
            dot_product = np.dot(embedding_arr, known_emb)
            norm_a = np.linalg.norm(embedding_arr)
            norm_b = np.linalg.norm(known_emb)
            
            cos_sim = dot_product / (norm_a * norm_b)
            distance = 1 - cos_sim
            
            if distance < min_distance:
                min_distance = distance
                best_match_idx = idx
                
        if best_match_idx != -1 and min_distance < self.similarity_threshold:
            name = self.known_names[best_match_idx]
            
        return name, bbox
