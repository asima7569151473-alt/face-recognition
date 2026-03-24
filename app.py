import streamlit as st
import cv2
import pandas as pd
import numpy as np
import time
from PIL import Image
from core.hand_tracker import HandTracker
from core.recognition import FaceRecognizer
from db_manager import register_user, init_db

# Initialize database on startup
init_db()

st.set_page_config(page_title="FaceGuard AI Pro", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS INJECTION ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    :root {
        --bg-slate: #0b0f19;
        --bg-panel: rgba(15, 23, 42, 0.6);
        --accent-primary: #0ea5e9;
        --accent-secondary: #8b5cf6;
        --accent-tertiary: #10b981;
        --glass-border: rgba(255, 255, 255, 0.08);
        --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
        --text-primary: #f8fafc;
        --text-muted: #94a3b8;
    }

    /* Global styles */
    .stApp {
        background-color: var(--bg-slate);
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(139, 92, 246, 0.12) 0%, transparent 50%),
            radial-gradient(circle at 85% 30%, rgba(14, 165, 233, 0.12) 0%, transparent 50%);
        background-attachment: fixed;
        color: var(--text-primary);
        font-family: 'Outfit', sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(11, 15, 25, 0.7) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid var(--glass-border);
    }
    
    section[data-testid="stSidebar"] .st-emotion-cache-1lv8su8 {
        background-color: transparent;
    }

    /* Sidebar links */
    .stRadio p {
        font-family: 'Outfit';
        font-size: 1.05rem;
        font-weight: 500;
        color: var(--text-muted);
        transition: color 0.3s;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: var(--bg-panel);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 30px;
        box-shadow: var(--glass-shadow);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease;
    }
    .glass-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: radial-gradient(800px circle at var(--mouse-x, 50%) var(--mouse-y, -20%), rgba(255,255,255,0.04), transparent 40%);
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.5s;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5);
    }
    .glass-card:hover::before {
        opacity: 1;
    }

    /* Premium Typography */
    .neon-text {
        background: linear-gradient(135deg, #38bdf8, #818cf8, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Scanning Animation */
    @keyframes scan {
        0% { top: 0%; }
        50% { top: 100%; }
        100% { top: 0%; }
    }
    .scan-line {
        position: absolute;
        width: 100%;
        height: 4px;
        background: linear-gradient(to bottom, transparent, var(--accent-primary), transparent);
        box-shadow: 0 0 15px var(--accent-primary);
        z-index: 10;
        animation: scan 3s linear infinite;
        pointer-events: none;
    }

    /* Status Indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-family: 'Space Grotesk', sans-serif;
        position: relative;
        overflow: hidden;
    }
    .status-active {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    }
    .status-active::before {
        content: "●";
        margin-right: 8px;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; text-shadow: 0 0 10px #34d399; }
        50% { opacity: 0.4; text-shadow: none; }
        100% { opacity: 1; text-shadow: 0 0 10px #34d399; }
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, var(--accent-secondary), var(--accent-primary)) !important;
        border: none !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: 0.5px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.25) !important;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    .stButton>button::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: linear-gradient(45deg, var(--accent-primary), var(--accent-secondary));
        z-index: -1;
        transition: opacity 0.4s ease;
        opacity: 0;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 35px rgba(14, 165, 233, 0.4) !important;
    }
    .stButton>button:hover::before {
        opacity: 1;
    }

    /* Streamlit Native Inputs restyling */
    .stTextInput input, .stFileUploader > div:first-child {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2) !important;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- TOP HEADER ---
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 40px; padding: 20px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
        <div>
            <h1 class="neon-text" style="font-size: 2.5rem; margin: 0; letter-spacing: -1px;">FaceGuard AI Pro</h1>
            <p style="font-size: 0.95rem; color: #94a3b8; margin: 8px 0 0 0; font-weight: 500; letter-spacing: 2px;">PREMIUM NEURAL SURVEILLANCE SUITE</p>
        </div>
        <div style="display: flex; gap: 20px; align-items: center;">
            <div class="status-badge status-active">System Online</div>
            <div style="color: #94a3b8; font-family: 'Space Grotesk'; font-size: 0.85rem; padding: 8px 16px; border-radius: 20px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                <span style="color: #0ea5e9; font-weight: 700;">DFS: 32ms</span> &nbsp;|&nbsp; <span style="color: #8b5cf6; font-weight: 700;">FPS: 24</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home Dashboard"

def nav_to(page_name):
    st.session_state.nav_page = page_name

page = st.sidebar.radio("Navigation Console", ["Home Dashboard", "Face Recognition Live", "Hand Gesture Live", "Register User", "Group Photo Match"], key="nav_page")

if page == "Home Dashboard":
    st.markdown('''
<style>
/* Premium Grid Background Override */
.stApp {
    background-color: #0c0a1a !important;
    background-image: 
        linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px) !important;
    background-size: 40px 40px !important;
    background-position: center top !important;
}

/* Pill Badge */
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: rgba(168, 85, 247, 0.08);
    border: 1px solid rgba(168, 85, 247, 0.25);
    padding: 8px 20px;
    border-radius: 50px;
    color: #c084fc;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    font-family: 'Inter', sans-serif;
    margin: 0 auto;
}
.hero-pill svg {
    width: 16px; height: 16px;
    fill: currentColor;
}

/* Hero Gradient Text */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 5.5rem;
    font-weight: 800;
    line-height: 1.05;
    text-align: center;
    letter-spacing: -2px;
    margin: 25px 0 20px 0;
    background: linear-gradient(135deg, #5eead4 0%, #a855f7 40%, #c084fc 70%, #93c5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.15rem;
    text-align: center;
    max-width: 650px;
    margin: 0 auto 50px auto;
    line-height: 1.6;
    font-family: 'Outfit', sans-serif;
}

/* Action Buttons Wrapper */
.hero-buttons {
    display: flex;
    gap: 20px;
    justify-content: center;
    margin-bottom: 60px;
}
.btn-primary-hero {
    background: linear-gradient(90deg, #8b5cf6, #3b82f6);
    color: white !important;
    padding: 16px 36px;
    border-radius: 14px;
    font-weight: 600;
    font-family: 'Outfit', sans-serif;
    text-decoration: none !important;
    box-shadow: 0 4px 25px rgba(139, 92, 246, 0.4);
    transition: transform 0.2s, box-shadow 0.2s;
}
.btn-primary-hero:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.6);
}
.btn-secondary-hero {
    background: rgba(255, 255, 255, 0.03);
    color: #e2e8f0 !important;
    padding: 16px 36px;
    border-radius: 14px;
    font-weight: 600;
    font-family: 'Outfit', sans-serif;
    text-decoration: none !important;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: background 0.2s;
}
.btn-secondary-hero:hover {
    background: rgba(255, 255, 255, 0.08);
}
</style>

<div style="display: flex; flex-direction: column; align-items: center; padding-top: 60px;">
<div class="hero-pill"><svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"/></svg> Powered by DeepFace & MediaPipe</div>
<h1 class="hero-title">Real-Time Face<br>Recognition &<br>Detection</h1>
<p class="hero-subtitle">A professional neural surveillance platform that analyzes live feeds using advanced deep-learning models and multi-layer heuristics to provide real-time facial identification and kinetic tracking.</p>
</div>
''', unsafe_allow_html=True)

    st.markdown("""
        <style>
        .nav-buttons-container {
            margin-top: 20px;
            margin-bottom: 60px;
            padding: 0 10%;
        }
        div[data-testid="column"] {
            display: flex;
            justify-content: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-buttons-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.button("Face Recognition Live", use_container_width=True, on_click=nav_to, args=("Face Recognition Live",))
    with col2:
        st.button("Hand Gesture Live", use_container_width=True, on_click=nav_to, args=("Hand Gesture Live",))
    with col3:
        st.button("Register User", use_container_width=True, on_click=nav_to, args=("Register User",))
    with col4:
        st.button("Group Photo Match", use_container_width=True, on_click=nav_to, args=("Group Photo Match",))
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Face Recognition Live":
    st.button("← Back to Home Dashboard", on_click=nav_to, args=("Home Dashboard",))
    st.markdown('<h2 class="neon-text module-title">Biometric Identification Feed</h2>', unsafe_allow_html=True)
    st.write("Real-time neural processing for multi-face detection and identification.")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    run_video = st.checkbox("Toggle Neural Uplink (Webcam)", key="face_cam")
    
    # Placeholder for the video feed
    FRAME_WINDOW = st.image([])
    
    if run_video:
        st.markdown('<div style="position: relative; width: 100%;"><div class="scan-line"></div>', unsafe_allow_html=True)
        
        cap = cv2.VideoCapture(0)
        from core.pipeline import VideoPipeline
        pipeline = VideoPipeline()
        
        prev_time = time.time()
        while st.session_state.face_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Uplink Failure: Signal Lost.")
                break
            
            frame = cv2.flip(frame, 1)
            frame = pipeline.process_frame(frame)
            
            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Show output
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, caption=f"Neural Engine | FPS: {int(fps)}")
                
        cap.release()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("System Standby. Activate uplink to begin scanning.")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Hand Gesture Live":
    st.button("← Back to Home Dashboard", on_click=nav_to, args=("Home Dashboard",))
    st.markdown('<h2 class="neon-text module-title">Kinetic Interface Control</h2>', unsafe_allow_html=True)
    st.write("Neural tracking of hand landmarks and skeletal gesture mapping.")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    run_video = st.checkbox("Toggle Kinetic Sensor", key="hand_cam")
    FRAME_WINDOW = st.image([])
    
    if run_video:
        cap = cv2.VideoCapture(0)
        tracker = HandTracker()
        
        prev_time = time.time()
        while st.session_state.hand_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Kinetic Sensor Failure: Connection Timed Out.")
                break
                
            frame_rgb = tracker.process_frame(frame)
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            FRAME_WINDOW.image(frame_rgb, caption=f"Hand Tracking Engine | FPS: {int(fps)}")
            
        cap.release()
    else:
        st.write("Kinetic sensor offline. Awaiting activation.")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Register User":
    st.button("← Back to Home Dashboard", on_click=nav_to, args=("Home Dashboard",))
    st.markdown('<h2 class="neon-text module-title">Biometric Database Enrollment</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--neon-blue); font-size: 1rem;">Subject Metadata</h3>', unsafe_allow_html=True)
        name = st.text_input("Designated Identity Name:")
        upload_method = st.radio("Acquisition Protocol", ["Webcam Capture", "Archived File Upload"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--neon-blue); font-size: 1rem;">Biometric Preview</h3>', unsafe_allow_html=True)
        FRAME_WINDOW = st.image([])
        st.markdown('</div>', unsafe_allow_html=True)

    capture_btn = False
    uploaded_file = None
    
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    if upload_method == "Webcam Capture":
        capture_btn = st.button("EXECUTE ENROLLMENT")
    else:
        uploaded_file = st.file_uploader("Select Neural Template", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            capture_btn = st.button("SYNC TEMPLATE")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if capture_btn and name:
        frame = None
        if upload_method == "Webcam Capture":
            cap = cv2.VideoCapture(0)
            st.info("Capturing from webcam...")
            ret, cap_frame = cap.read()
            cap.release()
            if ret:
                frame = cv2.flip(cap_frame, 1)
        else:
            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert('RGB')
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            
            with st.spinner("Extracting facial embeddings (only 1 face allowed)..."):
                recognizer = FaceRecognizer()
                embedding, bbox = recognizer.get_embedding(frame, enforce_detection=True, enforce_single_face=True)
                
                if embedding == "multiple":
                    st.error("Multiple faces detected! Please ensure only ONE clear face is in the photo.")
                elif embedding is not None:
                    # Draw a box on the captured image to show the user what was detected
                    x, y, w, h = bbox
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    FRAME_WINDOW.image(frame_rgb, caption="Face Registered!")
                    
                    success = register_user(name, embedding)
                    if success:
                        st.success(f"User '{name}' registered successfully!")
                        # In the real system, FaceRecognizer caches embeddings. Since it's a Streamlit script, 
                        # restarting the page initializes a fresh instance anyway, but this is good practice.
                        recognizer.reload_users()
                    else:
                        st.error("Registration failed. A user with this name may already exist.")
                else:
                    st.error("No face detected. Please try again in good lighting.")
        else:
            st.error("Failed to capture image.")

elif page == "Group Photo Match":
    st.button("← Back to Home Dashboard", on_click=nav_to, args=("Home Dashboard",))
    st.markdown('<h2 class="neon-text module-title">Multi-Subject Target Identification</h2>', unsafe_allow_html=True)
    st.write("Scanning high-density environments for targeted facial signatures.")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        ref_file = st.file_uploader("Neural Signature Source (Solo)", type=["jpg", "png", "jpeg"])
    with col2:
        group_file = st.file_uploader("Target Environment (Group)", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if ref_file and group_file:
        if st.button("Find Person", type="primary"):
            st.info("Analyzing photos...")
            
            # Convert uploaded files to numpy arrays for OpenCV
            ref_image = Image.open(ref_file).convert('RGB')
            ref_img_np = np.array(ref_image)
            ref_img_bgr = cv2.cvtColor(ref_img_np, cv2.COLOR_RGB2BGR)
            
            group_image = Image.open(group_file).convert('RGB')
            group_img_np = np.array(group_image)
            group_img_bgr = cv2.cvtColor(group_img_np, cv2.COLOR_RGB2BGR)
            
            recognizer = FaceRecognizer()
            ref_emb, _ = recognizer.get_embedding(ref_img_bgr, enforce_detection=True)
            
            if ref_emb is None:
                st.error("No clear face found in the Reference Photo!")
            else:
                from deepface import DeepFace
                from ultralytics import YOLO
                try:
                    # 1. YOLO for Total Person Count
                    yolo_model = YOLO("yolov8n.pt")  # lightweight model
                    yolo_results = yolo_model(group_img_np, conf=0.5)  # Increased confidence to 0.5
                    person_count = 0
                    
                    for r in yolo_results:
                        for box in r.boxes:
                            if int(box.cls[0]) == 0:  # 0 is 'person'
                                person_count += 1
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                # Draw faint red/pink box for all people
                                cv2.rectangle(group_img_np, (x1, y1), (x2, y2), (255, 0, 255), 2)
                                
                    st.write(f"### 👥 Total Persons Detected: {person_count}")

                    # Extract embeddings for all faces in group photo
                    # Setting enforce_detection to False so it doesn't crash if no faces are found
                    group_results = DeepFace.represent(
                        img_path=group_img_bgr, 
                        model_name=recognizer.model_name, 
                        detector_backend=recognizer.detector_backend, 
                        enforce_detection=False,
                        align=True
                    )
                    
                    found_match = False
                    for face_data in group_results:
                        embedding = face_data["embedding"]
                        facial_area = face_data["facial_area"]
                        
                        # Cosine similarity
                        emb_arr1 = np.array(ref_emb)
                        emb_arr2 = np.array(embedding)
                        dot = np.dot(emb_arr1, emb_arr2)
                        norm_a = np.linalg.norm(emb_arr1)
                        norm_b = np.linalg.norm(emb_arr2)
                        
                        if norm_a == 0 or norm_b == 0: continue
                        
                        cos_sim = dot / (norm_a * norm_b)
                        distance = 1 - cos_sim
                        
                        if distance < recognizer.similarity_threshold:
                            # Match found! Draw bounding box!
                            found_match = True
                            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                            cv2.rectangle(group_img_np, (x, y), (x+w, y+h), (0, 255, 0), 4)
                            cv2.putText(group_img_np, "MATCH", (x, max(10, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                            
                    if found_match:
                        st.markdown('<div class="status-badge status-active" style="margin-bottom: 10px;">✅ TARGET IDENTIFIED</div>', unsafe_allow_html=True)
                        st.success("Positive neural match confirmed within the target environment.")
                        st.image(group_img_np, caption="Neural Mapping Result", use_container_width=True)
                    else:
                        st.markdown('<div class="status-badge" style="background: rgba(255, 69, 0, 0.1); border: 1px solid #ff4500; color: #ff4500; margin-bottom: 10px;">⚠️ NEGATIVE MATCH</div>', unsafe_allow_html=True)
                        st.warning("No matching neural signatures detected in the source image.")
                        st.image(group_img_np, caption="Source Environment Analysis", use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error processing group photo: {str(e)}")

