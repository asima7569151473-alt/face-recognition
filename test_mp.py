import mediapipe as mp
try:
    print(f"mp.solutions: {mp.solutions}")
except Exception as e:
    print(f"Error: {e}")

import mediapipe.solutions as solutions
print(f"solutions: {solutions}")
