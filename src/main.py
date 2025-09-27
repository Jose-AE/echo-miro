import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0
emotion_text = "Detecting..."

# Load background image
bg_image_path = "bg.png"  # Change this to your image path
try:
    bg_image = cv2.imread(bg_image_path)
    if bg_image is None:
        print(f"Warning: Could not load background image from {bg_image_path}")
        print("Using solid color background instead")
        use_image_bg = False
        bg_color = (0, 128, 255)  # Fallback solid color
    else:
        use_image_bg = True
        print(f"Loaded background image: {bg_image_path}")
except Exception as e:
    print(f"Error loading background image: {e}")
    use_image_bg = False
    bg_color = (0, 128, 255)  # Fallback solid color


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (1280, 720))

    # Only analyze emotion every 10th frame
    if frame_count % 10 == 0:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion_text = result[0]['dominant_emotion']
        except:
            pass

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = segmenter.process(rgb_frame)

    # Create mask and apply background (image or solid color)
    mask = results.segmentation_mask
    condition = np.stack((mask,) * 3, axis=-1) > 0.5  # True for person, False for bg
    
    if use_image_bg:
        # Resize background image to match frame dimensions
        bg_frame = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
    else:
        # Use solid color background
        bg_frame = np.zeros(frame.shape, dtype=np.uint8)
        bg_frame[:] = bg_color
        
    output_frame = np.where(condition, frame, bg_frame)

    # Overlay emotion/temperature
    cv2.putText(output_frame, f'Emotion: {emotion_text}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 

    # Display
    cv2.namedWindow("Virtual Background", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Virtual Background", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Virtual Background", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
