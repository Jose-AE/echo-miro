import cv2
import mediapipe as mp
import numpy as np
import time
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list

# ---- Setup ----
device = "cpu"
model_name = get_model_list()[0]
fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
processing_start_time = time.time()


def get_emotion_from_frame(
    frame: cv2.typing.MatLike, resize_scale: float = 0.3, show_resized: bool = False, print_processing_time: bool = False
) -> tuple[str | None, tuple[int, int, int, int]]:
    """
    Detects emotion in the given frame using EmotiEffLib.
    Args:
        frame (cv2.typing.MatLike): The input frame from which to detect emotion.
        resize_scale (float): Scale factor for resizing the frame (0.0-1.0).
                              Default is 0.5 (50%). Lower values = faster processing but less accuracy.
        show_resized (bool): Whether to display the resized frame used for detection (DEBUG).

    Returns:
        tuple[str | None, tuple[int, int, int, int]]: A tuple containing the detected emotion label and the bounding box
        coordinates (x, y, width, height) of the detected face.
    """
    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (int(frame.shape[1] * resize_scale), int(frame.shape[0] * resize_scale)))

    processing_start_time = time.time()

    if show_resized:
        cv2.imshow("Resized Frame for Emotion Detection", resized_frame)

    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

    # If no faces are detected, return None for emotion and a default bounding box
    if len(faces) == 0:
        return (None, (0, 0, 0, 0))

    # Use the first detected face
    x, y, w, h = faces[0]

    # Extract the face region
    face_region = resized_frame[y : y + h, x : x + w]
    face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

    emotion, _ = fer.predict_emotions(face_region_rgb, logits=True)
    label = emotion[0]

    # Scale bounding box coordinates back to original frame size
    scale_factor = 1.0 / resize_scale
    x, y, w, h = int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor)

    if print_processing_time:
        print(f"Emotion detection took {time.time() - processing_start_time:.3f} seconds.")

    return (label, (x, y, w, h))
