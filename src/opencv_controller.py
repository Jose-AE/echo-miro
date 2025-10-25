import cv2
import time
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
import mediapipe as mp
import numpy as np


class OpenCVController:

    def __init__(
        self,
        mask_face=True,
        show_debugs=False,
        camera_index=0,
        screen_resolution=(600, 1024),
        webcam_resolution=(1280, 720),
        fullscreen=True,
        portrait_mode=True,
    ):
        self.portrait_mode = portrait_mode
        self.show_debugs = show_debugs
        self.video_capture: cv2.VideoCapture | None = None
        self.screen_resolution = screen_resolution
        self.camera_index = camera_index
        self.emotion_capture_interval = 10  # Number of frames between emotion captures
        self.debug_color = (219, 73, 24)

        self.frame = None
        self.frame_count = 0
        self.frame_start_time = time.time()
        self.fps = 0

        # Init emotion and face rec models
        model_name = get_model_list()[0]
        self.emotion_recognizer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device="cpu")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # user engagement settings
        self.live_emotion = None
        self.emotion_bbox = (0, 0, 0, 0)  # (x, y, w, h)
        self.engagement_timeout = 5.0  # number of secs user has to face away to unengage
        self.user_is_engaged: bool = False
        self.last_time_user_engaged: float | None = None  # Timestamp when user last appeared in frame

        # Initialize MediaPipe for Selfie Segmentation
        self.selfie_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.background_image = None  # "images/default.png"
        self.mask_face = mask_face

        # Initialize webcam
        self.video_capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if camera_index > 0 else None)
        if not self.video_capture.isOpened():
            print("Error: Could not open webcam.")
            exit()

        # Set camera capture res
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_resolution[0])
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_resolution[1])

        if fullscreen:
            cv2.namedWindow("Emotion Detection (Press 'q' to exit)", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Emotion Detection (Press 'q' to exit)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def update_emotion(self, resize_scale: float = 0.3, show_resized: bool = False):
        """
        Detects emotion in the given frame using EmotiEffLib.
        Args:
            frame (cv2.typing.MatLike): The input frame from which to detect emotion.
            resize_scale (float): Scale factor for resizing the frame (0.0-1.0).
                                Default is 0.5 (50%). Lower values = faster processing but less accuracy.
            show_resized (bool): Whether to display the resized frame used for detection (DEBUG).
        """
        # Resize frame for faster processing
        resized_frame = cv2.resize(self.frame, (int(self.frame.shape[1] * resize_scale), int(self.frame.shape[0] * resize_scale)))

        if show_resized:
            cv2.imshow("Resized Frame for Emotion Detection", resized_frame)

        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

        # If no faces are detected, return None for emotion and a default bounding box
        if len(faces) == 0:
            self.live_emotion = None
            self.emotion_bbox = (0, 0, 0, 0)
            return

        # Use the first detected face
        x, y, w, h = faces[0]

        # Extract the face region
        face_region = resized_frame[y : y + h, x : x + w]
        face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

        emotion, _ = self.emotion_recognizer.predict_emotions(face_region_rgb, logits=True)
        label = emotion[0]

        # Scale bounding box coordinates back to original frame size
        scale_factor = 1.0 / resize_scale
        x, y, w, h = int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor)

        self.live_emotion = label
        self.emotion_bbox = (x, y, w, h)

    def update_frame_background(self):
        """
        Replaces the background of the given frame using MediaPipe Selfie Segmentation.
        Args:
            frame (cv2.typing.MatLike): The input frame from which to replace the background.
            background_image_path (str | None): Path to the background image file. If None, a solid color background is used.
            show_face (bool): Whether to show the person's face in the output frame.
            bg_color (tuple): BGR color tuple for solid color background if no image is provided.
        Returns:
            cv2.typing.MatLike: The frame with the background replaced.
        """

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmenter.process(rgb_frame)

        # Create mask and apply background (image or solid color)
        mask = results.segmentation_mask
        condition = np.stack((mask,) * 3, axis=-1) > 0.5  # True for person, False for bg

        if self.background_image:
            # Load and resize background image to match frame dimensions
            background_image = cv2.imread(self.background_image)
            if background_image is not None:
                bg_frame = cv2.rotate(background_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                bg_frame = cv2.resize(bg_frame, (self.frame.shape[1], (self.frame.shape[0])))
            else:
                print(f"Warning: Could not load background image from {self.background_image}")
                # Use solid color background as fallback
                bg_frame = np.zeros(self.frame.shape, dtype=np.uint8)
                bg_frame[:] = (255, 0, 0)
        else:
            # Use solid color background
            bg_frame = np.zeros(self.frame.shape, dtype=np.uint8)
            bg_frame[:] = (0, 0, 0)

        # Create black frame for person silhouette
        person_frame = np.zeros(self.frame.shape, dtype=np.uint8) if self.mask_face else self.frame

        # Combine person and background frames
        self.frame = np.where(condition, person_frame, bg_frame)

    def update_fps(self):
        """
        Call this function each frame to update fps
        """

        self.frame_count += 1
        elapsed_time = time.time() - self.frame_start_time
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time  # set fps
            self.frame_count = 0
            self.frame_start_time = time.time()

    def update_user_engagement(self):
        # Always update engagement state
        if self.live_emotion is not None:
            self.user_is_engaged = True
            self.last_time_user_engaged = time.time()
        elif self.last_time_user_engaged and (time.time() - self.last_time_user_engaged > self.engagement_timeout):
            self.user_is_engaged = False
            self.last_time_user_engaged = None

    def preprocess_frame(self):

        # Crop webcam frame to match screen res
        h, w = self.frame.shape[:2]
        target_w, target_h = self.screen_resolution  # (width, height) for final output
        center_x, center_y = w // 2, h // 2

        # Crop to the maximum available size while maintaining aspect ratio
        # For portrait mode: we want width=600, height=1024
        # Since webcam is 1280x720, we'll crop to fit the aspect ratio first
        aspect_ratio = target_w / target_h  # 600/1024 = 0.586

        # Determine crop dimensions to maintain target aspect ratio
        if w / h > aspect_ratio:
            # Width is too large, crop width
            crop_h = h
            crop_w = int(h * aspect_ratio)
        else:
            # Height is too large, crop height
            crop_w = w
            crop_h = int(w / aspect_ratio)

        start_y = max(center_y - crop_h // 2, 0)
        end_y = start_y + crop_h

        start_x = max(center_x - crop_w // 2, 0)
        end_x = start_x + crop_w

        self.frame = self.frame[start_y:end_y, start_x:end_x]

        # Resize to exact target resolution
        self.frame = cv2.resize(self.frame, (target_w, target_h))

        # Flip frame horizontally to create mirror effect
        self.frame = cv2.flip(self.frame, 1)

    def draw_debugs(self):
        # Unpack bbox for drawing
        x, y, w, h = self.emotion_bbox

        # Draw bounding box and label
        cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(self.frame, self.live_emotion if self.live_emotion else "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # Display FPS on the frame
        cv2.putText(self.frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.debug_color, 2)
        cv2.putText(
            self.frame,
            "User Engaged" if self.user_is_engaged else "User Not Engaged",
            (250, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            self.debug_color,
            2,
        )

    def set_backround_image(self, img):
        self.background_image = img

    def get_emotion(self):
        return self.live_emotion

    def is_user_engaged(self):
        return self.user_is_engaged

    def run(self):
        while True:
            self.update_fps()

            ret, self.frame = self.video_capture.read()
            if not ret:
                print("Error: Could not read frame.")
                continue

            self.preprocess_frame()

            # Detect emotion and check for engagement every few frames
            if self.frame_count % self.emotion_capture_interval == 0:
                self.update_emotion()
                self.update_user_engagement()

            self.update_frame_background()

            if self.show_debugs:
                self.draw_debugs()

            # Show the result
            cv2.imshow(
                "Emotion Detection (Press 'q' to exit)",
                cv2.rotate(self.frame, cv2.ROTATE_90_COUNTERCLOCKWISE) if self.portrait_mode else self.frame,
            )
            self.frame_count += 1

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cv = OpenCVController(show_debugs=True, fullscreen=False, camera_index=0, portrait_mode=False, mask_face=False)
    cv.run()
