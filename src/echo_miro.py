import cv2

from calculate_fps import calculate_fps
from get_emotion_from_frame import get_emotion_from_frame
from replace_background import replace_background


class EchoMiro:
    def __init__(self, camera_index=0, capture_resolution=(1024, 600), fullscreen=True):
        self.video_capture: cv2.VideoCapture | None = None
        self.frame_count = 0
        self.emotion_bbox = (0, 0, 0, 0)  # (x, y, w, h)
        self.capture_resolution = capture_resolution
        self.fullscreen = fullscreen
        self.camera_index = camera_index
        self.detected_emotion = "Detecting..."

        self.__init_opencv()

    def __init_opencv(self):
        self.video_capture = cv2.VideoCapture(self.camera_index)
        if not self.video_capture.isOpened():
            print("Error: Could not open webcam.")
            exit()

        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_resolution[0])
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_resolution[1])

        if self.fullscreen:
            cv2.namedWindow("Emotion Detection (Press 'q' to exit)", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Emotion Detection (Press 'q' to exit)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def run(self):
        # Start emotion detection in background
        # self.start_emotion_detection_thread()

        # Frame counter for emotion detection

        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                print("Error: Could not read frame.")
                continue

            # Detect emotion only every 10 frames
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                self.detected_emotion, self.emotion_bbox = get_emotion_from_frame(frame)

            frame = replace_background(frame, background_image_path=None)

            # Unpack bbox for drawing
            x, y, w, h = self.emotion_bbox

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, self.detected_emotion if self.detected_emotion else "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            # Display FPS on the frame
            cv2.putText(frame, f"FPS: {calculate_fps():.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Show the result

            cv2.imshow("Emotion Detection (Press 'q' to exit)", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()
