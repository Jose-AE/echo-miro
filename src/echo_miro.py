from typing import Literal
import cv2
import asyncio
from calculate_fps import calculate_fps
from get_emotion_from_frame import get_emotion_from_frame
from replace_background import replace_background
import threading
import time
import multiprocessing as mp
from realtime_voice_client import RealtimeVoiceClient

Emotion = Literal[
    None,
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]


class EchoMiro:
    def __init__(self, camera_index=0, capture_resolution=(1024, 600), fullscreen=True):

        self.realtime_voice_client = RealtimeVoiceClient()

        # User
        self.engagement_timeout = 5.0  # seconds
        self.user_is_engaged: bool = False
        self.last_time_user_engaged: float | None = None  # Timestamp when user last appeared in frame

        # OpenCV video capture setup
        self.video_capture: cv2.VideoCapture | None = None
        self.frame_count = 0
        self.frame = None
        self.emotion_bbox = (0, 0, 0, 0)  # (x, y, w, h)
        self.capture_resolution = capture_resolution
        self.fullscreen = fullscreen
        self.camera_index = camera_index
        self.emotion_capture_interval = 10  # Number of frames between emotion captures

        # Emotion state
        self.live_emotion: Emotion = None
        self.detected_emotion: Emotion = None

        # Emotion collection state
        self.is_collecting_emotions: bool = False
        self.collection_start_time: float | None = None
        self.collection_duration: float = 5.0  # seconds to collect emotions

        # self.state: Literal["waiting_for_emotion", "interacting_with_user"] = "waiting_for_emotion"

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

    def select_random_emotion_image(emotion: Emotion) -> str:

        return ""

    def opencv_thread(self):
        while True:
            self.frame_count += 1
            ret, frame = self.video_capture.read()
            if not ret:
                print("Error: Could not read frame.")
                continue

            # Detect emotion and check for engagement every few frames
            if self.frame_count % self.emotion_capture_interval == 0:
                self.live_emotion, self.emotion_bbox = get_emotion_from_frame(frame)

                # Always update engagement state
                if self.live_emotion is not None:
                    self.user_is_engaged = True
                    self.last_time_user_engaged = time.time()
                elif self.last_time_user_engaged and (time.time() - self.last_time_user_engaged > self.engagement_timeout):
                    self.user_is_engaged = False
                    self.last_time_user_engaged = None

            frame = replace_background(frame, background_image_path=None)

            # Unpack bbox for drawing
            x, y, w, h = self.emotion_bbox

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, self.live_emotion if self.live_emotion else "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # Display FPS on the frame
            cv2.putText(frame, f"FPS: {calculate_fps():.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(
                frame,
                "User Engaged" if self.user_is_engaged else "User Not Engaged",
                (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                1,
            )
            # show detected emotion
            cv2.putText(
                frame,
                f"Detected Emotion: {self.detected_emotion if self.detected_emotion else 'Unknown'}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )

            # Show emotion collection status
            if self.is_collecting_emotions and self.collection_start_time:
                elapsed = time.time() - self.collection_start_time
                remaining = max(0, self.collection_duration - elapsed)
                cv2.putText(
                    frame,
                    f"Collecting emotions: {remaining:.1f}s",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )

            # Show the result

            cv2.imshow("Emotion Detection (Press 'q' to exit)", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()

    async def set_prevalent_emotion(self, seconds=5):
        """
        Collects emotions over the specified period and returns the most prevalent one.
        Ignores None values (when no face is detected).
        """
        self.detected_emotion = None  # Reset detected emotion
        dict_emotion_counts = {}
        start_time = time.time()
        end_time = start_time + seconds

        self.is_collecting_emotions = True
        self.collection_start_time = start_time

        print(f"Starting emotion collection for {seconds} seconds...")

        while time.time() < end_time:
            # Only count valid emotions (ignore None when face is not detected)
            if self.live_emotion is not None:
                if self.live_emotion not in dict_emotion_counts:
                    dict_emotion_counts[self.live_emotion] = 0
                dict_emotion_counts[self.live_emotion] += 1

            await asyncio.sleep(0.1)  # Sample emotion 10 times per second

        self.is_collecting_emotions = False
        self.collection_start_time = None

        # Find the most prevalent emotion
        if not dict_emotion_counts:
            print("No emotions detected during collection period.")
            self.detected_emotion = None

        prevalent_emotion = max(dict_emotion_counts, key=dict_emotion_counts.get)
        print(f"\n{'='*50}")
        print(f"Emotion collection complete!")
        print(f"Emotion counts over {seconds} seconds: {dict_emotion_counts}")
        print(f"Most prevalent emotion: {prevalent_emotion}")
        print(f"{'='*50}\n")

        self.detected_emotion = prevalent_emotion

    async def __main(self):
        await self.realtime_voice_client.init()

        while True:
            # Wait for user to face the camera
            if not self.user_is_engaged:
                print("Waiting for user to face the camera...")
                await asyncio.sleep(1)
                continue

            # User is now facing the camera, collect emotions for 5 seconds and set prevalent emotion
            print("\n🎥 User detected! Collecting emotions for 5 seconds...")
            await self.set_prevalent_emotion(seconds=self.collection_duration)

            if self.detected_emotion is None:
                print("⚠️ Could not determine emotion. User may have looked away.")
                await asyncio.sleep(1)
                continue

            # Store the detected emotion and start interaction
            print(f"✅ Starting interaction with emotion: {self.detected_emotion}")
            await self.realtime_voice_client.listen_and_respond(emotion=self.detected_emotion, get_user_is_engaged=lambda: self.user_is_engaged)

            # Reset after interaction
            self.detected_emotion = None
            print("\n💬 Interaction complete. Waiting for next user...\n")

    def run(self):
        t1 = threading.Thread(target=self.opencv_thread, daemon=True)
        t1.start()

        asyncio.run(self.__main())
