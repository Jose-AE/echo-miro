import os
import random
from typing import Literal
import asyncio
import threading
import time
from opencv_controller import OpenCVController
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
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

        self.realtime_voice_client = RealtimeVoiceClient()
        self.opencv_controller = OpenCVController(
            show_debugs=debug_mode, fullscreen=not debug_mode, mask_face=not debug_mode, portrait_mode=not debug_mode
        )

        # Emotion collection state
        self.prevalent_emotion = None
        self.is_collecting_emotions: bool = False
        self.collection_start_time: float | None = None
        self.collection_duration: float = 5.0  # seconds to collect emotions

    def set_random_emotion_image(self, emotion: Emotion) -> str:
        # read all image names inside folder images/[emotion]

        image_folder = f"images/{emotion}"
        image_files = os.listdir(image_folder)
        rng_image_path = None

        if image_files:
            rng_image_path = os.path.join(image_folder, random.choice(image_files))

        self.opencv_controller.set_backround_image(rng_image_path)

    async def set_prevalent_emotion(self, seconds=5):
        """
        Collects emotions over the specified period and returns the most prevalent one.
        Ignores None values (when no face is detected).
        """

        dict_emotion_counts = {}
        start_time = time.time()
        end_time = start_time + seconds

        self.is_collecting_emotions = True
        self.collection_start_time = start_time

        print(f"Starting emotion collection for {seconds} seconds...")

        while time.time() < end_time:
            current_emotion = self.opencv_controller.get_emotion()
            # Only count valid emotions (ignore None when face is not detected)
            if current_emotion is not None:
                if current_emotion not in dict_emotion_counts:
                    dict_emotion_counts[current_emotion] = 0
                dict_emotion_counts[current_emotion] += 1

            await asyncio.sleep(0.1)  # Sample emotion 10 times per second

        self.is_collecting_emotions = False
        self.collection_start_time = None

        # Find the most prevalent emotion
        if not dict_emotion_counts:
            print("No emotions detected during collection period.")
            self.prevalent_emotion = None
            return

        self.prevalent_emotion = max(dict_emotion_counts, key=dict_emotion_counts.get)
        print(f"\n{'='*50}")
        print(f"Emotion collection complete!")
        print(f"Emotion counts over {seconds} seconds: {dict_emotion_counts}")
        print(f"Most prevalent emotion: {self.prevalent_emotion}")
        print(f"{'='*50}\n")

    async def __main(self):
        print("Waiting for realtime client to connect")
        await self.realtime_voice_client.init()

        while True:
            # Reset emotion state
            self.prevalent_emotion = None
            self.selected_image = None

            # Wait for user to face the camera
            if not self.opencv_controller.is_user_engaged():
                self.opencv_controller.set_backround_image(None)
                print("Waiting for user to face the camera...")
                await asyncio.sleep(1)
                continue

            # User is now facing the camera, collect emotions for 5 seconds and set prevalent emotion
            print("\n🎥 User detected! Collecting emotions for 5 seconds...")
            await self.set_prevalent_emotion(seconds=self.collection_duration)

            if self.prevalent_emotion is None:
                print("⚠️ Could not determine emotion. User may have looked away.")
                await asyncio.sleep(1)
                continue

            # Set random background image based on prevalent emotion
            self.set_random_emotion_image(self.prevalent_emotion)

            # Store the detected emotion and start interaction
            print(f"✅ Starting interaction with emotion: {self.prevalent_emotion}")
            await self.realtime_voice_client.listen_and_respond(
                emotion=self.prevalent_emotion, get_user_is_engaged=self.opencv_controller.is_user_engaged
            )

            print("\n💬 Interaction complete. Waiting for next user...\n")

    def run(self):
        # Start the async __main in a separate thread
        def run_async_main():
            asyncio.run(self.__main())

        t1 = threading.Thread(target=run_async_main, daemon=True)
        t1.start()

        # Run OpenCV loop in the main thread
        self.opencv_controller.run()
