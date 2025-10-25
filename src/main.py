import cv2
import realtime_voice_client
from calculate_fps import calculate_fps
from echo_miro import EchoMiro

# from get_emotion_from_frame import get_emotion_from_frame
# from replace_background import replace_back
if __name__ == "__main__":
    echo_miro = EchoMiro(camera_index=0, fullscreen=True)
    echo_miro.run()
