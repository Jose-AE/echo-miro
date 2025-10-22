import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)


def replace_background(frame, background_image_path=None, show_face=False, bg_color=(255, 255, 255)):
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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = segmenter.process(rgb_frame)

    # Create mask and apply background (image or solid color)
    mask = results.segmentation_mask
    condition = np.stack((mask,) * 3, axis=-1) > 0.5  # True for person, False for bg

    if background_image_path:
        # Load and resize background image to match frame dimensions
        background_image = cv2.imread(background_image_path)
        if background_image is not None:
            bg_frame = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))
        else:
            print(f"Warning: Could not load background image from {background_image_path}")
            # Use solid color background as fallback
            bg_frame = np.zeros(frame.shape, dtype=np.uint8)
            bg_frame[:] = bg_color
    else:
        # Use solid color background
        bg_frame = np.zeros(frame.shape, dtype=np.uint8)
        bg_frame[:] = bg_color

    # Create black frame for person silhouette
    person_frame = np.zeros(frame.shape, dtype=np.uint8) if not show_face else frame

    # Combine person and background frames
    output_frame = np.where(condition, person_frame, bg_frame)

    return output_frame
