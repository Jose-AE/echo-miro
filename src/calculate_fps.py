import time


fps = 0
frame_count = 0
start_time = time.time()


def calculate_fps():
    """
    Call this function each frame to get frames per second (FPS)

    Returns:
        float: The calculated FPS.

    """
    global frame_count, start_time, fps

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    return fps
