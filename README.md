# Echo Miro - Hand Tracking Mirror

A real-time hand tracking application that creates a mirror effect with hand skeleton overlay using MediaPipe and OpenCV.

## Features

- Real-time hand detection and tracking
- Hand skeleton visualization with landmark points
- Mirror effect for natural interaction
- Support for up to 2 hands simultaneously
- Cross-platform support (Windows, Linux, macOS)

## Prerequisites

### For Local Installation
- Python 3.8 or higher
- Webcam/Camera device
- Git

### For Docker Installation
- Docker Engine 20.10+
- Docker Compose 2.0+
- Webcam/Camera device
- **Linux/macOS**: X11 server for GUI display
- **Windows**: X11 server (like VcXsrv or Xming) for GUI display

## Installation and Usage

### Option 1: Local Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd echo-miro
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv py_modules
   py_modules\Scripts\activate

   # Linux/macOS
   python3 -m venv py_modules
   source py_modules/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python src/main.py
   ```


## Controls

- **Q key**: Quit the application
- **Camera**: Make sure your camera is not being used by other applications

## Troubleshooting

### Common Issues

**Camera not detected:**
- Ensure no other application is using the camera
- Check camera permissions
- Try different camera indices (modify `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`, etc.)

**Docker GUI not showing (Linux/macOS):**
```bash
# Allow X11 connections
xhost +local:docker

# Check DISPLAY variable
echo $DISPLAY
```

**Docker GUI not showing (Windows):**
- Ensure X11 server (VcXsrv/Xming) is running
- Verify "Disable access control" is checked in X11 server settings
- Check firewall settings for X11 server

**Permission denied for camera device:**
```bash
# Linux: Add user to video group
sudo usermod -a -G video $USER
# Then logout and login again

# Or run with privileged mode (less secure)
docker run --privileged ...
```

**ModuleNotFoundError:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+)

### Performance Tips

- Ensure good lighting for better hand detection
- Keep hands within camera frame
- Close unnecessary applications to free up camera resources
- For better performance, consider reducing detection confidence in `src/main.py`

## Development

### Project Structure
```
echo-miro/
├── src/
│   └── main.py          # Main application file
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker image configuration
├── docker-compose.yml  # Docker Compose configuration
├── .dockerignore       # Docker ignore file
└── README.md           # This file
```

### Customization

You can modify the hand tracking parameters in `src/main.py`:
- `max_num_hands`: Maximum number of hands to detect (1-2)
- `min_detection_confidence`: Minimum confidence for hand detection (0.0-1.0)
- `min_tracking_confidence`: Minimum confidence for hand tracking (0.0-1.0)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the license file for details.