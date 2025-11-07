# Mirror of Erised - Interactive Smart Mirror

An emotion-aware smart mirror inspired by Harry Potter that detects user emotions and creates immersive, conversational experiences using AI.

## Overview

This project reimagines the Mirror of Erised as an interactive pervasive computing system. The mirror detects user presence, recognizes emotional states, and adapts its responses with personalized backgrounds and conversational AI—all housed in a 3D-printed Harry Potter-inspired enclosure.

## Features

- **Real-time Emotion Detection**: Uses webcam + EmotiEffLib to identify emotions (happiness, sadness, anger, surprise, etc.)
- **Dynamic Backgrounds**: Displays emotion-matched immersive scenes generated with Midjourney
- **AI Conversation**: Natural dialogue powered by OpenAI Realtime API with mystical Harry Potter-style responses
- **Background Segmentation**: Separates user from background using MediaPipe for AR-like overlay effect
- **Auto-activation**: Detects faces and initiates interaction; returns to mirror mode when idle

## Hardware Components

- Raspberry Pi 5 (8GB RAM)
- 10.1" LCD Display (1024x600)
- Two-way acrylic mirror (30% transparency)
- Logitech C510 720p webcam
- Mini sound-bar speaker
- Custom 3D-printed enclosure

## Software Stack

- **OpenCV**: Video processing and frame manipulation
- **EmotiEffLib**: Lightweight emotion recognition
- **MediaPipe**: Real-time face/body segmentation
- **OpenAI Realtime API**: Speech-to-text, text-to-speech, and conversational AI
- **uv**: Python package management

## Installation

### Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install uv python manager
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Python Dependencies
```bash
# Clone repository
git clone https://github.com/Jose-AE/echo-miro.git
cd echo-miro

# Install deps with uv (recommended)
uv sync
```

### Configuration
1. Set up OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

2. Organize emotion background images in folders:
```
backgrounds/
├── anger/
├── happiness/
├── sadness/
├── fear/
├── disgust/
├── surprise/
├── neutral/
└── contempt/
```

## Usage

### Run on Boot (Recommended)
```bash
# Edit autostart
mkdir -p ~/.config/autostart
nano ~/.config/autostart/echo-miro.desktop

#Paste inside file (replace <dir_of_cloned_repo>):
[Desktop Entry]
Type=Application
Name=Echo Miro
Exec=bash -c "cd <dir_of_cloned_repo> && /home/echo-miro/.local/bin/uv run src/main.py"
StartupNotify=false
```

### Manual Start
```bash
uv run .src/main.py
```

### User Interaction Flow
1. Stand in front of mirror (camera at eye level)
2. Wait ~5 seconds for emotion detection
3. Mirror displays emotion-matched background and initiates conversation
4. Speak naturally after mirror finishes talking
5. Walk away to return mirror to idle state

## Project Structure

```
mirror-of-erised/
├── src/
│   ├── main.py                    # Entry point
│   ├── echo_miro.py              # Main coordinator
│   ├── realtime_voice_client.py  # Audio I/O with OpenAI
│   └── opencv_controller.py      # Webcam + emotion detection
├── backgrounds/                   # Emotion-specific images
├── models/                        # 3D enclosure files (STL)
└── requirements.txt
```
