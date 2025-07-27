# Unified Motion Detection System

A comprehensive, modular motion detection system with intelligent conditional detection logic.

## 🚀 Features

- **Camera Failover**: Automatic switch to backup camera when primary camera disconnects
- **Conditional Motion Detection**: Motion detection ONLY activates when person/face is detected first
- **Instant Person Alerts**: Immediate notifications for any person detection (even 20% confidence)
- **Face Recognition**: Identify known faces and detect unknown faces
- **Smart Detection Logic**: Person/face detection → instant alert → enable motion tracking
- **Palm Gesture Exit**: Graceful shutdown using palm detection
- **Unified Notifications**: Single Pushover notification every 10 seconds with activity summary
- **Centralized Logging**: All activities logged to a single file with timestamps
- **Single Output Screen**: Combined view of all detection types with color-coded bounding boxes
- **Robust Operation**: Continues working even if cameras disconnect
- **Automatic Cleanup**: Old snapshots and logs are automatically managed

## 📁 Project Structure

```
motion detection/
├── lab.py                          # Main entry point - run this file
├── config.py                       # Unified configuration settings
├── logger.py                       # Centralized logging system
├── notifications.py                # Unified notification system
├── face_recognition_module.py      # Face recognition functionality
├── person_detection_module.py      # YOLO person detection
├── motion_detection_module.py      # Motion detection functionality
├── known_faces/                    # Directory for known face images
├── unknown_faces/                  # Directory for unknown face snapshots
└── README.md                       # This documentation
```

## 🛠️ Installation

### Prerequisites

```bash
pip install opencv-python
pip install face-recognition
pip install ultralytics
pip install mediapipe
pip install imutils
pip install requests
pip install numpy
```

### Setup

1. **Configure Pushover** (for notifications):
   - Install Pushover app on your phone
   - Create account at https://pushover.net
   - Get your User Key and create an App Token
   - Update `config.py` with your credentials

2. **Add Known Faces**:
   - Create folders in `known_faces/` directory for each person
   - Add multiple photos of each person (e.g., `known_faces/john/img1.jpg`)

3. **Camera Permissions** (macOS):
   - Go to System Preferences > Security & Privacy > Privacy > Camera
   - Enable camera access for Terminal or your Python IDE

## 🎯 Usage

### Basic Usage
```bash
python lab.py                      # Dual camera mode (default)
python lab.py --single-camera      # Single camera mode
```

### Command Line Options
```bash
python lab.py --help                # Show help
python lab.py --no-preview         # Run without GUI (headless mode)
python lab.py --test               # Run in test mode with extra logging
python lab.py --single-camera      # Use only first camera (disable dual mode)
```

### Controls
- **Q key**: Quit the application
- **Palm gesture**: Hold open palm for 2 seconds (60 frames) to exit gracefully
- **Ctrl+C**: Force quit

## 🎨 Visual Interface

The single output window shows:
- **Cyan boxes**: Motion detection
- **Green boxes**: Known faces with names and confidence
- **Red boxes**: Unknown faces or person detection
- **Yellow lines**: Palm detection landmarks
- **Status overlay**: Real-time statistics and information

## 📊 Detection Logic Flow

### 1. Person Detection (Primary Trigger)
- YOLO-based detection with **20% confidence threshold** (catches even uncertain detections)
- **INSTANT notifications** for any person detection (no cooldown)
- Automatically enables motion tracking when person detected
- Bounding boxes with confidence scores

### 2. Face Recognition (Secondary Trigger)
- Loads known faces from `known_faces/` directory
- Confidence threshold: 50% (configurable)
- Also enables motion tracking when face detected
- Automatic snapshot saving for known faces
- Unknown faces saved to `unknown_faces/` directory

### 3. Conditional Motion Detection
- **ONLY activates when person OR face is detected first**
- Uses frame differencing with Gaussian blur (sensitivity: 2500 pixels)
- Adaptive sensitivity based on time of day (more sensitive at night)
- Automatic background frame updates for lighting changes
- **Disables after 30 seconds** of no person/face detection
- Snapshots saved to dated directories

### Palm Detection
- MediaPipe-based hand landmark detection
- Requires all 4 fingers extended for 2 seconds
- Graceful shutdown mechanism
- Visual feedback during detection

### Camera Failover System
- **Primary Camera**: Index 0 (default)
- **Backup Camera**: Index 1 (automatic failover)
- **Failure Detection**: Monitors for 10 consecutive frame read failures
- **Automatic Switch**: Seamlessly switches to backup camera when primary fails
- **Critical Notifications**: Sends "📹 Camera X down - switched to camera Y" alert
- **Continuous Operation**: System continues running even with camera disconnections
- **Health Monitoring**: Displays camera failure count and current camera in use

## 📝 Logging

### Unified Log File
Location: `/Users/shreyansh/DOCS/motion_detection_unified.log`

Format:
```
[2025-01-28T02:47:18.819123] [INFO] [FACE] Known face detected: shreyansh (confidence: 85.2%)
[2025-01-28T02:47:19.123456] [CRITICAL] [PERSON] Person detected (confidence: 0.87, area: 15420)
[2025-01-28T02:47:20.234567] [INFO] [MOTION] Motion detected (area: 8500 pixels)
```

### CSV Logs
- **Face Recognition**: `face_recognition_log.csv`
- **Snapshots**: `/Users/shreyansh/DOCS/snapshot_log.csv`

## 📱 Notifications

### Unified Notifications (Every 10 seconds)
```
🕐 14:30:15 Detection Summary:
🚨 CRITICAL:
  • Person detected (87.3%)
👤 Faces:
  • shreyansh (85.2%)
📊 Activity: Motion: 3, Unknown faces: 1
```

### Notification Types
- **🚨 CRITICAL**: Person detection (immediate priority)
- **👤 Faces**: Known face recognition
- **📊 Activity**: Motion and other events summary
- **⚙️ System**: Startup, shutdown, errors

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Pushover credentials
API_TOKEN = "your_api_token"
USER_KEY = "your_user_key"

# Detection sensitivity
MOTION_SENSITIVITY = 5000000
FACE_RECOGNITION_THRESHOLD = 0.5
PERSON_CONFIDENCE_THRESHOLD = 0.5

# Notification timing
UNIFIED_NOTIFY_COOLDOWN = 10  # seconds
FACE_NOTIFY_COOLDOWN = 30     # seconds per person
PERSON_ALERT_COOLDOWN = 30    # seconds

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 500
```

## 🔧 Troubleshooting

### Common Issues

1. **Camera not working**:
   - Check camera permissions
   - Try different `CAMERA_INDEX` values (0, 1, 2...)
   - Ensure camera isn't used by another application

2. **YOLO model download fails**:
   - Check internet connection
   - Model will download automatically on first run
   - File saved as `yolov5s.pt` in project directory

3. **Face recognition not working**:
   - Ensure `known_faces/` directory exists
   - Add multiple clear photos of each person
   - Check lighting conditions

4. **Notifications not sending**:
   - Verify Pushover credentials in `config.py`
   - Check internet connection
   - Test with Pushover's web interface

### Performance Tips

- **Reduce frame size**: Lower `FRAME_WIDTH` in config
- **Adjust sensitivity**: Increase `MOTION_SENSITIVITY` to reduce false positives
- **Disable preview**: Use `--no-preview` for better performance
- **Lighting**: Ensure good lighting for better face recognition

## 📈 System Requirements

- **Python**: 3.7 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Multi-core processor recommended
- **Camera**: USB webcam or built-in camera
- **OS**: macOS, Linux, Windows (tested on macOS)

## 🔄 Maintenance

### Automatic Cleanup
- Old snapshots deleted after 1 hour
- Empty directories removed automatically
- Log rotation (can be configured)

### Manual Maintenance
```bash
# Clear all snapshots
rm -rf /Users/shreyansh/DOCS/snapshots/*

# Reset face recognition cooldowns
python -c "from face_recognition_module import reset_notification_cooldowns; reset_notification_cooldowns()"

# View system statistics
python -c "
from face_recognition_module import get_face_statistics
from person_detection_module import get_person_statistics
from motion_detection_module import get_motion_statistics
print('Face:', get_face_statistics())
print('Person:', get_person_statistics())
print('Motion:', get_motion_statistics())
"
```

## 🆘 Support

### Log Analysis
Check the unified log file for detailed information:
```bash
tail -f /Users/shreyansh/DOCS/motion_detection_unified.log
```

### Debug Mode
Run with additional logging:
```bash
python lab.py --test
```

### Reset System
If you encounter issues, try:
1. Delete `yolov5s.pt` to re-download YOLO model
2. Clear log files
3. Restart the application

## 📄 License

This project is for personal use. Ensure you comply with the licenses of all dependencies:
- OpenCV: Apache 2.0
- face_recognition: MIT
- ultralytics: AGPL-3.0
- MediaPipe: Apache 2.0

## 🔄 Version History

- **v2.0**: Complete refactoring with modular architecture
- **v1.0**: Original monolithic implementation

---

**Note**: This system processes video locally and only sends notification summaries via Pushover. No video data is transmitted or stored externally.