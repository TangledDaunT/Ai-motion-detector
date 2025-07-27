"""
Unified Configuration Module for Motion Detection System
======================================================

This module contains all configuration settings for the motion detection system,
including API keys, file paths, detection parameters, and system settings.
"""

import os
from datetime import datetime

# -----------------------------
# PUSHOVER NOTIFICATION SETTINGS
# -----------------------------
API_TOKEN = "ap5igza77o7kq8j6xtfkacv6g19xma"
USER_KEY = "us9hse3b4mvx8e23fobahwad1e53kd"

# -----------------------------
# CAMERA SETTINGS
# -----------------------------
CAMERA_INDICES = [0, 1]  # Primary camera 0, backup camera 1
FRAME_WIDTH = 500
DUAL_CAMERA_MODE = False  # Single camera with failover support
CAMERA_FAILOVER_ENABLED = True  # Enable automatic camera switching

# -----------------------------
# DETECTION SETTINGS
# -----------------------------
# Motion detection sensitivity (minimum contour area in pixels)
MOTION_SENSITIVITY = 2500

# Face recognition settings
FACE_RECOGNITION_THRESHOLD = 0.5
FACE_NOTIFY_COOLDOWN = 30  # seconds per person

# Person detection settings (YOLO)
PERSON_CONFIDENCE_THRESHOLD = 0.2  # Lower threshold to catch even low confidence detections
PERSON_ALERT_COOLDOWN = 0   # Instant notifications for person detection

# Palm detection settings
PALM_HOLD_FRAMES = 60  # frames to hold palm before exit (approximately 2 seconds at 30 FPS)
PALM_DETECTION_CONFIDENCE = 0.7
PALM_TRACKING_CONFIDENCE = 0.7

# Advanced AI features
AGE_GENDER_DETECTION_ENABLED = True
EMOTION_DETECTION_ENABLED = True
SMART_FILTERING_ENABLED = True

# Age/Gender detection settings
AGE_GENDER_CONFIDENCE_THRESHOLD = 0.6

# Emotion detection settings
EMOTION_CONFIDENCE_THRESHOLD = 0.5

# Smart filtering settings
FALSE_POSITIVE_THRESHOLD = 0.3  # Confidence below this is considered false positive
DETECTION_HISTORY_SIZE = 10     # Number of recent detections to analyze
CONSISTENCY_THRESHOLD = 0.7     # Percentage of consistent detections required

# Remote control settings
REMOTE_CONTROL_ENABLED = True
REMOTE_CONTROL_FILE = "/tmp/motion_detection_control.json"
DEFAULT_ARMED_STATE = True

# -----------------------------
# NOTIFICATION SETTINGS
# -----------------------------
UNIFIED_NOTIFY_COOLDOWN = 10  # seconds - unified notification every 10 seconds
MOTION_NOTIFY_COOLDOWN = 10   # seconds for motion detection

# -----------------------------
# FILE PATHS AND DIRECTORIES
# -----------------------------
BASE_DIR = "/Users/shreyansh/DOCS"
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"

# Log files
UNIFIED_LOG_FILE = os.path.join(BASE_DIR, "motion_detection_unified.log")
FACE_RECOGNITION_LOG_CSV = "face_recognition_log.csv"
SNAPSHOT_LOG_CSV = os.path.join(BASE_DIR, "snapshot_log.csv")

# Snapshot directories
SNAPSHOTS_BASE_DIR = os.path.join(BASE_DIR, "snapshots")

# Lock file for face recognition
LOCKFILE = "/tmp/face_rec.lock"

# -----------------------------
# YOLO MODEL SETTINGS
# -----------------------------
YOLO_MODEL_PATH = "yolov5s.pt"

# -----------------------------
# DISPLAY SETTINGS
# -----------------------------
SHOW_PREVIEW = True
WINDOW_NAME = "Unified Motion Detection System"

# Colors for different detection types (BGR format)
COLORS = {
    'motion': (255, 255, 0),      # Cyan for motion
    'face_known': (0, 255, 0),   # Green for known faces
    'face_unknown': (0, 0, 255), # Red for unknown faces
    'person': (0, 0, 255),       # Red for person detection
    'palm': (0, 255, 255)        # Yellow for palm detection
}

# -----------------------------
# SYSTEM SETTINGS
# -----------------------------
# Snapshot management
KEEP_SNAPSHOTS_COUNT = 50
SNAPSHOT_CLEANUP_HOURS = 1  # Delete snapshots older than 1 hour

# Lighting threshold
LOW_LIGHT_THRESHOLD = 40

# Adaptive sensitivity settings
NIGHT_SENSITIVITY_MULTIPLIER = 0.6  # More sensitive at night
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 6

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_snapshot_dir():
    """Get today's snapshot directory path"""
    today = datetime.now().strftime('%Y-%m-%d')
    return os.path.join(SNAPSHOTS_BASE_DIR, today)

def is_night_time():
    """Check if current time is considered night time"""
    hour = datetime.now().hour
    return hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR

def get_adaptive_sensitivity():
    """Get motion sensitivity adjusted for time of day"""
    base_sensitivity = MOTION_SENSITIVITY
    if is_night_time():
        return int(base_sensitivity * NIGHT_SENSITIVITY_MULTIPLIER)
    return base_sensitivity

# -----------------------------
# VALIDATION
# -----------------------------
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    if not API_TOKEN or API_TOKEN == "your_api_token":
        errors.append("Invalid API_TOKEN - please set your Pushover API token")
    
    if not USER_KEY or USER_KEY == "your_user_key":
        errors.append("Invalid USER_KEY - please set your Pushover user key")
    
    if not os.path.exists(KNOWN_FACES_DIR):
        errors.append(f"Known faces directory does not exist: {KNOWN_FACES_DIR}")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True

# Create necessary directories
def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        UNKNOWN_FACES_DIR,
        SNAPSHOTS_BASE_DIR,
        os.path.dirname(UNIFIED_LOG_FILE),
        os.path.dirname(SNAPSHOT_LOG_CSV)
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)