"""
Centralized Logging System for Motion Detection
==============================================

This module provides a unified logging system for all detection modules.
It handles both text logging and CSV logging with proper formatting and error handling.
"""

import os
import csv
import threading
from datetime import datetime
from typing import Optional, Dict, Any
import config

class UnifiedLogger:
    """Centralized logger for all motion detection modules"""
    
    def __init__(self):
        self.log_lock = threading.Lock()
        self.csv_lock = threading.Lock()
        self._ensure_log_files()
    
    def _ensure_log_files(self):
        """Ensure all log files and directories exist"""
        try:
            config.ensure_directories()
            
            # Create unified log file if it doesn't exist
            if not os.path.exists(config.UNIFIED_LOG_FILE):
                with open(config.UNIFIED_LOG_FILE, 'w') as f:
                    f.write(f"=== Motion Detection System Log Started at {datetime.now().isoformat()} ===\n")
            
            # Create CSV headers if files don't exist
            if not os.path.exists(config.FACE_RECOGNITION_LOG_CSV):
                with open(config.FACE_RECOGNITION_LOG_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'name', 'camera_id', 'confidence'])
            
            if not os.path.exists(config.SNAPSHOT_LOG_CSV):
                with open(config.SNAPSHOT_LOG_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'camera_id', 'snapshot_path', 'detection_type'])
                    
        except Exception as e:
            print(f"[ERROR] Failed to initialize log files: {e}")
    
    def log(self, message: str, module: str = "SYSTEM", level: str = "INFO"):
        """
        Log a message to the unified log file
        
        Args:
            message: The message to log
            module: The module name (MOTION, FACE, PERSON, PALM, SYSTEM)
            level: Log level (INFO, WARNING, ERROR, CRITICAL)
        """
        timestamp = datetime.now().isoformat()
        formatted_message = f"[{timestamp}] [{level}] [{module}] {message}"
        
        try:
            with self.log_lock:
                with open(config.UNIFIED_LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(formatted_message + '\n')
                    f.flush()
                
                # Also print to console for immediate feedback
                print(formatted_message)
                
        except Exception as e:
            print(f"[ERROR] Failed to write to log file: {e}")
    
    def log_detection(self, detection_type: str, details: Dict[str, Any]):
        """
        Log a detection event with structured data
        
        Args:
            detection_type: Type of detection (motion, face_known, face_unknown, person, palm)
            details: Dictionary containing detection details
        """
        timestamp = datetime.now().isoformat()
        
        if detection_type == "face_known" or detection_type == "face_unknown":
            self._log_face_detection(details, timestamp)
        elif detection_type == "motion":
            self._log_motion_detection(details, timestamp)
        elif detection_type == "person":
            self._log_person_detection(details, timestamp)
        elif detection_type == "palm":
            self._log_palm_detection(details, timestamp)
        
        # Always log to unified log
        message = self._format_detection_message(detection_type, details)
        module = self._get_module_name(detection_type)
        self.log(message, module, "INFO")
    
    def _log_face_detection(self, details: Dict[str, Any], timestamp: str):
        """Log face detection to CSV"""
        try:
            with self.csv_lock:
                with open(config.FACE_RECOGNITION_LOG_CSV, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        timestamp,
                        details.get('name', 'Unknown'),
                        details.get('camera_id', 0),
                        details.get('confidence', 0.0)
                    ])
        except Exception as e:
            self.log(f"Failed to log face detection to CSV: {e}", "FACE", "ERROR")
    
    def _log_motion_detection(self, details: Dict[str, Any], timestamp: str):
        """Log motion detection"""
        contour_area = details.get('contour_area', 0)
        sensitivity = details.get('sensitivity_used', config.MOTION_SENSITIVITY)
        self.log(f"Motion detected - Area: {contour_area}, Sensitivity: {sensitivity}", "MOTION", "INFO")
    
    def _log_person_detection(self, details: Dict[str, Any], timestamp: str):
        """Log person detection"""
        confidence = details.get('confidence', 0.0)
        bbox = details.get('bbox', [0, 0, 0, 0])
        self.log(f"Person detected - Confidence: {confidence:.2f}, BBox: {bbox}", "PERSON", "CRITICAL")
    
    def _log_palm_detection(self, details: Dict[str, Any], timestamp: str):
        """Log palm detection"""
        frames_held = details.get('frames_held', 0)
        self.log(f"Palm detected - Frames held: {frames_held}", "PALM", "INFO")
    
    def log_snapshot(self, camera_id: int, snapshot_path: str, detection_type: str):
        """
        Log snapshot creation to CSV
        
        Args:
            camera_id: Camera index
            snapshot_path: Path to saved snapshot
            detection_type: Type of detection that triggered snapshot
        """
        try:
            with self.csv_lock:
                with open(config.SNAPSHOT_LOG_CSV, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        camera_id,
                        snapshot_path,
                        detection_type
                    ])
            
            self.log(f"Snapshot saved: {snapshot_path} (Type: {detection_type})", "SNAPSHOT", "INFO")
            
        except Exception as e:
            self.log(f"Failed to log snapshot: {e}", "SNAPSHOT", "ERROR")
    
    def log_system_event(self, event: str, details: Optional[str] = None):
        """
        Log system events (startup, shutdown, errors, etc.)
        
        Args:
            event: Event type (startup, shutdown, error, etc.)
            details: Additional details about the event
        """
        message = f"System event: {event}"
        if details:
            message += f" - {details}"
        
        level = "ERROR" if "error" in event.lower() or "crash" in event.lower() else "INFO"
        self.log(message, "SYSTEM", level)
    
    def log_error(self, error: Exception, module: str, context: str = ""):
        """
        Log an error with full context
        
        Args:
            error: The exception that occurred
            module: Module where error occurred
            context: Additional context about what was happening
        """
        message = f"Error in {context}: {str(error)}" if context else f"Error: {str(error)}"
        self.log(message, module, "ERROR")
    
    def log_warning(self, message: str, module: str):
        """Log a warning message"""
        self.log(message, module, "WARNING")
    
    def log_critical(self, message: str, module: str):
        """Log a critical message"""
        self.log(message, module, "CRITICAL")
    
    def _format_detection_message(self, detection_type: str, details: Dict[str, Any]) -> str:
        """Format detection details into a readable message"""
        if detection_type == "face_known":
            name = details.get('name', 'Unknown')
            confidence = details.get('confidence', 0.0)
            return f"Known face detected: {name} (confidence: {confidence:.2f})"
        
        elif detection_type == "face_unknown":
            confidence = details.get('confidence', 0.0)
            return f"Unknown face detected (confidence: {confidence:.2f})"
        
        elif detection_type == "motion":
            area = details.get('contour_area', 0)
            return f"Motion detected (area: {area} pixels)"
        
        elif detection_type == "person":
            confidence = details.get('confidence', 0.0)
            return f"Person detected (confidence: {confidence:.2f})"
        
        elif detection_type == "palm":
            frames = details.get('frames_held', 0)
            return f"Palm detected ({frames} frames held)"
        
        return f"{detection_type} detected"
    
    def _get_module_name(self, detection_type: str) -> str:
        """Get module name from detection type"""
        mapping = {
            'face_known': 'FACE',
            'face_unknown': 'FACE',
            'motion': 'MOTION',
            'person': 'PERSON',
            'palm': 'PALM'
        }
        return mapping.get(detection_type, 'UNKNOWN')
    
    def cleanup_old_logs(self, days_to_keep: int = 7):
        """Clean up old log entries (optional maintenance function)"""
        try:
            # This is a placeholder for log rotation functionality
            # Could be implemented to keep logs manageable
            self.log(f"Log cleanup requested (keep {days_to_keep} days)", "SYSTEM", "INFO")
        except Exception as e:
            self.log(f"Log cleanup failed: {e}", "SYSTEM", "ERROR")

# Global logger instance
logger = UnifiedLogger()

# Convenience functions for easy importing
def log_info(message: str, module: str = "SYSTEM"):
    """Log an info message"""
    logger.log(message, module, "INFO")

def log_warning(message: str, module: str = "SYSTEM"):
    """Log a warning message"""
    logger.log(message, module, "WARNING")

def log_error(message: str, module: str = "SYSTEM"):
    """Log an error message"""
    logger.log(message, module, "ERROR")

def log_critical(message: str, module: str = "SYSTEM"):
    """Log a critical message"""
    logger.log(message, module, "CRITICAL")

def log_detection(detection_type: str, details: Dict[str, Any]):
    """Log a detection event"""
    logger.log_detection(detection_type, details)

def log_snapshot(camera_id: int, snapshot_path: str, detection_type: str):
    """Log a snapshot"""
    logger.log_snapshot(camera_id, snapshot_path, detection_type)

def log_system_event(event: str, details: Optional[str] = None):
    """Log a system event"""
    logger.log_system_event(event, details)