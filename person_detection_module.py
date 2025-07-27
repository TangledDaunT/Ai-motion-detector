"""
Person Detection Module
======================

This module provides person detection functionality using YOLO (YOLOv5)
that can be integrated into the main detection system.
"""

import cv2
import time
from typing import List, Dict, Optional
from ultralytics import YOLO
import numpy as np
import config
from logger import log_info, log_error, log_warning, log_detection
from notifications import notify_person_detected
from advanced_ai_module import apply_smart_filtering

class PersonDetectionModule:
    """Person detection module using YOLO"""
    
    def __init__(self):
        self.model = None
        self.last_alert_time = 0
        self.detection_count = 0
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            log_info(f"Loading YOLO model: {config.YOLO_MODEL_PATH}", "PERSON")
            self.model = YOLO(config.YOLO_MODEL_PATH)
            log_info("YOLO model loaded successfully", "PERSON")
        except Exception as e:
            log_error(f"Failed to load YOLO model: {e}", "PERSON")
            self.model = None
    
    def detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in a frame using YOLO
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of dictionaries containing person detection results
        """
        if self.model is None:
            log_warning("YOLO model not loaded, skipping person detection", "PERSON")
            return []
        
        try:
            # Run YOLO inference
            results = self.model(frame, stream=True, verbose=False)
            
            detections = []
            
            for result in results:
                if result.boxes is None:
                    continue
                
                for box in result.boxes:
                    # Check if detection is a person (class 0 in COCO dataset)
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id == 0 and confidence > config.PERSON_CONFIDENCE_THRESHOLD:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        detection = {
                            'class_id': class_id,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                            'area': (x2 - x1) * (y2 - y1),
                            'type': 'person'  # For smart filtering
                        }
                        
                        detections.append(detection)
            
            # Apply smart filtering to reduce false positives
            filtered_detections = apply_smart_filtering(detections)
            
            # Handle valid detections
            for detection in filtered_detections:
                self.detection_count += 1
                self._handle_person_detection(detection)
            
            return filtered_detections
            
        except Exception as e:
            log_error(f"Error in person detection: {e}", "PERSON")
            return []
    
    def _handle_person_detection(self, detection: Dict):
        """
        Handle person detection result (notifications, logging)
        
        Args:
            detection: Person detection result dictionary
        """
        confidence = detection['confidence']
        bbox = detection['bbox']
        
        # Log the detection
        log_detection("person", {
            'confidence': confidence,
            'bbox': bbox,
            'camera_id': detection.get('camera_id', 0),
            'area': detection['area']
        })
        
        current_time = time.time()
        
        # Check cooldown for critical notifications
        if current_time - self.last_alert_time > config.PERSON_ALERT_COOLDOWN:
            # Send critical notification
            notify_person_detected(confidence * 100, bbox)  # Convert to percentage
            self.last_alert_time = current_time
            
            log_info(f"CRITICAL: Person detected (confidence: {confidence:.2f}, area: {detection['area']})", "PERSON")
        else:
            log_info(f"Person detected (confidence: {confidence:.2f}) - cooldown active", "PERSON")
    
    def draw_person_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw person detection results on frame
        
        Args:
            frame: Input frame
            detections: List of person detection results
            
        Returns:
            Frame with person detections drawn
        """
        for detection in detections:
            confidence = detection['confidence']
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Use red color for person detection (critical)
            color = config.COLORS['person']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"Person: {confidence:.2f}"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw text background
            cv2.rectangle(
                frame, 
                (x1, y1 - text_height - baseline - 5), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Draw text
            cv2.putText(
                frame, 
                label, 
                (x1, y1 - baseline - 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # Add "CRITICAL" indicator for high confidence detections
            if confidence > 0.8:
                critical_label = "CRITICAL"
                cv2.putText(
                    frame,
                    critical_label,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),  # Red
                    2
                )
        
        return frame
    
    def get_statistics(self) -> Dict:
        """Get person detection statistics"""
        return {
            'model_loaded': self.model is not None,
            'total_detections': self.detection_count,
            'last_alert_time': self.last_alert_time,
            'cooldown_remaining': max(0, config.PERSON_ALERT_COOLDOWN - (time.time() - self.last_alert_time))
        }
    
    def reset_alert_cooldown(self):
        """Reset alert cooldown (for testing purposes)"""
        self.last_alert_time = 0
        log_info("Person detection alert cooldown reset", "PERSON")
    
    def is_model_ready(self) -> bool:
        """Check if YOLO model is loaded and ready"""
        return self.model is not None
    
    def reload_model(self):
        """Reload the YOLO model"""
        log_info("Reloading YOLO model...", "PERSON")
        self.load_model()

# Create global instance
person_detection_module = PersonDetectionModule()

# Convenience functions for easy importing
def detect_persons(frame: np.ndarray) -> List[Dict]:
    """Detect persons in a frame"""
    return person_detection_module.detect_persons(frame)

def draw_person_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw person detections on frame"""
    return person_detection_module.draw_person_detections(frame, detections)

def get_person_statistics() -> Dict:
    """Get person detection statistics"""
    return person_detection_module.get_statistics()

def is_person_model_ready() -> bool:
    """Check if person detection model is ready"""
    return person_detection_module.is_model_ready()

def reset_person_alert_cooldown():
    """Reset person detection alert cooldown"""
    person_detection_module.reset_alert_cooldown()