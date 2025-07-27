"""
Motion Detection Module
======================

This module provides motion detection functionality using OpenCV
that can be integrated into the main detection system.
"""

import cv2
import time
import os
import numpy as np
import imutils
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import config
from logger import log_info, log_error, log_warning, log_detection, log_snapshot
from notifications import notify_motion_detected

class MotionDetectionModule:
    """Motion detection module using frame differencing"""
    
    def __init__(self):
        self.first_frame = None
        self.last_notify_time = 0
        self.detection_count = 0
        self.frame_update_counter = 0
    
    def detect_motion(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect motion in a frame using background subtraction
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of dictionaries containing motion detection results
        """
        try:
            # Convert to grayscale and blur
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Check lighting conditions
            avg_light = np.mean(gray)
            if avg_light < config.LOW_LIGHT_THRESHOLD:
                log_warning(f"Low light detected: {avg_light:.2f}", "MOTION")
            
            # Initialize first frame
            if self.first_frame is None:
                self.first_frame = gray
                return []
            
            # Calculate frame difference
            frame_delta = cv2.absdiff(self.first_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Get adaptive sensitivity based on time of day
            sensitivity = config.get_adaptive_sensitivity()
            
            # Find contours
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            
            detections = []
            
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                
                if contour_area < sensitivity:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                detection = {
                    'contour_area': contour_area,
                    'bbox': [x, y, x + w, y + h],
                    'center': [x + w // 2, y + h // 2],
                    'sensitivity_used': sensitivity,
                    'contour': contour
                }
                
                detections.append(detection)
                self.detection_count += 1
            
            # Handle motion detection if any motion found
            if detections:
                self._handle_motion_detection(detections, frame)
            
            # Update background frame periodically
            self._update_background_frame(gray)
            
            return detections
            
        except Exception as e:
            log_error(f"Error in motion detection: {e}", "MOTION")
            return []
    
    def _handle_motion_detection(self, detections: List[Dict], frame: np.ndarray):
        """
        Handle motion detection results (notifications, logging, snapshots)
        
        Args:
            detections: List of motion detection results
            frame: Original frame for snapshot saving
        """
        current_time = time.time()
        
        # Get the largest motion area for reporting
        largest_detection = max(detections, key=lambda d: d['contour_area'])
        
        # Log the detection
        log_detection("motion", {
            'contour_area': largest_detection['contour_area'],
            'sensitivity_used': largest_detection['sensitivity_used'],
            'camera_id': largest_detection.get('camera_id', 0),
            'detection_count': len(detections)
        })
        
        # Save snapshot
        camera_id = largest_detection.get('camera_id', 0)
        self._save_motion_snapshot(frame, camera_id)
        
        # Check cooldown for notifications
        if current_time - self.last_notify_time > config.MOTION_NOTIFY_COOLDOWN:
            # Send notification
            notify_motion_detected(
                largest_detection['contour_area'], 
                largest_detection['sensitivity_used']
            )
            self.last_notify_time = current_time
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_info(f"Motion detected at {timestamp} - Area: {largest_detection['contour_area']}", "MOTION")
        else:
            log_info(f"Motion detected (area: {largest_detection['contour_area']}) - cooldown active", "MOTION")
    
    def _save_motion_snapshot(self, frame: np.ndarray, camera_id: int = 0):
        """
        Save a snapshot of the motion detection
        
        Args:
            frame: Frame containing the motion
            camera_id: Camera index for the snapshot
        """
        try:
            snapshot_dir = config.get_snapshot_dir()
            os.makedirs(snapshot_dir, exist_ok=True)
            
            timestamp = int(time.time())
            snapshot_path = os.path.join(snapshot_dir, f"motion_{timestamp}.jpg")
            
            success = cv2.imwrite(snapshot_path, frame)
            if success:
                log_snapshot(camera_id, snapshot_path, "motion_detection")
                log_info(f"Motion snapshot saved: {snapshot_path} (Camera {camera_id})", "MOTION")
                
                # Prune old snapshots
                self._prune_snapshots(snapshot_dir)
            else:
                log_error(f"Failed to save motion snapshot: {snapshot_path}", "MOTION")
                
        except Exception as e:
            log_error(f"Error saving motion snapshot: {e}", "MOTION")
    
    def _prune_snapshots(self, directory: str):
        """
        Remove old snapshots to keep storage manageable
        
        Args:
            directory: Directory containing snapshots
        """
        try:
            all_files = [
                os.path.join(directory, f) 
                for f in os.listdir(directory) 
                if f.endswith('.jpg')
            ]
            
            # Sort by modification time
            all_files.sort(key=os.path.getmtime)
            
            # Remove oldest files if we exceed the limit
            while len(all_files) > config.KEEP_SNAPSHOTS_COUNT:
                oldest_file = all_files.pop(0)
                os.remove(oldest_file)
                log_info(f"Pruned old snapshot: {oldest_file}", "MOTION")
                
        except Exception as e:
            log_error(f"Error pruning snapshots: {e}", "MOTION")
    
    def _update_background_frame(self, gray_frame: np.ndarray):
        """
        Update the background frame periodically to adapt to lighting changes
        
        Args:
            gray_frame: Current grayscale frame
        """
        self.frame_update_counter += 1
        
        # Update background every 150 frames (approximately every 5 seconds at 30 FPS)
        if self.frame_update_counter >= 150:
            self.first_frame = gray_frame.copy()
            self.frame_update_counter = 0
            log_info("Background frame updated for lighting adaptation", "MOTION")
    
    def draw_motion_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw motion detection results on frame
        
        Args:
            frame: Input frame
            detections: List of motion detection results
            
        Returns:
            Frame with motion detections drawn
        """
        for detection in detections:
            bbox = detection['bbox']
            contour_area = detection['contour_area']
            x1, y1, x2, y2 = bbox
            
            # Use cyan color for motion detection
            color = config.COLORS['motion']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"Motion: {contour_area:.0f}px"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
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
                0.6, 
                (255, 255, 255), 
                2
            )
        
        return frame
    
    def get_statistics(self) -> Dict:
        """Get motion detection statistics"""
        return {
            'total_detections': self.detection_count,
            'background_initialized': self.first_frame is not None,
            'last_notify_time': self.last_notify_time,
            'cooldown_remaining': max(0, config.MOTION_NOTIFY_COOLDOWN - (time.time() - self.last_notify_time)),
            'current_sensitivity': config.get_adaptive_sensitivity(),
            'is_night_time': config.is_night_time()
        }
    
    def reset_background(self):
        """Reset the background frame (useful for recalibration)"""
        self.first_frame = None
        self.frame_update_counter = 0
        log_info("Motion detection background reset", "MOTION")
    
    def reset_notification_cooldown(self):
        """Reset notification cooldown (for testing purposes)"""
        self.last_notify_time = 0
        log_info("Motion detection notification cooldown reset", "MOTION")

# Create global instance
motion_detection_module = MotionDetectionModule()

# Convenience functions for easy importing
def detect_motion(frame: np.ndarray) -> List[Dict]:
    """Detect motion in a frame"""
    return motion_detection_module.detect_motion(frame)

def draw_motion_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw motion detections on frame"""
    return motion_detection_module.draw_motion_detections(frame, detections)

def get_motion_statistics() -> Dict:
    """Get motion detection statistics"""
    return motion_detection_module.get_statistics()

def reset_motion_background():
    """Reset motion detection background"""
    motion_detection_module.reset_background()

def reset_motion_notification_cooldown():
    """Reset motion detection notification cooldown"""
    motion_detection_module.reset_notification_cooldown()