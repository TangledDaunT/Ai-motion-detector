"""
Face Recognition Module
======================

This module provides face recognition functionality that can be integrated
into the main detection system. It handles loading known faces, detecting
faces in frames, and managing face recognition logic.
"""

import os
import cv2
import numpy as np
import face_recognition
import time
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import config
from logger import log_info, log_error, log_warning, log_detection, log_snapshot
from notifications import notify_face_detected
from advanced_ai_module import analyze_person, format_analysis_for_display, format_analysis_for_notification

class FaceRecognitionModule:
    """Face recognition module for detecting and identifying faces"""
    
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.last_notify_times = {}
        self.unknown_face_count = 0
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known face encodings from the known_faces directory"""
        try:
            if not os.path.exists(config.KNOWN_FACES_DIR):
                log_warning(f"Known faces directory not found: {config.KNOWN_FACES_DIR}", "FACE")
                return
            
            self.known_encodings = []
            self.known_names = []
            
            for name in os.listdir(config.KNOWN_FACES_DIR):
                person_dir = os.path.join(config.KNOWN_FACES_DIR, name)
                if not os.path.isdir(person_dir):
                    continue
                
                person_encodings = 0
                for filename in os.listdir(person_dir):
                    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    
                    filepath = os.path.join(person_dir, filename)
                    try:
                        image = face_recognition.load_image_file(filepath)
                        encodings = face_recognition.face_encodings(image)
                        
                        if encodings:
                            self.known_encodings.append(encodings[0])
                            self.known_names.append(name)
                            person_encodings += 1
                        else:
                            log_warning(f"No face found in {filepath}", "FACE")
                            
                    except Exception as e:
                        log_error(f"Could not process {filepath}: {e}", "FACE")
                
                if person_encodings > 0:
                    log_info(f"Loaded {person_encodings} encodings for {name}", "FACE")
            
            log_info(f"Total loaded: {len(self.known_encodings)} face encodings for {len(set(self.known_names))} people", "FACE")
            
        except Exception as e:
            log_error(f"Failed to load known faces: {e}", "FACE")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and recognize faces in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of dictionaries containing face detection results
        """
        try:
            # Convert BGR to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            results = []
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Try to match with known faces
                name, confidence = self._identify_face(face_encoding)
                
                # Extract face image for AI analysis
                top, right, bottom, left = face_location
                face_image = rgb_frame[top:bottom, left:right]
                
                # Perform AI analysis (age, gender, emotion)
                person_id = f"{name}_{int(time.time())}" if name != "Unknown" else None
                ai_analysis = analyze_person(face_image, person_id)
                
                # Create result dictionary
                result = {
                    'name': name,
                    'confidence': confidence,
                    'location': face_location,  # (top, right, bottom, left)
                    'is_known': name != "Unknown",
                    'bbox': self._location_to_bbox(face_location, frame.shape),
                    'ai_analysis': ai_analysis,
                    'type': 'face'  # For smart filtering
                }
                
                results.append(result)
                
                # Handle notifications and logging
                self._handle_face_detection(result, frame)
            
            return results
            
        except Exception as e:
            log_error(f"Error in face detection: {e}", "FACE")
            return []
    
    def _identify_face(self, face_encoding: np.ndarray) -> Tuple[str, float]:
        """
        Identify a face encoding against known faces
        
        Args:
            face_encoding: Face encoding to identify
            
        Returns:
            Tuple of (name, confidence)
        """
        if not self.known_encodings:
            return "Unknown", 0.0
        
        # Calculate distances to all known faces
        distances = face_recognition.face_distance(self.known_encodings, face_encoding)
        best_match_index = np.argmin(distances)
        best_distance = distances[best_match_index]
        
        # Convert distance to confidence (lower distance = higher confidence)
        confidence = max(0, (1 - best_distance) * 100)
        
        if best_distance < config.FACE_RECOGNITION_THRESHOLD:
            name = self.known_names[best_match_index]
            return name, confidence
        else:
            return "Unknown", confidence
    
    def _location_to_bbox(self, face_location: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> List[int]:
        """
        Convert face_recognition location format to bounding box
        
        Args:
            face_location: (top, right, bottom, left) from face_recognition
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            Bounding box as [x1, y1, x2, y2]
        """
        top, right, bottom, left = face_location
        return [left, top, right, bottom]
    
    def _handle_face_detection(self, result: Dict, frame: np.ndarray):
        """
        Handle face detection result (notifications, logging, snapshots)
        
        Args:
            result: Face detection result dictionary
            frame: Original frame for snapshot saving
        """
        name = result['name']
        confidence = result['confidence']
        is_known = result['is_known']
        
        # Log the detection
        log_detection("face_known" if is_known else "face_unknown", {
            'name': name,
            'confidence': confidence,
            'camera_id': result.get('camera_id', 0)
        })
        
        current_time = time.time()
        
        if is_known:
            # Handle known face
            if name not in self.last_notify_times or \
               current_time - self.last_notify_times[name] > config.FACE_NOTIFY_COOLDOWN:
                
                # Send notification
                notify_face_detected(name, confidence, True)
                self.last_notify_times[name] = current_time
                
                # Save snapshot
                self._save_face_snapshot(name, frame, is_known=True)
                
                log_info(f"Known face detected: {name} (confidence: {confidence:.1f}%)", "FACE")
        else:
            # Handle unknown face
            notify_face_detected("Unknown", confidence, False)
            self._save_face_snapshot("Unknown", frame, is_known=False)
            log_info(f"Unknown face detected (confidence: {confidence:.1f}%)", "FACE")
    
    def _save_face_snapshot(self, name: str, frame: np.ndarray, is_known: bool = True):
        """
        Save a snapshot of the detected face
        
        Args:
            name: Name of the person (or "Unknown")
            frame: Frame containing the face
            is_known: Whether this is a known or unknown face
        """
        try:
            if is_known and name != "Unknown":
                # Save to person's directory
                person_dir = os.path.join(config.KNOWN_FACES_DIR, name)
                os.makedirs(person_dir, exist_ok=True)
                
                # Count existing images to generate next number
                existing_imgs = [
                    f for f in os.listdir(person_dir) 
                    if f.lower().startswith('img') and f.lower().endswith('.jpg')
                ]
                n = len(existing_imgs) + 1
                snapshot_path = os.path.join(person_dir, f"img{n}.jpg")
                
            else:
                # Save to unknown faces directory
                os.makedirs(config.UNKNOWN_FACES_DIR, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.unknown_face_count += 1
                snapshot_path = os.path.join(
                    config.UNKNOWN_FACES_DIR, 
                    f"unknown_{timestamp}_{self.unknown_face_count}.jpg"
                )
            
            # Save the frame
            success = cv2.imwrite(snapshot_path, frame)
            if success:
                camera_id = 0  # Default camera ID, will be updated by caller
                log_snapshot(camera_id, snapshot_path, "face_detection")
                log_info(f"Face snapshot saved: {snapshot_path}", "FACE")
            else:
                log_error(f"Failed to save face snapshot: {snapshot_path}", "FACE")
                
        except Exception as e:
            log_error(f"Error saving face snapshot: {e}", "FACE")
    
    def draw_face_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw face detection results on frame
        
        Args:
            frame: Input frame
            detections: List of face detection results
            
        Returns:
            Frame with face detections drawn
        """
        for detection in detections:
            name = detection['name']
            confidence = detection['confidence']
            bbox = detection['bbox']
            is_known = detection['is_known']
            
            # Choose color based on whether face is known
            color = config.COLORS['face_known'] if is_known else config.COLORS['face_unknown']
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create main label
            main_label = f"{name} ({confidence:.1f}%)" if is_known else f"Unknown ({confidence:.1f}%)"
            
            # Add AI analysis if available
            ai_analysis = detection.get('ai_analysis', {})
            ai_text = format_analysis_for_display(ai_analysis)
            
            labels = [main_label]
            if ai_text and ai_text != "AI: Not available":
                labels.append(ai_text)
            
            # Calculate total text area needed
            total_height = 0
            max_width = 0
            label_info = []
            
            for i, label in enumerate(labels):
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 if i > 0 else 0.6, 2
                )
                label_info.append((label, text_width, text_height, baseline))
                total_height += text_height + baseline + 5
                max_width = max(max_width, text_width)
            
            # Draw text background
            bg_y1 = y1 - total_height - 5
            bg_y2 = y1
            cv2.rectangle(frame, (x1, bg_y1), (x1 + max_width + 10, bg_y2), color, -1)
            
            # Draw labels
            current_y = y1 - 5
            for i, (label, text_width, text_height, baseline) in enumerate(label_info):
                font_scale = 0.5 if i > 0 else 0.6
                text_color = (255, 255, 255) if i == 0 else (200, 200, 200)
                
                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, current_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    2 if i == 0 else 1
                )
                current_y -= (text_height + baseline + 3)
        
        return frame
    
    def get_statistics(self) -> Dict:
        """Get face recognition statistics"""
        return {
            'known_faces_loaded': len(self.known_encodings),
            'unique_people': len(set(self.known_names)),
            'unknown_faces_detected': self.unknown_face_count,
            'last_notifications': dict(self.last_notify_times)
        }
    
    def reload_known_faces(self):
        """Reload known faces from directory (useful for adding new faces)"""
        log_info("Reloading known faces...", "FACE")
        self.load_known_faces()
    
    def reset_notification_cooldowns(self):
        """Reset all notification cooldowns (for testing purposes)"""
        self.last_notify_times.clear()
        log_info("Face recognition notification cooldowns reset", "FACE")

# Create global instance
face_recognition_module = FaceRecognitionModule()

# Convenience functions for easy importing
def detect_faces(frame: np.ndarray) -> List[Dict]:
    """Detect faces in a frame"""
    return face_recognition_module.detect_faces(frame)

def draw_face_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw face detections on frame"""
    return face_recognition_module.draw_face_detections(frame, detections)

def get_face_statistics() -> Dict:
    """Get face recognition statistics"""
    return face_recognition_module.get_statistics()

def reload_known_faces():
    """Reload known faces"""
    face_recognition_module.reload_known_faces()