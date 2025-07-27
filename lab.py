"""
Unified Motion Detection System - Main Controller (Camera Failover Support)
==========================================================================

This is the main entry point for the motion detection system with camera failover.
If the primary camera disconnects, it automatically switches to backup camera
and sends a "camera down" notification.

Detection Logic:
1. First detect persons and faces
2. Only check for motion if person/face is detected
3. Send instant notifications for any person detection (even low confidence)
4. Automatic camera failover with notifications

Usage:
    python lab.py [--no-preview] [--test]

Features:
- Camera failover (primary → backup automatically)
- Person detection with instant notifications (low threshold)
- Face recognition with known faces
- Motion detection ONLY when person/face detected
- Palm detection for graceful exit
- Unified notifications every 10 seconds
- Single log file for all activities
- Single camera operation with backup
"""

import cv2
import time
import signal
import sys
import os
import argparse
import threading
import shutil
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict

# Import configuration and core modules
import config
from logger import log_info, log_error, log_warning, log_system_event
from notifications import start_notifications, stop_notifications, notify_system_event, send_critical_notification

# Import detection modules
from motion_detection_module import detect_motion, draw_motion_detections
from face_recognition_module import detect_faces, draw_face_detections
from person_detection_module import detect_persons, draw_person_detections, is_person_model_ready

# Import advanced features
from advanced_ai_module import get_ai_statistics, format_analysis_for_notification
from remote_control_module import start_remote_control, stop_remote_control, is_system_armed, get_system_status

# Import MediaPipe for palm detection
import mediapipe as mp

class UnifiedDetectionSystem:
    """Main controller with conditional motion detection and camera failover"""
    
    def __init__(self, show_preview: bool = True):
        self.show_preview = show_preview
        self.running = False
        self.cap = None
        self.current_camera_id = None
        self.available_cameras = config.CAMERA_INDICES.copy()
        self.cleanup_thread = None
        
        # Palm detection setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=config.PALM_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.PALM_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.palm_frames = 0
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.camera_failures = 0
        
        # Detection state tracking
        self.motion_enabled = False
        self.last_person_detection_time = 0
        self.last_face_detection_time = 0
        
        # Camera health monitoring
        self.consecutive_frame_failures = 0
        self.max_consecutive_failures = 10  # Switch camera after 10 consecutive failures
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_camera(self, camera_id: int) -> bool:
        """Initialize a specific camera"""
        try:
            log_info(f"Attempting to initialize camera {camera_id}", "CAMERA")
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                log_warning(f"Cannot open camera at index {camera_id}", "CAMERA")
                return False
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Test camera by reading a few frames
            for _ in range(5):
                ret, frame = cap.read()
                if not ret or frame is None:
                    log_warning(f"Camera {camera_id} failed frame test", "CAMERA")
                    cap.release()
                    return False
            
            # Success - store the camera
            if self.cap:
                self.cap.release()
            
            self.cap = cap
            self.current_camera_id = camera_id
            self.consecutive_frame_failures = 0
            
            log_info(f"Camera {camera_id} initialized successfully", "CAMERA")
            return True
            
        except Exception as e:
            log_error(f"Error initializing camera {camera_id}: {e}", "CAMERA")
            return False
    
    def _switch_to_backup_camera(self) -> bool:
        """Switch to the next available camera"""
        if not config.CAMERA_FAILOVER_ENABLED:
            log_error("Camera failover disabled - cannot switch cameras", "CAMERA")
            return False
        
        # Find next available camera
        current_index = self.available_cameras.index(self.current_camera_id) if self.current_camera_id in self.available_cameras else -1
        
        for i in range(len(self.available_cameras)):
            next_index = (current_index + 1 + i) % len(self.available_cameras)
            next_camera_id = self.available_cameras[next_index]
            
            if next_camera_id != self.current_camera_id:
                log_info(f"Attempting camera failover: {self.current_camera_id} → {next_camera_id}", "CAMERA")
                
                if self._initialize_camera(next_camera_id):
                    # Send critical notification about camera switch
                    message = f"📹 Camera {self.current_camera_id} down - switched to camera {next_camera_id}"
                    send_critical_notification(message)
                    log_system_event("camera_failover", f"Switched from camera {self.current_camera_id} to {next_camera_id}")
                    self.camera_failures += 1
                    return True
        
        # No backup cameras available
        log_error("No backup cameras available for failover", "CAMERA")
        send_critical_notification("🚨 All cameras down - system cannot continue")
        return False
    
    def initialize(self) -> bool:
        """Initialize the detection system with camera failover"""
        try:
            log_system_event("startup", "Initializing unified detection system with camera failover")
            
            # Validate configuration
            config.validate_config()
            config.ensure_directories()
            
            # Try to initialize primary camera first
            for camera_id in self.available_cameras:
                if self._initialize_camera(camera_id):
                    break
            else:
                log_error("No cameras could be initialized", "SYSTEM")
                return False
            
            # Check if person detection model is ready
            if not is_person_model_ready():
                log_warning("Person detection model not ready - person detection will be disabled", "SYSTEM")
            
            # Start notification system
            start_notifications()
            
            # Start cleanup thread
            self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self.cleanup_thread.start()
            
            log_system_event("startup", f"System initialized successfully with camera {self.current_camera_id}")
            notify_system_event("System started", f"Primary camera: {self.current_camera_id}, Failover: {'enabled' if config.CAMERA_FAILOVER_ENABLED else 'disabled'}")
            
            return True
            
        except Exception as e:
            log_error(f"Failed to initialize system: {e}", "SYSTEM")
            return False
    
    def _read_frame_with_failover(self):
        """Read frame with automatic camera failover"""
        if not self.cap:
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            self.consecutive_frame_failures += 1
            log_warning(f"Frame read failed from camera {self.current_camera_id} (failure #{self.consecutive_frame_failures})", "CAMERA")
            
            # If we've had too many consecutive failures, try to switch cameras
            if self.consecutive_frame_failures >= self.max_consecutive_failures:
                log_error(f"Camera {self.current_camera_id} appears to be disconnected", "CAMERA")
                
                if self._switch_to_backup_camera():
                    # Try reading from new camera
                    return self._read_frame_with_failover()
                else:
                    # No backup available
                    return False, None
            
            return False, None
        
        # Success - reset failure counter
        self.consecutive_frame_failures = 0
        return True, frame
    
    def run(self):
        """Main detection loop with camera failover"""
        if not self.initialize():
            log_error("System initialization failed", "SYSTEM")
            return
        
        self.running = True
        log_info(f"Starting unified detection system with camera failover (primary: {self.current_camera_id})", "SYSTEM")
        
        try:
            while self.running:
                # Read frame with automatic failover
                ret, frame = self._read_frame_with_failover()
                
                if not ret or frame is None:
                    log_error("Failed to read frame from any available camera", "SYSTEM")
                    time.sleep(1)  # Wait before retrying
                    continue
                
                # Resize frame for consistent processing
                frame = cv2.resize(frame, (config.FRAME_WIDTH, 
                                         int(frame.shape[0] * config.FRAME_WIDTH / frame.shape[1])))
                
                self.frame_count += 1
                
                # Check for palm exit gesture
                if self._check_palm_exit(frame):
                    log_info("Palm exit gesture detected - shutting down", "SYSTEM")
                    break
                
                # STEP 1: Always check for persons and faces first
                person_detections = self._detect_persons_with_camera_id(frame, self.current_camera_id)
                face_detections = self._detect_faces_with_camera_id(frame, self.current_camera_id)
                
                # STEP 2: Determine if motion detection should be enabled
                current_time = time.time()
                person_or_face_detected = False
                
                # Check if any person detected (even low confidence)
                if person_detections:
                    person_or_face_detected = True
                    self.last_person_detection_time = current_time
                    self.motion_enabled = True
                    
                    # Send INSTANT notification for person detection
                    for person in person_detections:
                        confidence = person['confidence']
                        log_info(f"INSTANT ALERT: Person detected on camera {self.current_camera_id} (confidence: {confidence:.2f})", "PERSON")
                        notify_system_event("CRITICAL PERSON DETECTED", f"Camera {self.current_camera_id} - Confidence: {confidence:.2f} - Motion tracking enabled")
                
                # Check if any face detected
                if face_detections:
                    person_or_face_detected = True
                    self.last_face_detection_time = current_time
                    self.motion_enabled = True
                    
                    for face in face_detections:
                        name = face['name']
                        confidence = face['confidence']
                        if face['is_known']:
                            log_info(f"Known face detected on camera {self.current_camera_id}: {name} (confidence: {confidence:.1f}%) - Motion tracking enabled", "FACE")
                        else:
                            log_info(f"Unknown face detected on camera {self.current_camera_id} (confidence: {confidence:.1f}%) - Motion tracking enabled", "FACE")
                
                # STEP 3: Only check motion if person/face was detected
                motion_detections = []
                if self.motion_enabled and person_or_face_detected:
                    motion_detections = self._detect_motion_with_camera_id(frame, self.current_camera_id)
                    if motion_detections:
                        log_info(f"Motion detected on camera {self.current_camera_id} while person/face present - {len(motion_detections)} areas", "MOTION")
                        notify_system_event("MOTION WITH PERSON/FACE", f"Camera {self.current_camera_id} - Motion areas: {len(motion_detections)}")
                
                # Disable motion detection after 30 seconds of no person/face detection
                if (current_time - self.last_person_detection_time > 30 and 
                    current_time - self.last_face_detection_time > 30):
                    if self.motion_enabled:
                        self.motion_enabled = False
                        log_info("Motion detection disabled - no person/face for 30 seconds", "SYSTEM")
                
                # Draw all detections on frame
                if motion_detections and self.motion_enabled:
                    frame = draw_motion_detections(frame, motion_detections)
                
                if face_detections:
                    frame = draw_face_detections(frame, face_detections)
                
                if person_detections:
                    frame = draw_person_detections(frame, person_detections)
                
                # Add system information overlay
                frame = self._add_system_overlay(frame, motion_detections, face_detections, person_detections)
                
                # Display frame if preview is enabled
                if self.show_preview:
                    cv2.imshow(config.WINDOW_NAME, frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    # Check for quit key or window close
                    if key == ord('q') or cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        log_info("User requested shutdown", "SYSTEM")
                        break
                
                # Brief pause to prevent excessive CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            log_info("Interrupted by user", "SYSTEM")
        except Exception as e:
            log_error(f"Error in main loop: {e}", "SYSTEM")
        finally:
            self._cleanup()
    
    def _detect_motion_with_camera_id(self, frame: np.ndarray, camera_id: int) -> List[Dict]:
        """Detect motion and add camera_id to results"""
        detections = detect_motion(frame)
        for detection in detections:
            detection['camera_id'] = camera_id
        return detections
    
    def _detect_faces_with_camera_id(self, frame: np.ndarray, camera_id: int) -> List[Dict]:
        """Detect faces and add camera_id to results"""
        detections = detect_faces(frame)
        for detection in detections:
            detection['camera_id'] = camera_id
        return detections
    
    def _detect_persons_with_camera_id(self, frame: np.ndarray, camera_id: int) -> List[Dict]:
        """Detect persons and add camera_id to results"""
        detections = detect_persons(frame)
        for detection in detections:
            detection['camera_id'] = camera_id
        return detections
    
    def _check_palm_exit(self, frame) -> bool:
        """Check for palm exit gesture"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            palm_detected = False
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Check if all fingers are extended (open palm)
                    landmarks = hand_landmarks.landmark
                    fingers_up = []
                    
                    # Check 4 fingers (excluding thumb for robustness)
                    fingers_up.append(landmarks[8].y < landmarks[6].y)   # Index
                    fingers_up.append(landmarks[12].y < landmarks[10].y) # Middle
                    fingers_up.append(landmarks[16].y < landmarks[14].y) # Ring
                    fingers_up.append(landmarks[20].y < landmarks[18].y) # Pinky
                    
                    if all(fingers_up):
                        palm_detected = True
                        # Draw hand landmarks
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        break
            
            if palm_detected:
                self.palm_frames += 1
                progress = (self.palm_frames / config.PALM_HOLD_FRAMES) * 100
                cv2.putText(frame, f"Palm detected - hold to exit ({self.palm_frames}/{config.PALM_HOLD_FRAMES}) {progress:.0f}%", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLORS['palm'], 2)
                
                # Draw progress bar
                bar_width = 200
                bar_height = 10
                bar_x = 10
                bar_y = 70
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress / 100), bar_y + bar_height), config.COLORS['palm'], -1)
                
                if self.palm_frames >= config.PALM_HOLD_FRAMES:
                    notify_system_event("Palm exit", "System terminated by palm gesture")
                    return True
            else:
                self.palm_frames = 0
            
            return False
            
        except Exception as e:
            log_error(f"Error in palm detection: {e}", "SYSTEM")
            return False
    
    def _add_system_overlay(self, frame, motion_detections, face_detections, person_detections):
        """Add system information overlay to frame"""
        try:
            # Calculate FPS
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Create status text
            timestamp = datetime.now().strftime('%H:%M:%S')
            motion_status = "ENABLED" if self.motion_enabled else "DISABLED"
            
            status_lines = [
                f"Time: {timestamp} | FPS: {fps:.1f} | Camera: {self.current_camera_id} | Motion: {motion_status}",
                f"Detections - Motion: {len(motion_detections)} | Faces: {len(face_detections)} | Persons: {len(person_detections)}",
                f"Frame: {self.frame_count} | Runtime: {elapsed_time:.0f}s | Camera failures: {self.camera_failures}"
            ]
            
            # Add detection details
            if face_detections:
                known_faces = [f['name'] for f in face_detections if f['is_known']]
                if known_faces:
                    status_lines.append(f"Known faces: {', '.join(known_faces[:3])}")
            
            if person_detections:
                max_conf = max(p['confidence'] for p in person_detections)
                status_lines.append(f"Person confidence: {max_conf:.2f}")
            
            # Add motion status indicator
            if self.motion_enabled:
                time_since_detection = min(
                    time.time() - self.last_person_detection_time,
                    time.time() - self.last_face_detection_time
                )
                status_lines.append(f"Motion enabled - last detection: {time_since_detection:.1f}s ago")
            
            # Add camera health indicator
            if self.consecutive_frame_failures > 0:
                status_lines.append(f"Camera issues: {self.consecutive_frame_failures} consecutive failures")
            
            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (frame.shape[1] - 5, 25 + len(status_lines) * 25), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Draw status text
            for i, line in enumerate(status_lines):
                color = (255, 255, 255)
                if "Motion: ENABLED" in line:
                    color = (0, 255, 0)  # Green when motion enabled
                elif "Motion: DISABLED" in line:
                    color = (0, 0, 255)  # Red when motion disabled
                elif "Camera issues" in line:
                    color = (0, 165, 255)  # Orange for camera issues
                
                cv2.putText(frame, line, (10, 25 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return frame
            
        except Exception as e:
            log_error(f"Error adding system overlay: {e}", "SYSTEM")
            return frame
    
    def _cleanup_worker(self):
        """Background worker for cleanup tasks"""
        while self.running:
            try:
                # Clean up old snapshots every hour
                self._cleanup_old_snapshots()
                time.sleep(3600)  # 1 hour
            except Exception as e:
                log_error(f"Error in cleanup worker: {e}", "SYSTEM")
                time.sleep(300)  # 5 minutes on error
    
    def _cleanup_old_snapshots(self):
        """Clean up old snapshot files"""
        try:
            cutoff_time = time.time() - (config.SNAPSHOT_CLEANUP_HOURS * 3600)
            snapshots_dir = config.SNAPSHOTS_BASE_DIR
            
            if not os.path.exists(snapshots_dir):
                return
            
            deleted_count = 0
            for root, dirs, files in os.walk(snapshots_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                        deleted_count += 1
                
                # Remove empty directories
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if os.path.isdir(dir_path) and not os.listdir(dir_path):
                        shutil.rmtree(dir_path)
            
            if deleted_count > 0:
                log_info(f"Cleaned up {deleted_count} old snapshot files", "SYSTEM")
                
        except Exception as e:
            log_error(f"Error cleaning up snapshots: {e}", "SYSTEM")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        log_info(f"Received signal {signum} - initiating shutdown", "SYSTEM")
        self.running = False
    
    def _cleanup(self):
        """Clean up resources"""
        try:
            log_system_event("shutdown", "Cleaning up resources")
            
            self.running = False
            
            if self.cap:
                self.cap.release()
                log_info(f"Camera {self.current_camera_id} released", "CAMERA")
            
            if self.show_preview:
                cv2.destroyAllWindows()
            
            # Stop notification system
            stop_notifications()
            
            # Final system log
            runtime = time.time() - self.start_time
            log_system_event("shutdown", f"System stopped after {runtime:.1f} seconds, {self.frame_count} frames processed, {self.camera_failures} camera failures")
            notify_system_event("System stopped", f"Runtime: {runtime:.1f}s, Frames: {self.frame_count}, Camera failures: {self.camera_failures}")
            
        except Exception as e:
            log_error(f"Error during cleanup: {e}", "SYSTEM")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Unified Motion Detection System (Camera Failover)")
    parser.add_argument('--no-preview', action='store_true', 
                       help='Run without preview window (headless mode)')
    parser.add_argument('--test', action='store_true', 
                       help='Run in test mode with additional logging')
    
    args = parser.parse_args()
    
    # Set preview mode - show preview by default unless explicitly disabled
    show_preview = config.SHOW_PREVIEW and not args.no_preview
    
    if args.test:
        log_info("Running in test mode", "SYSTEM")
        notify_system_event("Test mode", "System started in test mode")
    
    # Create and run the detection system
    system = UnifiedDetectionSystem(show_preview=show_preview)
    
    try:
        system.run()
    except Exception as e:
        log_error(f"Fatal error: {e}", "SYSTEM")
        sys.exit(1)

if __name__ == "__main__":
    main()
