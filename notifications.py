"""
Unified Notification System for Motion Detection
==============================================

This module provides a centralized notification system that consolidates all
detection events and sends a single Pushover notification every 10 seconds
with a summary of all activity.
"""

import time
import threading
import requests
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict, deque
import config
from logger import log_info, log_error, log_warning

class NotificationManager:
    """Manages unified notifications with cooldown and event aggregation"""
    
    def __init__(self):
        self.events_queue = deque()
        self.events_lock = threading.Lock()
        self.last_notification_time = 0
        self.notification_thread = None
        self.running = False
        self.event_counts = defaultdict(int)
        self.latest_events = {}
        
    def start(self):
        """Start the notification manager"""
        if not self.running:
            self.running = True
            self.notification_thread = threading.Thread(target=self._notification_worker, daemon=True)
            self.notification_thread.start()
            log_info("Notification manager started", "NOTIFICATIONS")
    
    def stop(self):
        """Stop the notification manager"""
        self.running = False
        if self.notification_thread:
            self.notification_thread.join(timeout=2)
        log_info("Notification manager stopped", "NOTIFICATIONS")
    
    def add_event(self, event_type: str, details: Dict, priority: str = "normal"):
        """
        Add an event to the notification queue
        
        Args:
            event_type: Type of event (motion, face_known, face_unknown, person, palm, system)
            details: Event details dictionary
            priority: Event priority (normal, high, critical)
        """
        timestamp = datetime.now()
        event = {
            'type': event_type,
            'details': details,
            'priority': priority,
            'timestamp': timestamp
        }
        
        with self.events_lock:
            self.events_queue.append(event)
            self.event_counts[event_type] += 1
            self.latest_events[event_type] = event
            
            # Keep queue size manageable
            if len(self.events_queue) > 100:
                self.events_queue.popleft()
    
    def _notification_worker(self):
        """Background worker that sends periodic notifications"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time to send a notification
                if current_time - self.last_notification_time >= config.UNIFIED_NOTIFY_COOLDOWN:
                    self._send_unified_notification()
                    self.last_notification_time = current_time
                
                # Sleep for a short interval
                time.sleep(1)
                
            except Exception as e:
                log_error(f"Error in notification worker: {e}", "NOTIFICATIONS")
                time.sleep(5)  # Wait before retrying
    
    def _send_unified_notification(self):
        """Send a unified notification with summary of recent events"""
        with self.events_lock:
            if not self.events_queue:
                return  # No events to report
            
            # Get events from the last notification period
            cutoff_time = datetime.now().timestamp() - config.UNIFIED_NOTIFY_COOLDOWN
            recent_events = [
                event for event in self.events_queue 
                if event['timestamp'].timestamp() > cutoff_time
            ]
            
            if not recent_events:
                return
            
            # Create summary message
            message = self._create_summary_message(recent_events)
            
            # Send the notification
            success = self._send_pushover_notification(message)
            
            if success:
                log_info(f"Unified notification sent: {len(recent_events)} events", "NOTIFICATIONS")
                # Clear processed events
                self._clear_old_events(cutoff_time)
            else:
                log_warning("Failed to send unified notification", "NOTIFICATIONS")
    
    def _create_summary_message(self, events: List[Dict]) -> str:
        """Create a summary message from recent events"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Count events by type
        event_summary = defaultdict(int)
        critical_events = []
        face_detections = []
        
        for event in events:
            event_type = event['type']
            event_summary[event_type] += 1
            
            # Collect critical events
            if event['priority'] == 'critical' or event_type == 'person':
                critical_events.append(event)
            
            # Collect face detections
            if event_type == 'face_known':
                name = event['details'].get('name', 'Unknown')
                if name not in [f['name'] for f in face_detections]:
                    face_detections.append({'name': name, 'confidence': event['details'].get('confidence', 0)})
        
        # Build message
        message_parts = [f"🕐 {timestamp} Detection Summary:"]
        
        # Add critical alerts first
        if critical_events:
            message_parts.append("🚨 CRITICAL:")
            for event in critical_events[:3]:  # Limit to 3 critical events
                if event['type'] == 'person':
                    conf = event['details'].get('confidence', 0)
                    message_parts.append(f"  • Person detected ({conf:.1f}%)")
        
        # Add face detections
        if face_detections:
            message_parts.append("👤 Faces:")
            for face in face_detections[:3]:  # Limit to 3 faces
                message_parts.append(f"  • {face['name']} ({face['confidence']:.1f}%)")
        
        # Add other events summary
        other_events = []
        if event_summary['motion'] > 0:
            other_events.append(f"Motion: {event_summary['motion']}")
        if event_summary['face_unknown'] > 0:
            other_events.append(f"Unknown faces: {event_summary['face_unknown']}")
        if event_summary['palm'] > 0:
            other_events.append(f"Palm: {event_summary['palm']}")
        
        if other_events:
            message_parts.append("📊 Activity: " + ", ".join(other_events))
        
        # Add system events if any
        if event_summary['system'] > 0:
            message_parts.append(f"⚙️ System events: {event_summary['system']}")
        
        return "\n".join(message_parts)
    
    def _send_pushover_notification(self, message: str) -> bool:
        """Send notification via Pushover API"""
        try:
            data = {
                "token": config.API_TOKEN,
                "user": config.USER_KEY,
                "message": message,
                "title": "Motion Detection System"
            }
            
            response = requests.post(
                "https://api.pushover.net/1/messages.json", 
                data=data,
                timeout=10
            )
            
            if response.status_code == 200:
                return True
            else:
                log_error(f"Pushover API error: {response.status_code} - {response.text}", "NOTIFICATIONS")
                return False
                
        except requests.exceptions.RequestException as e:
            log_error(f"Network error sending notification: {e}", "NOTIFICATIONS")
            return False
        except Exception as e:
            log_error(f"Unexpected error sending notification: {e}", "NOTIFICATIONS")
            return False
    
    def _clear_old_events(self, cutoff_timestamp: float):
        """Remove old events from the queue"""
        # Keep only recent events for next cycle
        self.events_queue = deque([
            event for event in self.events_queue 
            if event['timestamp'].timestamp() > cutoff_timestamp - 30  # Keep 30 seconds of history
        ])
        
        # Reset counters
        self.event_counts.clear()
    
    def send_immediate_notification(self, message: str, priority: str = "normal"):
        """
        Send an immediate notification (bypasses cooldown for critical events)
        
        Args:
            message: Message to send
            priority: Priority level (normal, high, critical)
        """
        if priority == "critical":
            success = self._send_pushover_notification(f"🚨 CRITICAL: {message}")
            if success:
                log_info(f"Immediate critical notification sent: {message}", "NOTIFICATIONS")
            return success
        else:
            # Add to queue for next unified notification
            self.add_event("system", {"message": message}, priority)
            return True
    
    def get_status(self) -> Dict:
        """Get current status of notification manager"""
        with self.events_lock:
            return {
                "running": self.running,
                "queue_size": len(self.events_queue),
                "last_notification": self.last_notification_time,
                "event_counts": dict(self.event_counts),
                "time_until_next": max(0, config.UNIFIED_NOTIFY_COOLDOWN - (time.time() - self.last_notification_time))
            }

# Global notification manager instance
notification_manager = NotificationManager()

# Convenience functions for easy importing
def start_notifications():
    """Start the notification system"""
    notification_manager.start()

def stop_notifications():
    """Stop the notification system"""
    notification_manager.stop()

def notify_motion_detected(contour_area: int, sensitivity: int):
    """Notify about motion detection"""
    notification_manager.add_event("motion", {
        "contour_area": contour_area,
        "sensitivity": sensitivity
    })

def notify_face_detected(name: str, confidence: float, is_known: bool = True):
    """Notify about face detection"""
    event_type = "face_known" if is_known else "face_unknown"
    notification_manager.add_event(event_type, {
        "name": name,
        "confidence": confidence
    })

def notify_person_detected(confidence: float, bbox: List[int]):
    """Notify about person detection (critical priority)"""
    notification_manager.add_event("person", {
        "confidence": confidence,
        "bbox": bbox
    }, priority="critical")

def notify_palm_detected(frames_held: int):
    """Notify about palm detection"""
    notification_manager.add_event("palm", {
        "frames_held": frames_held
    })

def notify_system_event(event: str, details: Optional[str] = None):
    """Notify about system events"""
    notification_manager.add_event("system", {
        "event": event,
        "details": details
    })

def send_critical_notification(message: str):
    """Send an immediate critical notification"""
    return notification_manager.send_immediate_notification(message, "critical")

def get_notification_status():
    """Get current notification system status"""
    return notification_manager.get_status()