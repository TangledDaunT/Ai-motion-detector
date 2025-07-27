"""
Remote Control Module
====================

This module provides remote control functionality for the motion detection system:
- Arm/Disarm detection remotely
- Status monitoring
- Configuration changes
- Remote commands via JSON file

Control file format: /tmp/motion_detection_control.json
{
    "armed": true/false,
    "timestamp": "2025-01-28T10:30:00",
    "command": "arm|disarm|status|restart",
    "settings": {
        "motion_sensitivity": 2500,
        "person_threshold": 0.2
    }
}
"""

import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, Optional, Any
import config
from logger import log_info, log_error, log_warning, log_system_event
from notifications import notify_system_event

class RemoteControlModule:
    """Remote control system for motion detection"""
    
    def __init__(self):
        self.armed = config.DEFAULT_ARMED_STATE
        self.control_file = config.REMOTE_CONTROL_FILE
        self.last_check_time = 0
        self.monitoring_thread = None
        self.running = False
        self.command_handlers = {
            'arm': self._handle_arm_command,
            'disarm': self._handle_disarm_command,
            'status': self._handle_status_command,
            'restart': self._handle_restart_command,
            'settings': self._handle_settings_command
        }
        
        # Initialize control file
        self._initialize_control_file()
    
    def _initialize_control_file(self):
        """Initialize the remote control file"""
        try:
            initial_state = {
                "armed": self.armed,
                "timestamp": datetime.now().isoformat(),
                "status": "initialized",
                "system_info": {
                    "version": "2.0",
                    "features": ["camera_failover", "ai_analysis", "smart_filtering"],
                    "cameras": config.CAMERA_INDICES
                }
            }
            
            with open(self.control_file, 'w') as f:
                json.dump(initial_state, f, indent=2)
            
            log_info(f"Remote control initialized - Armed: {self.armed}", "REMOTE")
            
        except Exception as e:
            log_error(f"Failed to initialize remote control file: {e}", "REMOTE")
    
    def start_monitoring(self):
        """Start monitoring the control file for commands"""
        if not config.REMOTE_CONTROL_ENABLED:
            log_info("Remote control disabled in configuration", "REMOTE")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitor_control_file, daemon=True)
        self.monitoring_thread.start()
        log_info("Remote control monitoring started", "REMOTE")
    
    def stop_monitoring(self):
        """Stop monitoring the control file"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        log_info("Remote control monitoring stopped", "REMOTE")
    
    def _monitor_control_file(self):
        """Monitor the control file for changes"""
        while self.running:
            try:
                if os.path.exists(self.control_file):
                    file_mtime = os.path.getmtime(self.control_file)
                    
                    if file_mtime > self.last_check_time:
                        self._process_control_file()
                        self.last_check_time = file_mtime
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                log_error(f"Error monitoring control file: {e}", "REMOTE")
                time.sleep(5)  # Wait longer on error
    
    def _process_control_file(self):
        """Process commands from the control file"""
        try:
            with open(self.control_file, 'r') as f:
                data = json.load(f)
            
            # Process command if present
            command = data.get('command')
            if command and command in self.command_handlers:
                log_info(f"Processing remote command: {command}", "REMOTE")
                self.command_handlers[command](data)
                
                # Clear the command after processing
                data['command'] = None
                data['last_processed'] = datetime.now().isoformat()
                
                with open(self.control_file, 'w') as f:
                    json.dump(data, f, indent=2)
            
            # Update armed state if changed
            new_armed_state = data.get('armed', self.armed)
            if new_armed_state != self.armed:
                self._set_armed_state(new_armed_state, remote=True)
                
        except Exception as e:
            log_error(f"Error processing control file: {e}", "REMOTE")
    
    def _handle_arm_command(self, data: Dict):
        """Handle arm command"""
        self._set_armed_state(True, remote=True)
        notify_system_event("Remote Control", "System ARMED remotely")
        log_system_event("remote_arm", "System armed via remote control")
    
    def _handle_disarm_command(self, data: Dict):
        """Handle disarm command"""
        self._set_armed_state(False, remote=True)
        notify_system_event("Remote Control", "System DISARMED remotely")
        log_system_event("remote_disarm", "System disarmed via remote control")
    
    def _handle_status_command(self, data: Dict):
        """Handle status request command"""
        status = self.get_system_status()
        
        # Update control file with current status
        data['system_status'] = status
        data['status_updated'] = datetime.now().isoformat()
        
        with open(self.control_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        log_info("System status updated in control file", "REMOTE")
    
    def _handle_restart_command(self, data: Dict):
        """Handle restart command"""
        log_system_event("remote_restart", "System restart requested via remote control")
        notify_system_event("Remote Control", "System restart requested - please restart manually")
        # Note: Actual restart would need to be handled by the main application
    
    def _handle_settings_command(self, data: Dict):
        """Handle settings update command"""
        settings = data.get('settings', {})
        
        for key, value in settings.items():
            if hasattr(config, key.upper()):
                old_value = getattr(config, key.upper())
                setattr(config, key.upper(), value)
                log_info(f"Remote setting update: {key.upper()} = {value} (was {old_value})", "REMOTE")
        
        notify_system_event("Remote Control", f"Settings updated: {list(settings.keys())}")
        log_system_event("remote_settings", f"Settings updated via remote control: {settings}")
    
    def _set_armed_state(self, armed: bool, remote: bool = False):
        """Set the armed state of the system"""
        if self.armed != armed:
            self.armed = armed
            state_text = "ARMED" if armed else "DISARMED"
            source_text = "remotely" if remote else "locally"
            
            log_info(f"System {state_text} {source_text}", "REMOTE")
            
            # Update control file
            try:
                if os.path.exists(self.control_file):
                    with open(self.control_file, 'r') as f:
                        data = json.load(f)
                else:
                    data = {}
                
                data['armed'] = armed
                data['timestamp'] = datetime.now().isoformat()
                data['changed_by'] = 'remote' if remote else 'local'
                
                with open(self.control_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            except Exception as e:
                log_error(f"Failed to update control file: {e}", "REMOTE")
    
    def is_armed(self) -> bool:
        """Check if the system is currently armed"""
        return self.armed
    
    def arm_system(self):
        """Arm the system locally"""
        self._set_armed_state(True, remote=False)
    
    def disarm_system(self):
        """Disarm the system locally"""
        self._set_armed_state(False, remote=False)
    
    def toggle_armed_state(self):
        """Toggle the armed state"""
        self._set_armed_state(not self.armed, remote=False)
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            "armed": self.armed,
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time(),
            "remote_control_enabled": config.REMOTE_CONTROL_ENABLED,
            "features": {
                "camera_failover": config.CAMERA_FAILOVER_ENABLED,
                "age_gender_detection": config.AGE_GENDER_DETECTION_ENABLED,
                "emotion_detection": config.EMOTION_DETECTION_ENABLED,
                "smart_filtering": config.SMART_FILTERING_ENABLED
            },
            "settings": {
                "motion_sensitivity": config.MOTION_SENSITIVITY,
                "person_threshold": config.PERSON_CONFIDENCE_THRESHOLD,
                "face_threshold": config.FACE_RECOGNITION_THRESHOLD
            },
            "cameras": config.CAMERA_INDICES
        }
    
    def create_remote_command(self, command: str, **kwargs) -> Dict:
        """Create a remote command dictionary"""
        cmd_data = {
            "command": command,
            "timestamp": datetime.now().isoformat(),
            "armed": kwargs.get('armed', self.armed)
        }
        
        if kwargs:
            cmd_data.update(kwargs)
        
        return cmd_data
    
    def send_remote_command(self, command: str, **kwargs):
        """Send a remote command by updating the control file"""
        try:
            cmd_data = self.create_remote_command(command, **kwargs)
            
            with open(self.control_file, 'w') as f:
                json.dump(cmd_data, f, indent=2)
            
            log_info(f"Remote command sent: {command}", "REMOTE")
            
        except Exception as e:
            log_error(f"Failed to send remote command: {e}", "REMOTE")
    
    def get_control_file_path(self) -> str:
        """Get the path to the control file"""
        return self.control_file
    
    def cleanup(self):
        """Clean up remote control resources"""
        self.stop_monitoring()
        
        # Update control file with shutdown status
        try:
            shutdown_data = {
                "armed": False,
                "timestamp": datetime.now().isoformat(),
                "status": "shutdown",
                "last_seen": datetime.now().isoformat()
            }
            
            with open(self.control_file, 'w') as f:
                json.dump(shutdown_data, f, indent=2)
                
        except Exception as e:
            log_error(f"Error updating control file during cleanup: {e}", "REMOTE")

# Global remote control instance
remote_control = RemoteControlModule()

# Convenience functions
def start_remote_control():
    """Start remote control monitoring"""
    remote_control.start_monitoring()

def stop_remote_control():
    """Stop remote control monitoring"""
    remote_control.stop_monitoring()

def is_system_armed() -> bool:
    """Check if system is armed"""
    return remote_control.is_armed()

def arm_system():
    """Arm the system"""
    remote_control.arm_system()

def disarm_system():
    """Disarm the system"""
    remote_control.disarm_system()

def toggle_system():
    """Toggle armed state"""
    remote_control.toggle_armed_state()

def get_system_status() -> Dict:
    """Get system status"""
    return remote_control.get_system_status()

def send_remote_command(command: str, **kwargs):
    """Send a remote command"""
    remote_control.send_remote_command(command, **kwargs)

def get_control_file_path() -> str:
    """Get control file path"""
    return remote_control.get_control_file_path()