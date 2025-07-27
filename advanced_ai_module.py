"""
Advanced AI Module
==================

This module provides advanced AI features including:
- Age and gender detection
- Emotion detection
- Smart filtering for false positive reduction
- AI-powered analysis of detected persons

Dependencies:
    pip install deepface tensorflow
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from collections import deque
import threading
import config
from logger import log_info, log_error, log_warning

# Try to import DeepFace for advanced AI features
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    log_info("DeepFace library loaded successfully", "AI")
except ImportError:
    DEEPFACE_AVAILABLE = False
    log_warning("DeepFace not available - advanced AI features disabled. Install with: pip install deepface tensorflow", "AI")

class AdvancedAIModule:
    """Advanced AI analysis module"""
    
    def __init__(self):
        self.detection_history = deque(maxlen=config.DETECTION_HISTORY_SIZE)
        self.analysis_cache = {}
        self.cache_lock = threading.Lock()
        self.last_analysis_time = {}
        
        # Initialize models if available
        if DEEPFACE_AVAILABLE and config.AGE_GENDER_DETECTION_ENABLED:
            self._warmup_models()
    
    def _warmup_models(self):
        """Warm up AI models by running a test analysis"""
        try:
            log_info("Warming up AI models...", "AI")
            # Create a dummy image for model warmup
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            dummy_image.fill(128)  # Gray image
            
            # This will download and initialize the models
            DeepFace.analyze(dummy_image, actions=['age', 'gender', 'emotion'], enforce_detection=False)
            log_info("AI models warmed up successfully", "AI")
        except Exception as e:
            log_error(f"Error warming up AI models: {e}", "AI")
    
    def analyze_person(self, face_image: np.ndarray, person_id: str = None) -> Dict:
        """
        Perform comprehensive AI analysis on a detected person
        
        Args:
            face_image: Cropped face image for analysis
            person_id: Unique identifier for caching
            
        Returns:
            Dictionary containing age, gender, emotion, and confidence scores
        """
        if not DEEPFACE_AVAILABLE:
            return self._get_default_analysis()
        
        # Check cache first
        if person_id and person_id in self.analysis_cache:
            cache_time = self.last_analysis_time.get(person_id, 0)
            if time.time() - cache_time < 5:  # Cache for 5 seconds
                return self.analysis_cache[person_id]
        
        try:
            # Ensure image is in correct format
            if face_image.shape[0] < 50 or face_image.shape[1] < 50:
                # Image too small for reliable analysis
                return self._get_default_analysis()
            
            # Resize image for better analysis
            face_image = cv2.resize(face_image, (224, 224))
            
            # Perform DeepFace analysis
            actions = []
            if config.AGE_GENDER_DETECTION_ENABLED:
                actions.extend(['age', 'gender'])
            if config.EMOTION_DETECTION_ENABLED:
                actions.append('emotion')
            
            if not actions:
                return self._get_default_analysis()
            
            result = DeepFace.analyze(face_image, actions=actions, enforce_detection=False)
            
            # Extract results
            analysis = {
                'age': result.get('age', 'Unknown'),
                'gender': result.get('dominant_gender', 'Unknown'),
                'gender_confidence': result.get('gender', {}).get(result.get('dominant_gender', ''), 0),
                'emotion': result.get('dominant_emotion', 'Unknown'),
                'emotion_confidence': result.get('emotion', {}).get(result.get('dominant_emotion', ''), 0),
                'emotion_scores': result.get('emotion', {}),
                'analysis_time': time.time(),
                'valid': True
            }
            
            # Cache the result
            if person_id:
                with self.cache_lock:
                    self.analysis_cache[person_id] = analysis
                    self.last_analysis_time[person_id] = time.time()
            
            return analysis
            
        except Exception as e:
            log_error(f"Error in AI analysis: {e}", "AI")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict:
        """Return default analysis when AI is not available"""
        return {
            'age': 'Unknown',
            'gender': 'Unknown',
            'gender_confidence': 0,
            'emotion': 'Unknown',
            'emotion_confidence': 0,
            'emotion_scores': {},
            'analysis_time': time.time(),
            'valid': False
        }
    
    def apply_smart_filtering(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply smart filtering to reduce false positives
        
        Args:
            detections: List of detection results
            
        Returns:
            Filtered list of detections
        """
        if not config.SMART_FILTERING_ENABLED:
            return detections
        
        filtered_detections = []
        
        for detection in detections:
            if self._is_valid_detection(detection):
                filtered_detections.append(detection)
            else:
                log_info(f"Smart filter removed false positive: {detection.get('type', 'unknown')} with confidence {detection.get('confidence', 0):.2f}", "AI")
        
        # Update detection history
        self.detection_history.extend(filtered_detections)
        
        return filtered_detections
    
    def _is_valid_detection(self, detection: Dict) -> bool:
        """
        Determine if a detection is valid or a false positive
        
        Args:
            detection: Detection result dictionary
            
        Returns:
            True if detection is valid, False if likely false positive
        """
        try:
            confidence = detection.get('confidence', 0)
            detection_type = detection.get('type', 'unknown')
            
            # Basic confidence threshold
            if confidence < config.FALSE_POSITIVE_THRESHOLD:
                return False
            
            # Check consistency with recent detections
            if len(self.detection_history) >= 3:
                recent_detections = list(self.detection_history)[-3:]
                similar_detections = [
                    d for d in recent_detections 
                    if d.get('type') == detection_type and 
                    abs(d.get('confidence', 0) - confidence) < 0.3
                ]
                
                consistency_ratio = len(similar_detections) / len(recent_detections)
                if consistency_ratio < config.CONSISTENCY_THRESHOLD:
                    return False
            
            # Additional filtering based on detection type
            if detection_type == 'person':
                return self._validate_person_detection(detection)
            elif detection_type == 'face':
                return self._validate_face_detection(detection)
            
            return True
            
        except Exception as e:
            log_error(f"Error in smart filtering: {e}", "AI")
            return True  # Default to accepting detection if filtering fails
    
    def _validate_person_detection(self, detection: Dict) -> bool:
        """Validate person detection specifically"""
        confidence = detection.get('confidence', 0)
        bbox = detection.get('bbox', [0, 0, 0, 0])
        
        # Check bounding box size (too small might be false positive)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        if area < 1000:  # Very small detection
            return confidence > 0.7  # Require higher confidence
        
        return True
    
    def _validate_face_detection(self, detection: Dict) -> bool:
        """Validate face detection specifically"""
        confidence = detection.get('confidence', 0)
        
        # Face detections are generally more reliable
        return confidence > 0.3
    
    def get_emotion_summary(self, emotion_scores: Dict) -> str:
        """
        Generate a human-readable emotion summary
        
        Args:
            emotion_scores: Dictionary of emotion scores
            
        Returns:
            Formatted emotion summary string
        """
        if not emotion_scores:
            return "Unknown"
        
        # Sort emotions by confidence
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_emotions) >= 2:
            primary = sorted_emotions[0]
            secondary = sorted_emotions[1]
            return f"{primary[0]} ({primary[1]:.1f}%), {secondary[0]} ({secondary[1]:.1f}%)"
        elif len(sorted_emotions) == 1:
            emotion = sorted_emotions[0]
            return f"{emotion[0]} ({emotion[1]:.1f}%)"
        
        return "Unknown"
    
    def format_analysis_for_display(self, analysis: Dict) -> str:
        """
        Format AI analysis results for display
        
        Args:
            analysis: Analysis results dictionary
            
        Returns:
            Formatted string for display
        """
        if not analysis.get('valid', False):
            return "AI: Not available"
        
        parts = []
        
        # Age and gender
        if analysis.get('age') != 'Unknown' and analysis.get('gender') != 'Unknown':
            gender_conf = analysis.get('gender_confidence', 0)
            parts.append(f"{analysis['gender']} ({gender_conf:.0f}%), Age: {analysis['age']}")
        
        # Emotion
        if analysis.get('emotion') != 'Unknown':
            emotion_conf = analysis.get('emotion_confidence', 0)
            parts.append(f"Emotion: {analysis['emotion']} ({emotion_conf:.0f}%)")
        
        return " | ".join(parts) if parts else "AI: Processing..."
    
    def format_analysis_for_notification(self, analysis: Dict) -> str:
        """
        Format AI analysis results for notifications
        
        Args:
            analysis: Analysis results dictionary
            
        Returns:
            Formatted string for notifications
        """
        if not analysis.get('valid', False):
            return ""
        
        parts = []
        
        # Gender and age
        if analysis.get('gender') != 'Unknown':
            parts.append(f"{analysis['gender']}")
        if analysis.get('age') != 'Unknown':
            parts.append(f"age {analysis['age']}")
        
        # Emotion if significant
        emotion_conf = analysis.get('emotion_confidence', 0)
        if analysis.get('emotion') != 'Unknown' and emotion_conf > 60:
            parts.append(f"appears {analysis['emotion'].lower()}")
        
        return " - " + ", ".join(parts) if parts else ""
    
    def get_statistics(self) -> Dict:
        """Get AI module statistics"""
        return {
            'deepface_available': DEEPFACE_AVAILABLE,
            'age_gender_enabled': config.AGE_GENDER_DETECTION_ENABLED,
            'emotion_enabled': config.EMOTION_DETECTION_ENABLED,
            'smart_filtering_enabled': config.SMART_FILTERING_ENABLED,
            'cache_size': len(self.analysis_cache),
            'detection_history_size': len(self.detection_history),
            'total_analyses': len(self.last_analysis_time)
        }
    
    def clear_cache(self):
        """Clear analysis cache"""
        with self.cache_lock:
            self.analysis_cache.clear()
            self.last_analysis_time.clear()
        log_info("AI analysis cache cleared", "AI")

# Global AI module instance
ai_module = AdvancedAIModule()

# Convenience functions
def analyze_person(face_image: np.ndarray, person_id: str = None) -> Dict:
    """Analyze a person's face for age, gender, and emotion"""
    return ai_module.analyze_person(face_image, person_id)

def apply_smart_filtering(detections: List[Dict]) -> List[Dict]:
    """Apply smart filtering to reduce false positives"""
    return ai_module.apply_smart_filtering(detections)

def format_analysis_for_display(analysis: Dict) -> str:
    """Format analysis for display"""
    return ai_module.format_analysis_for_display(analysis)

def format_analysis_for_notification(analysis: Dict) -> str:
    """Format analysis for notifications"""
    return ai_module.format_analysis_for_notification(analysis)

def get_ai_statistics() -> Dict:
    """Get AI module statistics"""
    return ai_module.get_statistics()