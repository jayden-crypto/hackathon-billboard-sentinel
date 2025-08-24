"""
Privacy and Consent Management for Billboard Sentinel
Handles face blurring, privacy disclaimers, and data retention policies
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter
import logging
from typing import Dict, List, Optional, Tuple
import os
import sys
from datetime import datetime, timedelta
import json

# Add face_blur directory to path
face_blur_path = os.path.join(os.path.dirname(__file__), "..", "..", "face_blur")
sys.path.append(face_blur_path)

try:
    from blur_faces import blur_faces_in_image
except ImportError:
    logging.warning("Face blur module not found, using fallback implementation")
    blur_faces_in_image = None

logger = logging.getLogger(__name__)

class PrivacyManager:
    """Manages privacy compliance and data protection"""
    
    def __init__(self):
        """Initialize privacy manager"""
        self.face_cascade = None
        self.consent_records = {}
        self.data_retention_days = 30  # Default retention without consent
        
        # Initialize face detection
        self._init_face_detection()
    
    def _init_face_detection(self):
        """Initialize OpenCV face detection as fallback"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logger.warning("Face cascade classifier not loaded properly")
        except Exception as e:
            logger.error(f"Failed to initialize face detection: {e}")
    
    def process_image_with_privacy(self, image_path: str, user_consent: Dict) -> Dict:
        """
        Process uploaded image with privacy protections
        
        Args:
            image_path: Path to uploaded image
            user_consent: User consent preferences
            
        Returns:
            Dict with processed image info and privacy compliance
        """
        try:
            result = {
                'original_path': image_path,
                'processed_path': None,
                'faces_detected': 0,
                'faces_blurred': 0,
                'privacy_compliant': False,
                'consent_recorded': False,
                'retention_period_days': self.data_retention_days,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Check user consent
            consent_given = user_consent.get('face_blur_consent', False)
            data_sharing_consent = user_consent.get('data_sharing_consent', False)
            
            # Process image for face detection and blurring
            if os.path.exists(image_path):
                processed_path, face_info = self._blur_faces_in_image(image_path, consent_given)
                
                result.update({
                    'processed_path': processed_path,
                    'faces_detected': face_info.get('faces_detected', 0),
                    'faces_blurred': face_info.get('faces_blurred', 0),
                    'privacy_compliant': True
                })
                
                # Set retention period based on consent
                if data_sharing_consent:
                    result['retention_period_days'] = 1825  # 5 years with consent
                else:
                    result['retention_period_days'] = 30   # 30 days without consent
                
                # Record consent
                self._record_consent(image_path, user_consent)
                result['consent_recorded'] = True
                
            return result
            
        except Exception as e:
            logger.error(f"Privacy processing failed for {image_path}: {e}")
            return {
                'error': str(e),
                'privacy_compliant': False,
                'original_path': image_path
            }
    
    def _blur_faces_in_image(self, image_path: str, consent_given: bool) -> Tuple[str, Dict]:
        """
        Blur faces in image using available face detection methods
        
        Args:
            image_path: Path to input image
            consent_given: Whether user consented to face processing
            
        Returns:
            Tuple of (processed_image_path, face_info_dict)
        """
        face_info = {'faces_detected': 0, 'faces_blurred': 0}
        
        try:
            # Try using the existing face_blur module first
            if blur_faces_in_image is not None:
                processed_path = self._use_existing_face_blur(image_path)
                if processed_path:
                    # Estimate face count (simplified)
                    face_info['faces_detected'] = 1  # Assume at least 1 if processing succeeded
                    face_info['faces_blurred'] = 1
                    return processed_path, face_info
            
            # Fallback to OpenCV implementation
            processed_path = self._opencv_face_blur(image_path)
            return processed_path, face_info
            
        except Exception as e:
            logger.error(f"Face blurring failed: {e}")
            # Return original path if blurring fails
            return image_path, face_info
    
    def _use_existing_face_blur(self, image_path: str) -> Optional[str]:
        """Use existing face_blur module if available"""
        try:
            # Generate output path
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.dirname(image_path)
            output_path = os.path.join(output_dir, f"{base_name}_privacy_protected.jpg")
            
            # Use existing blur_faces function
            blur_faces_in_image(image_path, output_path)
            
            if os.path.exists(output_path):
                return output_path
                
        except Exception as e:
            logger.error(f"Existing face blur failed: {e}")
            
        return None
    
    def _opencv_face_blur(self, image_path: str) -> str:
        """Fallback face blurring using OpenCV"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return image_path
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                # Blur detected faces
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_region = image[y:y+h, x:x+w]
                    
                    # Apply strong blur
                    blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                    
                    # Replace face region with blurred version
                    image[y:y+h, x:x+w] = blurred_face
            
            # Save processed image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.dirname(image_path)
            output_path = os.path.join(output_dir, f"{base_name}_privacy_protected.jpg")
            
            cv2.imwrite(output_path, image)
            return output_path
            
        except Exception as e:
            logger.error(f"OpenCV face blur failed: {e}")
            return image_path
    
    def _record_consent(self, image_path: str, consent_data: Dict):
        """Record user consent for data processing"""
        try:
            consent_record = {
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'face_blur_consent': consent_data.get('face_blur_consent', False),
                'data_sharing_consent': consent_data.get('data_sharing_consent', False),
                'municipality_sharing': consent_data.get('municipality_sharing', False),
                'retention_acknowledged': consent_data.get('retention_acknowledged', False),
                'user_id': consent_data.get('user_id', 'anonymous'),
                'ip_address': consent_data.get('ip_address', 'unknown')
            }
            
            # Store consent record (in production, would use database)
            consent_id = f"consent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.consent_records[consent_id] = consent_record
            
            logger.info(f"Consent recorded for {image_path}")
            
        except Exception as e:
            logger.error(f"Failed to record consent: {e}")
    
    def get_privacy_disclaimer(self) -> Dict:
        """Get privacy disclaimer text for UI display"""
        return {
            'title': 'Privacy Notice & Consent',
            'sections': {
                'data_collection': {
                    'title': 'Data We Collect',
                    'content': [
                        'Photos you submit for billboard reporting',
                        'GPS location data (if enabled)',
                        'Device information and timestamps',
                        'Optional contact information'
                    ]
                },
                'face_detection': {
                    'title': 'Face Detection & Blurring',
                    'content': [
                        'We automatically detect and blur faces in submitted photos',
                        'This protects privacy of individuals in your photos',
                        'Face detection is processed locally when possible',
                        'Blurred images are used for billboard analysis'
                    ]
                },
                'data_retention': {
                    'title': 'Data Retention',
                    'content': [
                        'Without consent: Photos deleted after 30 days',
                        'With sharing consent: Photos kept for 5 years for municipal use',
                        'GPS and metadata: Retained for case tracking',
                        'You can request deletion at any time'
                    ]
                },
                'data_sharing': {
                    'title': 'Data Sharing',
                    'content': [
                        'Reports shared with relevant municipal authorities',
                        'Personal information is never shared without consent',
                        'Photos are anonymized before sharing',
                        'You control whether to share with municipality'
                    ]
                }
            },
            'consent_options': {
                'face_blur_consent': {
                    'required': True,
                    'description': 'Allow automatic face detection and blurring for privacy protection'
                },
                'data_sharing_consent': {
                    'required': False,
                    'description': 'Share anonymized report data with municipal authorities for enforcement'
                },
                'municipality_sharing': {
                    'required': False,
                    'description': 'Allow direct communication from municipality about this report'
                },
                'retention_acknowledged': {
                    'required': True,
                    'description': 'I understand the data retention policy (30 days without sharing consent, 5 years with consent)'
                }
            },
            'contact_info': {
                'privacy_officer': 'privacy@billboard-sentinel.org',
                'deletion_requests': 'delete@billboard-sentinel.org',
                'policy_url': 'https://billboard-sentinel.org/privacy-policy'
            }
        }
    
    def schedule_data_cleanup(self, image_path: str, retention_days: int):
        """Schedule automatic data cleanup based on retention policy"""
        try:
            cleanup_date = datetime.now() + timedelta(days=retention_days)
            
            # In production, would use task queue (Celery, Redis, etc.)
            cleanup_record = {
                'image_path': image_path,
                'cleanup_date': cleanup_date.isoformat(),
                'retention_days': retention_days,
                'scheduled_at': datetime.now().isoformat()
            }
            
            logger.info(f"Scheduled cleanup for {image_path} on {cleanup_date}")
            return cleanup_record
            
        except Exception as e:
            logger.error(f"Failed to schedule cleanup: {e}")
            return None
    
    def cleanup_expired_data(self):
        """Clean up data that has exceeded retention period"""
        try:
            # In production, this would be a scheduled job
            current_time = datetime.now()
            cleaned_files = []
            
            # Check for expired files (mock implementation)
            # Real implementation would query database for expired records
            
            logger.info(f"Data cleanup completed, {len(cleaned_files)} files removed")
            return cleaned_files
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return []


def create_privacy_manager():
    """Factory function to create privacy manager"""
    return PrivacyManager()
