"""
Security middleware and input validation for Billboard Sentinel
Implements rate limiting, input validation, and file upload security
"""

import time
import hashlib
import mimetypes
from typing import Dict, List, Optional, Any
from fastapi import HTTPException, Request, UploadFile, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
import os
import magic
from PIL import Image
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self):
        """Initialize rate limiter"""
        self.clients = defaultdict(lambda: {
            'tokens': 100,  # Initial tokens
            'last_refill': time.time(),
            'requests': deque()  # Track request timestamps
        })
        
        # Rate limit configurations
        self.limits = {
            'default': {'requests': 100, 'window': 3600},  # 100 requests per hour
            'upload': {'requests': 10, 'window': 3600},    # 10 uploads per hour
            'auth': {'requests': 5, 'window': 300},        # 5 auth attempts per 5 minutes
            'report': {'requests': 20, 'window': 3600}     # 20 reports per hour
        }
    
    def is_allowed(self, client_id: str, endpoint_type: str = 'default') -> bool:
        """Check if request is allowed under rate limit"""
        try:
            current_time = time.time()
            client_data = self.clients[client_id]
            limit_config = self.limits.get(endpoint_type, self.limits['default'])
            
            # Clean old requests outside the window
            window_start = current_time - limit_config['window']
            while client_data['requests'] and client_data['requests'][0] < window_start:
                client_data['requests'].popleft()
            
            # Check if under limit
            if len(client_data['requests']) >= limit_config['requests']:
                return False
            
            # Add current request
            client_data['requests'].append(current_time)
            return True
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return True  # Allow on error to avoid blocking legitimate requests
    
    def get_client_id(self, request: Request) -> str:
        """Generate client identifier from request"""
        # Use IP address as primary identifier
        client_ip = request.client.host if request.client else 'unknown'
        
        # Add user agent for additional uniqueness
        user_agent = request.headers.get('user-agent', '')
        client_hash = hashlib.md5(f"{client_ip}:{user_agent}".encode()).hexdigest()[:16]
        
        return f"{client_ip}_{client_hash}"


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request filtering and validation"""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.blocked_ips = set()  # In production, use Redis or database
        self.suspicious_patterns = [
            'script', 'javascript:', 'vbscript:', 'onload=', 'onerror=',
            '<script', '</script>', 'eval(', 'document.cookie',
            'union select', 'drop table', 'insert into', 'delete from'
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request through security filters"""
        try:
            start_time = time.time()
            
            # Get client identifier
            client_id = self.rate_limiter.get_client_id(request)
            
            # Check if IP is blocked
            if client_id.split('_')[0] in self.blocked_ips:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"error": "Access denied"}
                )
            
            # Determine endpoint type for rate limiting
            endpoint_type = self._get_endpoint_type(request.url.path)
            
            # Apply rate limiting
            if not self.rate_limiter.is_allowed(client_id, endpoint_type):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after": 3600,
                        "limit_type": endpoint_type
                    }
                )
            
            # Check for suspicious patterns in URL and headers
            if self._contains_suspicious_content(request):
                logger.warning(f"Suspicious request from {client_id}: {request.url}")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"error": "Invalid request"}
                )
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: https:; script-src 'self'"
            
            # Log request
            process_time = time.time() - start_time
            logger.info(f"Request: {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal server error"}
            )
    
    def _get_endpoint_type(self, path: str) -> str:
        """Determine endpoint type for rate limiting"""
        if '/upload' in path or '/reports' in path:
            return 'upload'
        elif '/auth' in path or '/login' in path or '/register' in path:
            return 'auth'
        elif '/api/reports' in path:
            return 'report'
        else:
            return 'default'
    
    def _contains_suspicious_content(self, request: Request) -> bool:
        """Check for suspicious patterns in request"""
        try:
            # Check URL path
            url_path = str(request.url).lower()
            for pattern in self.suspicious_patterns:
                if pattern in url_path:
                    return True
            
            # Check headers
            for header_name, header_value in request.headers.items():
                header_value_lower = header_value.lower()
                for pattern in self.suspicious_patterns:
                    if pattern in header_value_lower:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Suspicious content check failed: {e}")
            return False


class FileValidator:
    """Secure file upload validation"""
    
    def __init__(self):
        """Initialize file validator"""
        self.allowed_mime_types = {
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            'image/bmp', 'image/tiff'
        }
        self.allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.max_image_dimensions = (4096, 4096)  # 4K max
        
        # Initialize libmagic for file type detection
        try:
            self.magic = magic.Magic(mime=True)
        except Exception as e:
            logger.warning(f"libmagic not available, using fallback validation: {e}")
            self.magic = None
    
    async def validate_upload(self, file: UploadFile) -> Dict[str, Any]:
        """
        Comprehensive file validation for uploads
        
        Args:
            file: FastAPI UploadFile object
            
        Returns:
            Dict with validation results
        """
        try:
            validation_result = {
                'valid': False,
                'filename': file.filename,
                'size': 0,
                'mime_type': None,
                'dimensions': None,
                'errors': []
            }
            
            # Read file content
            content = await file.read()
            await file.seek(0)  # Reset file pointer
            
            validation_result['size'] = len(content)
            
            # Check file size
            if len(content) > self.max_file_size:
                validation_result['errors'].append(f"File too large: {len(content)} bytes (max: {self.max_file_size})")
                return validation_result
            
            if len(content) == 0:
                validation_result['errors'].append("Empty file")
                return validation_result
            
            # Validate file extension
            if file.filename:
                file_ext = os.path.splitext(file.filename.lower())[1]
                if file_ext not in self.allowed_extensions:
                    validation_result['errors'].append(f"Invalid file extension: {file_ext}")
                    return validation_result
            
            # Detect actual MIME type
            detected_mime = self._detect_mime_type(content)
            validation_result['mime_type'] = detected_mime
            
            if detected_mime not in self.allowed_mime_types:
                validation_result['errors'].append(f"Invalid file type: {detected_mime}")
                return validation_result
            
            # Validate image content
            image_validation = self._validate_image_content(content)
            if not image_validation['valid']:
                validation_result['errors'].extend(image_validation['errors'])
                return validation_result
            
            validation_result['dimensions'] = image_validation['dimensions']
            
            # Check for embedded threats
            threat_check = self._check_for_threats(content)
            if not threat_check['safe']:
                validation_result['errors'].extend(threat_check['threats'])
                return validation_result
            
            validation_result['valid'] = True
            return validation_result
            
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            return {
                'valid': False,
                'filename': file.filename,
                'errors': [f"Validation error: {str(e)}"]
            }
    
    def _detect_mime_type(self, content: bytes) -> str:
        """Detect MIME type from file content"""
        try:
            if self.magic:
                return self.magic.from_buffer(content)
            else:
                # Fallback: check file signature
                if content.startswith(b'\xff\xd8\xff'):
                    return 'image/jpeg'
                elif content.startswith(b'\x89PNG\r\n\x1a\n'):
                    return 'image/png'
                elif content.startswith(b'GIF8'):
                    return 'image/gif'
                elif content.startswith(b'RIFF') and b'WEBP' in content[:12]:
                    return 'image/webp'
                else:
                    return 'application/octet-stream'
        except Exception as e:
            logger.error(f"MIME type detection failed: {e}")
            return 'application/octet-stream'
    
    def _validate_image_content(self, content: bytes) -> Dict[str, Any]:
        """Validate image content and extract metadata"""
        try:
            # Try to load image with PIL
            try:
                image = Image.open(io.BytesIO(content))
                width, height = image.size
                
                # Check dimensions
                if width > self.max_image_dimensions[0] or height > self.max_image_dimensions[1]:
                    return {
                        'valid': False,
                        'errors': [f"Image too large: {width}x{height} (max: {self.max_image_dimensions[0]}x{self.max_image_dimensions[1]})"]
                    }
                
                # Verify image can be processed
                image.verify()
                
                return {
                    'valid': True,
                    'dimensions': (width, height),
                    'format': image.format,
                    'errors': []
                }
                
            except Exception as pil_error:
                # Try with OpenCV as fallback
                nparr = np.frombuffer(content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    return {
                        'valid': False,
                        'errors': [f"Invalid image format: {str(pil_error)}"]
                    }
                
                height, width = img.shape[:2]
                return {
                    'valid': True,
                    'dimensions': (width, height),
                    'errors': []
                }
                
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Image validation failed: {str(e)}"]
            }
    
    def _check_for_threats(self, content: bytes) -> Dict[str, Any]:
        """Check for embedded threats in file content"""
        try:
            content_str = content.decode('utf-8', errors='ignore').lower()
            
            # Check for script injections
            script_patterns = [
                'javascript:', 'vbscript:', '<script', 'eval(',
                'document.cookie', 'window.location', 'alert('
            ]
            
            threats = []
            for pattern in script_patterns:
                if pattern in content_str:
                    threats.append(f"Suspicious content detected: {pattern}")
            
            # Check for executable signatures
            executable_signatures = [
                b'MZ',  # Windows PE
                b'\x7fELF',  # Linux ELF
                b'\xfe\xed\xfa',  # macOS Mach-O
            ]
            
            for sig in executable_signatures:
                if content.startswith(sig):
                    threats.append("Executable file detected")
            
            return {
                'safe': len(threats) == 0,
                'threats': threats
            }
            
        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
            return {'safe': True, 'threats': []}


class InputValidator:
    """Input validation for API requests"""
    
    @staticmethod
    def validate_gps_coordinates(lat: float, lon: float) -> bool:
        """Validate GPS coordinates"""
        return -90 <= lat <= 90 and -180 <= lon <= 180
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Basic email validation"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def sanitize_string(text: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not text:
            return ""
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', '\x00']
        for char in dangerous_chars:
            text = text.replace(char, '')
        
        # Limit length
        return text[:max_length].strip()
    
    @staticmethod
    def validate_report_data(data: Dict) -> Dict[str, Any]:
        """Validate report submission data"""
        errors = []
        
        # Required fields
        required_fields = ['description', 'latitude', 'longitude']
        for field in required_fields:
            if field not in data or not data[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate GPS coordinates
        if 'latitude' in data and 'longitude' in data:
            try:
                lat = float(data['latitude'])
                lon = float(data['longitude'])
                if not InputValidator.validate_gps_coordinates(lat, lon):
                    errors.append("Invalid GPS coordinates")
            except (ValueError, TypeError):
                errors.append("GPS coordinates must be numbers")
        
        # Validate description
        if 'description' in data:
            if len(data['description']) > 2000:
                errors.append("Description too long (max 2000 characters)")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }


# Factory functions
def create_rate_limiter():
    """Create rate limiter instance"""
    return RateLimiter()

def create_security_middleware(app, rate_limiter: RateLimiter):
    """Create security middleware"""
    return SecurityMiddleware(app, rate_limiter)

def create_file_validator():
    """Create file validator instance"""
    return FileValidator()

# Import for BytesIO
import io
