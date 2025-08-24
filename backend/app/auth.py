"""
Role-Based Access Control (RBAC) Authentication System
Implements JWT authentication with citizen, inspector, and admin roles
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from enum import Enum
import os

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'billboard-sentinel-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

class UserRole(Enum):
    """User roles for RBAC system"""
    CITIZEN = "citizen"
    INSPECTOR = "municipal_inspector" 
    ADMIN = "admin"

class Permission(Enum):
    """System permissions"""
    SUBMIT_REPORT = "submit_report"
    VIEW_OWN_REPORTS = "view_own_reports"
    VIEW_ALL_REPORTS = "view_all_reports"
    UPDATE_REPORT_STATUS = "update_report_status"
    DELETE_REPORT = "delete_report"
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_DATA = "export_data"
    MANAGE_VIOLATIONS = "manage_violations"
    APPROVE_REPORTS = "approve_reports"

# Role-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.CITIZEN: [
        Permission.SUBMIT_REPORT,
        Permission.VIEW_OWN_REPORTS
    ],
    UserRole.INSPECTOR: [
        Permission.SUBMIT_REPORT,
        Permission.VIEW_OWN_REPORTS,
        Permission.VIEW_ALL_REPORTS,
        Permission.UPDATE_REPORT_STATUS,
        Permission.MANAGE_VIOLATIONS,
        Permission.APPROVE_REPORTS,
        Permission.VIEW_ANALYTICS
    ],
    UserRole.ADMIN: [
        Permission.SUBMIT_REPORT,
        Permission.VIEW_OWN_REPORTS,
        Permission.VIEW_ALL_REPORTS,
        Permission.UPDATE_REPORT_STATUS,
        Permission.DELETE_REPORT,
        Permission.MANAGE_USERS,
        Permission.VIEW_ANALYTICS,
        Permission.EXPORT_DATA,
        Permission.MANAGE_VIOLATIONS,
        Permission.APPROVE_REPORTS
    ]
}

class AuthManager:
    """Handles authentication and authorization"""
    
    def __init__(self):
        """Initialize authentication manager"""
        self.security = HTTPBearer()
        self.users_db = {}  # In production, use proper database
        self._init_default_users()
    
    def _init_default_users(self):
        """Initialize default users for demo"""
        default_users = [
            {
                'user_id': 'citizen_demo',
                'email': 'citizen@demo.com',
                'password': 'demo123',
                'role': UserRole.CITIZEN,
                'name': 'Demo Citizen',
                'verified': True
            },
            {
                'user_id': 'inspector_demo',
                'email': 'inspector@city.gov',
                'password': 'inspector123',
                'role': UserRole.INSPECTOR,
                'name': 'City Inspector',
                'department': 'Code Enforcement',
                'verified': True
            },
            {
                'user_id': 'admin_demo',
                'email': 'admin@billboard-sentinel.org',
                'password': 'admin123',
                'role': UserRole.ADMIN,
                'name': 'System Admin',
                'verified': True
            }
        ]
        
        for user_data in default_users:
            self.create_user(user_data)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def create_user(self, user_data: Dict) -> Dict:
        """Create new user account"""
        try:
            user_id = user_data['user_id']
            
            if user_id in self.users_db:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User already exists"
                )
            
            # Hash password
            hashed_password = self.hash_password(user_data['password'])
            
            user_record = {
                'user_id': user_id,
                'email': user_data['email'],
                'password_hash': hashed_password,
                'role': user_data['role'],
                'name': user_data['name'],
                'created_at': datetime.now().isoformat(),
                'verified': user_data.get('verified', False),
                'active': True,
                'department': user_data.get('department'),
                'phone': user_data.get('phone'),
                'last_login': None
            }
            
            self.users_db[user_id] = user_record
            logger.info(f"Created user: {user_id} with role: {user_data['role'].value}")
            
            # Return user info without password
            user_info = user_record.copy()
            del user_info['password_hash']
            return user_info
            
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="User creation failed"
            )
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate user credentials"""
        try:
            # Find user by email
            user = None
            for user_record in self.users_db.values():
                if user_record['email'] == email:
                    user = user_record
                    break
            
            if not user:
                return None
            
            if not user['active']:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is deactivated"
                )
            
            # Verify password
            if not self.verify_password(password, user['password_hash']):
                return None
            
            # Update last login
            user['last_login'] = datetime.now().isoformat()
            
            # Return user info without password
            user_info = user.copy()
            del user_info['password_hash']
            return user_info
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None
    
    def create_access_token(self, user_data: Dict) -> str:
        """Create JWT access token"""
        try:
            payload = {
                'user_id': user_data['user_id'],
                'email': user_data['email'],
                'role': user_data['role'].value if isinstance(user_data['role'], UserRole) else user_data['role'],
                'name': user_data['name'],
                'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
                'iat': datetime.utcnow(),
                'iss': 'billboard-sentinel'
            }
            
            token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
            return token
            
        except Exception as e:
            logger.error(f"Token creation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token creation failed"
            )
    
    def verify_token(self, token: str) -> Dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            # Check if user still exists and is active
            user_id = payload['user_id']
            if user_id not in self.users_db:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            user = self.users_db[user_id]
            if not user['active']:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is deactivated"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def has_permission(self, user_role: Union[str, UserRole], permission: Permission) -> bool:
        """Check if user role has specific permission"""
        try:
            if isinstance(user_role, str):
                user_role = UserRole(user_role)
            
            role_permissions = ROLE_PERMISSIONS.get(user_role, [])
            return permission in role_permissions
            
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False
    
    def require_permission(self, permission: Permission):
        """Decorator factory for requiring specific permissions"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # This would be used with FastAPI dependency injection
                # Implementation depends on how current user is passed
                return func(*args, **kwargs)
            return wrapper
        return decorator


# FastAPI Dependencies
auth_manager = AuthManager()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    """FastAPI dependency to get current authenticated user"""
    try:
        token = credentials.credentials
        payload = auth_manager.verify_token(token)
        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_citizen(current_user: dict = Depends(get_current_user)):
    """Require citizen role or higher"""
    user_role = UserRole(current_user['role'])
    if user_role not in [UserRole.CITIZEN, UserRole.INSPECTOR, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    return current_user

async def get_current_inspector(current_user: dict = Depends(get_current_user)):
    """Require inspector role or higher"""
    user_role = UserRole(current_user['role'])
    if user_role not in [UserRole.INSPECTOR, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inspector access required"
        )
    return current_user

async def get_current_admin(current_user: dict = Depends(get_current_user)):
    """Require admin role"""
    user_role = UserRole(current_user['role'])
    if user_role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

def require_permission(permission: Permission):
    """Dependency factory for permission-based access control"""
    async def permission_checker(current_user: dict = Depends(get_current_user)):
        user_role = UserRole(current_user['role'])
        if not auth_manager.has_permission(user_role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission.value}"
            )
        return current_user
    return permission_checker

# Login/Registration Models (for API endpoints)
class LoginRequest:
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password

class RegisterRequest:
    def __init__(self, email: str, password: str, name: str, role: str = "citizen"):
        self.email = email
        self.password = password
        self.name = name
        self.role = role

class TokenResponse:
    def __init__(self, access_token: str, token_type: str, user_info: Dict):
        self.access_token = access_token
        self.token_type = token_type
        self.user_info = user_info


def create_auth_manager():
    """Factory function to create auth manager"""
    return auth_manager
