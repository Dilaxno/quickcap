import jwt
import hashlib
import secrets
import os
from passlib.hash import pbkdf2_sha256
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Request
import logging

logger = logging.getLogger(__name__)

# Load JWT configuration from environment variables
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET or JWT_SECRET == "your-secret-key-change-in-production":
    if os.getenv("ENVIRONMENT") == "production":
        raise ValueError("JWT_SECRET must be set in production environment")
    else:
        # Generate a random secret for development
        JWT_SECRET = secrets.token_urlsafe(64)
        logger.warning("Using generated JWT secret for development. Set JWT_SECRET environment variable for production.")

JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

class AuthManager:
    def __init__(self):
        self.secret = JWT_SECRET
        self.algorithm = JWT_ALGORITHM
    
    def create_token(self, user_id: str, email: str, plan: str = "free") -> str:
        """Create a JWT token for a user"""
        payload = {
            "user_id": user_id,
            "email": email,
            "plan": plan,
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def get_user_from_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """Extract user information from request headers"""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        return self.verify_token(token)
    
    def require_auth(self, request: Request) -> Dict[str, Any]:
        """Require authentication and return user info"""
        user = self.get_user_from_request(request)
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")
        return user
    
    def require_pro_plan(self, request: Request) -> Dict[str, Any]:
        """Require pro plan and return user info"""
        user = self.require_auth(request)
        if user["plan"] not in ["pro", "premium", "enterprise"]:
            raise HTTPException(
                status_code=403, 
                detail="This feature is only available for Pro users"
            )
        return user

# Global auth manager instance
auth = AuthManager()

def create_demo_token(user_id: str = "demo_user", email: str = "demo@example.com", plan: str = "pro") -> str:
    """Create a demo token for testing purposes"""
    return auth.create_token(user_id, email, plan)

def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-SHA256 (secure implementation)"""
    # Use passlib's pbkdf2_sha256 hasher which handles salt generation internally
    return pbkdf2_sha256.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its PBKDF2-SHA256 hash"""
    try:
        return pbkdf2_sha256.verify(password, hashed)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def generate_user_id() -> str:
    """Generate a unique user ID"""
    return secrets.token_urlsafe(16)