import os
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RecaptchaService:
    """Service for verifying Google reCAPTCHA v2 tokens"""
    
    def __init__(self):
        self.secret_key = os.getenv("RECAPTCHA_SECRET_KEY", "6LfjIYgrAAAAANbYFjjaMGGQ1T6X6fRaoInrpehv")
        self.site_key = os.getenv("RECAPTCHA_SITE_KEY", "6LfjIYgrAAAAAID6ug4Dy57QX4zUZhbNzeNJjPRs")
        self.verify_url = "https://www.google.com/recaptcha/api/siteverify"
        
    def verify_token(self, token: str, remote_ip: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify a reCAPTCHA token with Google's API
        
        Args:
            token: The reCAPTCHA response token
            remote_ip: Optional IP address of the user
            
        Returns:
            Dict containing verification result and additional info
        """
        try:
            # Prepare data for verification
            data = {
                'secret': self.secret_key,
                'response': token
            }
            
            if remote_ip:
                data['remoteip'] = remote_ip
            
            # Make request to Google's verification endpoint
            response = requests.post(
                self.verify_url,
                data=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                logger.info(f"reCAPTCHA verification result: {result}")
                
                return {
                    'success': result.get('success', False),
                    'challenge_ts': result.get('challenge_ts'),
                    'hostname': result.get('hostname'),
                    'error_codes': result.get('error-codes', []),
                    'score': result.get('score'),  # For v3 compatibility
                    'action': result.get('action')  # For v3 compatibility
                }
            else:
                logger.error(f"reCAPTCHA API request failed with status {response.status_code}")
                return {
                    'success': False,
                    'error_codes': ['request-failed'],
                    'message': f'API request failed with status {response.status_code}'
                }
                
        except requests.exceptions.Timeout:
            logger.error("reCAPTCHA verification timeout")
            return {
                'success': False,
                'error_codes': ['timeout'],
                'message': 'Verification request timed out'
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"reCAPTCHA verification request error: {e}")
            return {
                'success': False,
                'error_codes': ['request-error'],
                'message': f'Request error: {str(e)}'
            }
        except Exception as e:
            logger.error(f"Unexpected error during reCAPTCHA verification: {e}")
            return {
                'success': False,
                'error_codes': ['unknown-error'],
                'message': f'Unexpected error: {str(e)}'
            }
    
    def is_valid_token(self, token: str, remote_ip: Optional[str] = None) -> bool:
        """
        Simple boolean check for token validity
        
        Args:
            token: The reCAPTCHA response token
            remote_ip: Optional IP address of the user
            
        Returns:
            True if token is valid, False otherwise
        """
        result = self.verify_token(token, remote_ip)
        return result.get('success', False)
    
    def get_site_key(self) -> str:
        """
        Get the reCAPTCHA site key for frontend use
        
        Returns:
            The reCAPTCHA site key
        """
        return self.site_key

# Global instance
recaptcha_service = RecaptchaService()