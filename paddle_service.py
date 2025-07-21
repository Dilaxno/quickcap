"""
Paddle API service for subscription management
"""
import os
import requests
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class PaddleService:
    def __init__(self):
        self.api_key = os.getenv('PADDLE_API_KEY')
        self.vendor_id = os.getenv('PADDLE_VENDOR_ID')
        self.base_url = "https://api.paddle.com/v1"
        
        if not self.api_key or not self.vendor_id:
            logger.error("Paddle API key or vendor ID not configured")
            raise ValueError("Paddle API key or vendor ID not configured")
    
    def _make_request(self, endpoint: str, method: str = 'POST', data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a request to the Paddle API"""
        url = f"{self.base_url}/{endpoint}"
        
        # Add vendor_id to all requests
        request_data = data or {}
        request_data['vendor_id'] = self.vendor_id
        
        # For Paddle API v1, we need to use form data, not JSON
        headers = {}
        
        try:
            if method == 'POST':
                response = requests.post(url, data=request_data, headers=headers)
            elif method == 'GET':
                response = requests.get(url, params=request_data, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Paddle API request failed: {e}")
            raise Exception(f"Paddle API request failed: {e}")
    
    def cancel_subscription(self, subscription_id: str) -> Dict[str, Any]:
        """Cancel a subscription using Paddle API"""
        try:
            logger.info(f"Canceling subscription: {subscription_id}")
            
            # Cancel the subscription
            response = self._make_request(
                'subscription/users_cancel',
                method='POST',
                data={
                    'subscription_id': subscription_id
                }
            )
            
            if response.get('success'):
                logger.info(f"Successfully canceled subscription: {subscription_id}")
                return {
                    'success': True,
                    'message': 'Subscription canceled successfully',
                    'data': response.get('response', {})
                }
            else:
                logger.error(f"Failed to cancel subscription: {response}")
                return {
                    'success': False,
                    'message': 'Failed to cancel subscription',
                    'error': response.get('error', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"Error canceling subscription {subscription_id}: {e}")
            return {
                'success': False,
                'message': 'An error occurred while canceling subscription',
                'error': str(e)
            }
    
    def get_subscription_info(self, subscription_id: str) -> Dict[str, Any]:
        """Get subscription information from Paddle"""
        try:
            logger.info(f"Getting subscription info: {subscription_id}")
            
            response = self._make_request(
                'subscription/users',
                method='POST',
                data={
                    'subscription_id': subscription_id
                }
            )
            
            if response.get('success'):
                return {
                    'success': True,
                    'data': response.get('response', [])
                }
            else:
                return {
                    'success': False,
                    'error': response.get('error', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"Error getting subscription info {subscription_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_subscription(self, subscription_id: str, **kwargs) -> Dict[str, Any]:
        """Update a subscription"""
        try:
            logger.info(f"Updating subscription: {subscription_id}")
            
            data = {
                'subscription_id': subscription_id,
                **kwargs
            }
            
            response = self._make_request(
                'subscription/users/update',
                method='POST',
                data=data
            )
            
            if response.get('success'):
                return {
                    'success': True,
                    'data': response.get('response', {})
                }
            else:
                return {
                    'success': False,
                    'error': response.get('error', 'Unknown error')
                }
                
        except Exception as e:
            logger.error(f"Error updating subscription {subscription_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }