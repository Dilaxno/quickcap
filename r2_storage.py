"""
Cloudflare R2 Storage Service
Handles video upload, storage, and retrieval from Cloudflare R2
"""

import os
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional, Tuple
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class R2StorageService:
    """Service for managing Cloudflare R2 storage operations"""
    
    def __init__(self):
        """Initialize R2 client with credentials from environment"""
        self.account_id = os.getenv("R2_ACCOUNT_ID")
        self.access_key_id = os.getenv("R2_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("R2_BUCKET_NAME")
        self.endpoint_url = os.getenv("R2_ENDPOINT_URL")
        self.public_url = os.getenv("R2_PUBLIC_URL")
        
        # Validate required environment variables
        required_vars = [
            "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", 
            "R2_BUCKET_NAME", "R2_ENDPOINT_URL"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing required R2 environment variables: {missing_vars}")
            self.client = None
            self.enabled = False
            return
        
        try:
            # Initialize boto3 S3 client for R2
            self.client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name='auto'  # R2 uses 'auto' as region
            )
            
            # Test connection
            self._test_connection()
            self.enabled = True
            logger.info("✅ R2 Storage service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize R2 client: {e}")
            self.client = None
            self.enabled = False
    
    def _test_connection(self) -> bool:
        """Test R2 connection by listing bucket contents"""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"✅ R2 bucket '{self.bucket_name}' is accessible")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"❌ R2 bucket '{self.bucket_name}' not found")
            else:
                logger.error(f"❌ R2 connection test failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ R2 connection test failed: {e}")
            return False
    
    def upload_video(self, local_file_path: str, video_id: str, content_type: str = "video/mp4", 
                    metadata: dict = None, transcription_file_path: Optional[str] = None) -> Optional[str]:
        """
        Upload a video file to R2 storage with metadata
        
        Args:
            local_file_path: Path to the local video file
            video_id: Unique identifier for the video
            content_type: MIME type of the file
            metadata: Additional metadata to store with the video
            transcription_file_path: Path to the transcription file (optional)
            
        Returns:
            Public URL of the uploaded file, or None if upload failed
        """
        if not self.enabled:
            logger.warning("R2 storage is not enabled - skipping upload")
            return None
            
        if not os.path.exists(local_file_path):
            logger.error(f"File not found: {local_file_path}")
            return None
        
        # Generate object key (filename in R2)
        file_extension = os.path.splitext(local_file_path)[1]
        object_key = f"videos/{video_id}{file_extension}"
        
        try:
            upload_start = time.time()
            file_size = os.path.getsize(local_file_path)
            
            logger.info(f"📤 Uploading to R2: {local_file_path} -> {object_key} ({file_size / 1024 / 1024:.1f} MB)")
            
            # Upload file with metadata
            file_metadata = {
                'video-id': video_id,
                'upload-timestamp': str(int(time.time())),
                'original-filename': os.path.basename(local_file_path)
            }
            
            # Add additional metadata if provided
            if metadata:
                # Convert all metadata values to strings for R2 compatibility
                for key, value in metadata.items():
                    # Replace invalid characters in metadata keys
                    clean_key = key.replace('_', '-').replace(' ', '-').lower()
                    file_metadata[clean_key] = str(value)
            
            extra_args = {
                'ContentType': content_type,
                'Metadata': file_metadata
            }
            
            self.client.upload_file(
                local_file_path,
                self.bucket_name,
                object_key,
                ExtraArgs=extra_args
            )
            
            upload_time = time.time() - upload_start
            logger.info(f"✅ Upload completed in {upload_time:.2f}s")
            
            # Generate public URL
            public_url = self._get_public_url(object_key)
            logger.info(f"📁 Video available at: {public_url}")
            
            return public_url
            
        except ClientError as e:
            logger.error(f"❌ R2 upload failed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error during R2 upload: {e}")
            return None
    
    def _get_public_url(self, object_key: str) -> str:
        """Generate public URL for an R2 object"""
        if self.public_url:
            # Use custom domain if configured
            base_url = self.public_url.rstrip('/')
            return f"{base_url}/{object_key}"
        else:
            # Use default R2 public URL format
            return f"https://{self.bucket_name}.{self.account_id}.r2.cloudflarestorage.com/{object_key}"
    
    def delete_video(self, object_key: str) -> bool:
        """
        Delete a video from R2 storage
        
        Args:
            object_key: The object key (filename) in R2
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self.enabled:
            logger.warning("R2 storage is not enabled - skipping delete")
            return False
        
        try:
            logger.info(f"🗑️  Deleting from R2: {object_key}")
            
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            
            logger.info(f"✅ Successfully deleted: {object_key}")
            return True
            
        except ClientError as e:
            logger.error(f"❌ R2 delete failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error during R2 delete: {e}")
            return False
    
    def get_video_info(self, object_key: str) -> Optional[dict]:
        """
        Get metadata information about a video in R2
        
        Args:
            object_key: The object key (filename) in R2
            
        Returns:
            Dictionary with video metadata, or None if not found
        """
        if not self.enabled:
            return None
        
        try:
            response = self.client.head_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            
            return {
                'size': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType'),
                'metadata': response.get('Metadata', {}),
                'public_url': self._get_public_url(object_key)
            }
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.warning(f"Video not found in R2: {object_key}")
            else:
                logger.error(f"Error getting video info: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting video info: {e}")
            return None
    
    def list_videos(self, prefix: str = "videos/", limit: int = 100) -> list:
        """
        List videos in R2 storage
        
        Args:
            prefix: Object key prefix to filter by
            limit: Maximum number of objects to return
            
        Returns:
            List of video objects with metadata
        """
        if not self.enabled:
            return []
        
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=limit
            )
            
            videos = []
            for obj in response.get('Contents', []):
                videos.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'public_url': self._get_public_url(obj['Key'])
                })
            
            return videos
            
        except Exception as e:
            logger.error(f"Error listing videos: {e}")
            return []
    
    def generate_presigned_url(self, object_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for temporary access to a video
        
        Args:
            object_key: The object key (filename) in R2
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL or None if generation failed
        """
        if not self.enabled:
            return None
        
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None
    
    def cleanup_expired_videos(self, max_age_seconds: int = 432000) -> dict:
        """
        Clean up videos older than the specified age (default: 5 days)
        
        Args:
            max_age_seconds: Maximum age in seconds before deletion (default: 432000 = 5 days)
            
        Returns:
            Dictionary with cleanup results
        """
        if not self.enabled:
            return {'success': False, 'deletedCount': 0, 'error': 'R2 storage not enabled'}
        
        try:
            # List all videos in the bucket
            videos = self.list_videos()
            
            deleted_count = 0
            current_time = time.time()
            
            for video in videos:
                # Convert last_modified datetime to timestamp
                video_timestamp = video['last_modified'].timestamp()
                age_seconds = current_time - video_timestamp
                
                if age_seconds > max_age_seconds:
                    # Delete the expired video
                    if self.delete_video(video['key']):
                        deleted_count += 1
                        logger.info(f"Auto-deleted expired video: {video['key']} (age: {age_seconds:.1f}s)")
                    else:
                        logger.warning(f"Failed to delete expired video: {video['key']}")
            
            logger.info(f"Cleanup completed: {deleted_count} videos deleted")
            return {
                'success': True,
                'deletedCount': deleted_count,
                'totalChecked': len(videos),
                'maxAge': max_age_seconds
            }
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {'success': False, 'deletedCount': 0, 'error': str(e)}

# Create global instance
r2_storage = R2StorageService()