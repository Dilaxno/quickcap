# Cloudflare R2 Integration for GPU QuickCap

## 🎯 Overview

Your GPU QuickCap application now includes full Cloudflare R2 cloud storage integration! Videos are automatically uploaded to R2 after processing while maintaining local copies for redundancy.

## ✅ **Current Status: READY TO USE**

Your R2 integration is fully configured and tested:
- ✅ Credentials configured
- ✅ Bucket connection verified  
- ✅ API endpoints working
- ✅ Upload functionality integrated

## 🚀 Quick Start

### 1. **Start Your Application**
```bash
cd "d:/GPU quickcap/backend"
python app.py
```

### 2. **Test R2 Integration**
```bash
# Test R2 connection
python test_r2_connection.py

# Test API endpoints (while server is running)
python test_api_endpoints.py

# Manage R2 storage
python r2_manager.py list
python r2_manager.py stats
```

## 📋 New API Endpoints

Your application now includes these R2 storage endpoints:

### **GET /api/storage/status**
Check R2 storage status and configuration
```bash
curl http://localhost:8080/api/storage/status
```

### **GET /api/storage/videos**
List all videos in R2 storage
```bash
curl http://localhost:8080/api/storage/videos
```

### **GET /api/storage/videos/{video_id}**
Get detailed information about a specific video
```bash
curl http://localhost:8080/api/storage/videos/12345
```

### **DELETE /api/storage/videos/{video_id}**
Delete a video from R2 storage
```bash
curl -X DELETE http://localhost:8080/api/storage/videos/12345
```

### **GET /api/storage/videos/{video_id}/url**
Get public and presigned URLs for a video
```bash
curl http://localhost:8080/api/storage/videos/12345/url
```

## 🔄 Automatic Upload Workflow

When you process a video through `/api/upload`, the system now:

1. **Processes** the video locally (whisper, object detection, etc.)
2. **Saves** the result to local storage
3. **Uploads** to R2 automatically 
4. **Returns** the video with R2 URLs in response headers:
   - `X-R2-URL`: Public R2 URL
   - `X-R2-Upload-Time`: Upload duration
   - `X-Storage-Type`: "local+r2"

## 🛠️ Command Line Tools

### **R2 Manager** (`r2_manager.py`)
Complete R2 storage management:

```bash
# List all videos
python r2_manager.py list

# Get video information
python r2_manager.py info videos/video-123.mp4

# Delete a video
python r2_manager.py delete videos/video-123.mp4

# Show storage statistics  
python r2_manager.py stats

# Cleanup old videos (older than 30 days)
python r2_manager.py cleanup --days 30
```

### **Connection Test** (`test_r2_connection.py`)
Verify R2 configuration and connectivity:

```bash
python test_r2_connection.py
```

### **API Test** (`test_api_endpoints.py`)
Test all R2 API endpoints (run while server is active):

```bash
python test_api_endpoints.py
```

## 📊 Current Configuration

Your R2 setup:
- **Account ID**: `c0cf485e1783f826edf7ebfa958f3d49`
- **Bucket**: `quickcap-videos`
- **Public Domain**: `videos.quickcap.pro`
- **Region**: Cloudflare R2 Global

## 🌐 Frontend Integration

Your frontend can now access R2 URLs from response headers:

```javascript
// After video upload
const r2Url = response.headers['X-R2-URL'];
const uploadTime = response.headers['X-R2-Upload-Time'];
const storageType = response.headers['X-Storage-Type'];

console.log('Video available at:', r2Url);
```

## 📈 Benefits

- **🌍 Global CDN**: Videos served from Cloudflare's global network
- **⚡ Fast Access**: Direct R2 URLs for immediate video playback
- **💾 Redundancy**: Videos stored both locally and in cloud
- **🔒 Security**: Presigned URLs for secure, temporary access
- **📊 Management**: Full API for video lifecycle management
- **💰 Cost Effective**: R2's competitive pricing vs traditional cloud storage

## 🔍 Monitoring & Logs

R2 operations are fully logged in your application:
- Upload success/failure
- Upload timing
- API endpoint usage
- Error handling

Check your application logs for detailed R2 activity.

## 🛡️ Security Features

- **Environment Variables**: Credentials stored securely in `.env`
- **Presigned URLs**: Temporary, secure access to videos
- **Error Handling**: Graceful fallback if R2 is unavailable
- **Access Control**: Bucket-level permissions

## 🔧 Troubleshooting

If you encounter issues:

1. **Check R2 Status**: `python test_r2_connection.py`
2. **Verify Credentials**: Ensure `.env` has correct values
3. **Test Connectivity**: Check bucket permissions and network
4. **Review Logs**: Application logs show detailed R2 operations

## 📞 Next Steps

Your R2 integration is ready! You can now:

1. **Upload videos** and see them automatically stored in R2
2. **Use the API endpoints** to manage your video storage
3. **Access videos** via fast R2 URLs
4. **Monitor usage** through the management tools

Enjoy your new cloud-powered video storage! 🎉