# QuickCap Production Logging Debug Guide

## Problem Summary
When running the QuickCap backend with systemd + Gunicorn in production, video processing logs are not appearing in log files after application startup.

## Root Cause
The application was using a mix of `print()` statements and `logger.info()` calls. When running with systemd + Gunicorn:
1. `print()` statements are not captured by the logging system
2. stdout/stderr redirection works differently than in development
3. Gunicorn may not forward all stdout to log files

## Solution Applied

### 1. Enhanced Logging Configuration
- **File**: `app.py` (lines 51-103)
- **Changes**:
  - Added production-specific logging setup
  - Redirects all output to `/var/log/quickcap/` in production
  - Created custom `log_print()` function that writes to both logs and stdout
  - Automatically replaces `print()` with `log_print()` in production

### 2. Gunicorn Configuration
- **File**: `gunicorn.conf.py`
- **Features**:
  - Proper worker configuration for video processing
  - Explicit log file paths
  - `capture_output = True` to capture print statements
  - Enhanced timeout settings for video processing

### 3. Systemd Service Configuration
- **File**: `quickcap.service`
- **Features**:
  - Proper environment variables
  - Security hardening
  - Log redirection to systemd journal
  - Auto-restart on failure

## Deployment Steps

### 1. Setup Logging Infrastructure
```bash
# Run the setup script (on your production server)
sudo ./setup_production_logging.sh
```

### 2. Update Your Current Deployment
```bash
# Copy the new configuration files to your production server
scp gunicorn.conf.py your-server:/opt/quickcap/backend/
scp quickcap.service your-server:/etc/systemd/system/

# Copy the updated app.py
scp app.py your-server:/opt/quickcap/backend/

# On your production server:
sudo systemctl daemon-reload
sudo systemctl restart quickcap.service
```

### 3. Verify Logging is Working
```bash
# Check if logs are being written
tail -f /var/log/quickcap/quickcap.log

# Check for errors
tail -f /var/log/quickcap/quickcap_error.log

# Check systemd journal
journalctl -u quickcap.service -f

# Test with a video upload and watch logs in real-time
```

## Log File Locations

### Production Environment
- **Main Log**: `/var/log/quickcap/quickcap.log`
- **Error Log**: `/var/log/quickcap/quickcap_error.log`
- **Access Log**: `/var/log/quickcap/access.log` (Gunicorn access logs)
- **Gunicorn Error Log**: `/var/log/quickcap/error.log`
- **Systemd Journal**: `journalctl -u quickcap.service`

### Development Environment
- **Main Log**: `gpu_quickcap.log` (in backend directory)
- **Console**: stdout/stderr

## Debugging Commands

### Check Service Status
```bash
sudo systemctl status quickcap.service
```

### View Recent Logs
```bash
# Last 50 lines of main log
tail -50 /var/log/quickcap/quickcap.log

# Follow logs in real-time
tail -f /var/log/quickcap/quickcap.log

# View systemd journal
journalctl -u quickcap.service --since "10 minutes ago"
```

### Test Video Processing
```bash
# Upload a test video and monitor logs
tail -f /var/log/quickcap/quickcap.log | grep -E "(PROCESSING|FFMPEG|OK|ERROR)"
```

### Check Log Permissions
```bash
ls -la /var/log/quickcap/
# Should show: drwxr-xr-x quickcap quickcap
```

## Expected Log Output During Video Processing

You should now see detailed logs like:
```
2024-01-20 10:30:15 - __main__ - INFO - [FFMPEG] NEW VIDEO PROCESSING REQUEST - 2024-01-20 10:30:15
2024-01-20 10:30:15 - __main__ - INFO - Processing file: test.mp4, template: standard, position: 0.70
2024-01-20 10:30:16 - __main__ - INFO - File saved: 15.2 MB in 0.45 seconds
2024-01-20 10:30:16 - __main__ - INFO - Video duration: 45.3 seconds (0.76 minutes)
2024-01-20 10:30:18 - __main__ - INFO - Transcription completed in 2.34 seconds
2024-01-20 10:30:25 - __main__ - INFO - FFmpeg processing completed successfully
```

## Troubleshooting

### If Logs Still Don't Appear
1. **Check log directory permissions**:
   ```bash
   sudo chown -R quickcap:quickcap /var/log/quickcap
   sudo chmod 755 /var/log/quickcap
   ```

2. **Verify environment variables**:
   ```bash
   sudo systemctl show quickcap.service | grep Environment
   ```

3. **Check if the service is using the right config**:
   ```bash
   ps aux | grep gunicorn
   ```

4. **Test logging manually**:
   ```bash
   # SSH to your server and test
   cd /opt/quickcap/backend
   python3 -c "
   import os
   os.environ['ENVIRONMENT'] = 'production'
   from app import logger
   logger.info('Test log message')
   print('Test print message')
   "
   ```

### If Video Processing Hangs
Check the Gunicorn timeout settings in `gunicorn.conf.py`:
```python
timeout = 300  # Increase if needed for large videos
```

## Log Rotation
Logs are automatically rotated daily and kept for 30 days. Check log rotation:
```bash
sudo logrotate -d /etc/logrotate.d/quickcap
```

## Performance Monitoring
Monitor log file sizes:
```bash
du -h /var/log/quickcap/*
```

## Next Steps After Deployment
1. Upload a test video to verify logging is working
2. Monitor resource usage: `htop`, `iotop`
3. Set up log monitoring/alerting if needed
4. Consider centralized logging (ELK stack, etc.) for production scale