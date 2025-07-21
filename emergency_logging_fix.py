#!/usr/bin/env python3
"""
Emergency logging fix that can be applied to running production systems
This script patches the logging system without requiring a full restart
"""

import os
import sys
import logging
from datetime import datetime

def emergency_logging_setup():
    """
    Quick fix for production logging issues
    This can be imported and called from your existing app
    """
    
    # Ensure we're in production mode
    is_production = os.getenv("ENVIRONMENT") == "production" or os.getenv("NODE_ENV") == "production"
    
    if not is_production:
        print("⚠️  Not in production mode - no changes made")
        return
    
    print("🚨 Applying emergency logging fix...")
    
    # Create log directory if it doesn't exist
    log_dir = "/var/log/quickcap"
    try:
        os.makedirs(log_dir, exist_ok=True)
        print(f"📁 Log directory ensured: {log_dir}")
    except PermissionError:
        # Fallback to tmp if we can't write to /var/log
        log_dir = "/tmp/quickcap_logs"
        os.makedirs(log_dir, exist_ok=True)
        print(f"📁 Using fallback log directory: {log_dir}")
    
    # Setup file logging
    log_file = os.path.join(log_dir, "quickcap.log")
    
    # Create a new file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Get the root logger and add our handler
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
    
    # Create a custom print function that also logs
    original_print = __builtins__['print']
    
    def logging_print(*args, **kwargs):
        # Convert arguments to string
        message = ' '.join(str(arg) for arg in args)
        
        # Log the message
        if message.startswith('[ERROR]') or '❌' in message:
            root_logger.error(message)
        elif message.startswith('[WARNING]') or '⚠️' in message:
            root_logger.warning(message)
        else:
            root_logger.info(message)
        
        # Still print to stdout for systemd journal
        original_print(*args, **kwargs)
    
    # Replace the print function
    __builtins__['print'] = logging_print
    
    # Log the fix application
    root_logger.info("🔧 Emergency logging fix applied successfully")
    root_logger.info(f"📝 Logs are now being written to: {log_file}")
    
    print(f"✅ Emergency logging fix applied!")
    print(f"📝 Logs are now being written to: {log_file}")
    print(f"🔍 Monitor with: tail -f {log_file}")
    
    return log_file

if __name__ == "__main__":
    emergency_logging_setup()