# Gunicorn configuration file for production deployment
import os
import time
import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = min(4, multiprocessing.cpu_count())
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 300
keepalive = 2

# Application
module = "app:app"

# Logging
loglevel = "info"
accesslog = "/var/log/quickcap/access.log"
errorlog = "/var/log/quickcap/error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Capture output from print statements and send to logs
capture_output = True

# Process naming
proc_name = "quickcap-backend"

# Server mechanics
preload_app = True
daemon = False
pidfile = "/var/run/quickcap/quickcap.pid"
tmp_upload_dir = None

# SSL (if needed)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

def when_ready(server):
    server.log.info("QuickCap backend server is ready. Listening on: %s", server.address)

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)
    
    # Initialize Whisper model for this worker
    try:
        from worker_model_manager import ensure_worker_model_loaded
        server.log.info("Initializing Whisper model for worker %s", worker.pid)
        start_time = time.time()
        
        if ensure_worker_model_loaded():
            init_time = time.time() - start_time
            server.log.info("✅ Worker %s model loaded in %.2fs", worker.pid, init_time)
        else:
            server.log.error("❌ Worker %s failed to load model", worker.pid)
    except Exception as e:
        server.log.error("❌ Worker %s model initialization error: %s", worker.pid, e)