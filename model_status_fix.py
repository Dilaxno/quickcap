"""
Simple script to force the model loading status to be correctly reported.
This creates a status file that can be read by any process.
"""
import os
import json
import time

# Define the status file path
STATUS_FILE = os.path.join(os.path.dirname(__file__), "whisper_model_status.json")

def update_model_status(loaded=True, loading=False):
    """Update the model status file"""
    status = {
        "model_loaded": loaded,
        "loading_in_progress": loading,
        "last_updated": time.time()
    }
    
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)
    
    print(f"Updated model status: {status}")

def get_model_status():
    """Get the current model status"""
    if not os.path.exists(STATUS_FILE):
        return {"model_loaded": False, "loading_in_progress": False}
    
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading model status: {e}")
        return {"model_loaded": False, "loading_in_progress": False, "error": str(e)}

if __name__ == "__main__":
    # If run directly, update the status to indicate the model is loaded
    update_model_status(loaded=True, loading=False)
    print("Model status updated to loaded=True")