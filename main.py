import cv2  # type: ignore
import numpy as np # type: ignore
import math
import base64
import time
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect # type: ignore

app = FastAPI()

# --- 1. PHYSICS ENGINE CLASS ---
class PhysicsEngine:
    def __init__(self):
        self.total_rotations = 0
        self.last_angle = None
        self.last_time = time.time()
        self.current_rpm = 0.0
        # Stabilization buffer
        self.rpm_buffer = []

    def process_coordinates(self, x, y, width, height):
        # Calculate center
        cx, cy = width // 2, height // 2
        dx = x - cx
        dy = y - cy
        
        # Calculate Angle (-PI to +PI)
        angle = math.atan2(dy, dx)

        # Logic: Detect full rotation
        if self.last_angle is not None:
            delta = angle - self.last_angle
            
            # Check for "Wrap Around" (passing the 3 o'clock position)
            # If angle jumps from ~3.14 to ~-3.14 (Counter-Clockwise)
            if delta < -5:
                self.count_rotation()
            # If angle jumps from ~-3.14 to ~3.14 (Clockwise)
            elif delta > 5:
                self.count_rotation()
        
        self.last_angle = angle
        return self.total_rotations, int(self.current_rpm)

    def count_rotation(self):
        self.total_rotations += 1
        now = time.time()
        diff = now - self.last_time
        
        if diff > 0:
            # Instant RPM = (1 rot / time_diff) * 60 seconds
            inst_rpm = (1.0 / diff) * 60.0
            
            # SMOOTHING ALGORITHM:
            # Real-world physics data is noisy. We use a "Moving Average".
            # 80% previous stable speed + 20% new speed
            self.current_rpm = (self.current_rpm * 0.8) + (inst_rpm * 0.2)
            
        self.last_time = now

# Initialize the engine
physics = PhysicsEngine()

# --- 2. COMPUTER VISION PROCESSOR ---
def process_frame(base64_string):
    try:
        # Decode the image sent from the phone
        # Remove header if present (data:image/jpeg;base64,...)
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
            
        nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return None
    except:
        return None

    # A. Convert to HSV (Hue, Saturation, Value)
    # This is critical for ignoring Rust. Rust has low Saturation.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # B. Define the "Vivid Red" Range
    # We use two ranges because Red wraps around 0 and 180 in HSV
    
    # Range 1: 0-10 (Deep Red)
    lower_red1 = np.array([0, 120, 70])    # Saturation min 120 filters out brown rust
    upper_red1 = np.array([10, 255, 255])
    
    # Range 2: 170-180 (Bright Red)
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Combine masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # C. Noise Reduction (Morphological Ops)
    # Erode removes small specks (rust noise), Dilate restores the marker size
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # D. Find the biggest red object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 50: # Minimum size filter
            moments = cv2.moments(largest_contour)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                return cx, cy, frame.shape[1], frame.shape[0]

    return None

# --- 3. WEBSOCKET SERVER ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client Connected")
    
    try:
        while True:
            # Receive image data
            data = await websocket.receive_text()
            
            # Process Frame
            result = process_frame(data)
            
            if result:
                cx, cy, width, height = result
                
                # Update Physics
                rotations, rpm = physics.process_coordinates(cx, cy, width, height)
                
                # Send data back: "rotations,rpm,x,y"
                await websocket.send_text(f"{rotations},{rpm},{cx},{cy}")
            else:
                # If marker lost, send "null" but keep connection alive
                await websocket.send_text("null")
                
            # Allow a tiny sleep to prevent CPU overload
            await asyncio.sleep(0.001)
            
    except WebSocketDisconnect:
        print("Client Disconnected")