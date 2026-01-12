import cv2
import numpy as np
import math
import base64
import time
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

# --- 1. ROBUST PHYSICS ENGINE (Phase Accumulation) ---
class PhysicsEngine:
    def __init__(self):
        self.accumulated_angle = 0.0  # Total radians travelled (can go to infinity)
        self.last_raw_angle = None
        self.last_time = time.time()
        
        self.current_rpm = 0.0
        self.rpm_buffer = []  # For smoothing
        
        # Output values
        self.display_rotations = 0
        self.display_rpm = 0

    def process_coordinates(self, x, y, width, height):
        # 1. Get Raw Angle (-PI to +PI)
        cx, cy = width // 2, height // 2
        # Note: We invert Y because image coordinates go down, but math coordinates go up
        dx = x - cx
        dy = -(y - cy) 
        
        raw_angle = math.atan2(dy, dx)

        if self.last_raw_angle is not None:
            # 2. Calculate Change (Delta)
            delta = raw_angle - self.last_raw_angle
            
            # 3. Fix Phase Wrap-Around (The "Jitter Killer")
            # If jump is huge (e.g. +3.1 to -3.1), it means we crossed the line.
            # We assume the wheel physically cannot rotate 180 degrees (PI) in one frame (0.05s).
            if delta > math.pi:
                delta -= 2 * math.pi
            elif delta < -math.pi:
                delta += 2 * math.pi
            
            # 4. Add to Total
            self.accumulated_angle += delta
            
            # 5. Calculate Rotations (Total Angle / 2PI)
            # Use floor logic so 0.99 rotations shows as 0, 1.01 shows as 1
            self.display_rotations = int(abs(self.accumulated_angle) / (2 * math.pi))
            
            # 6. Calculate RPM
            self.calculate_rpm(delta)
        
        self.last_raw_angle = raw_angle
        return self.display_rotations, self.display_rpm

    def calculate_rpm(self, delta_radians):
        now = time.time()
        dt = now - self.last_time
        
        # Only update RPM every 100ms to prevent number flickering
        if dt > 0.1:
            # RPM = (Radians per sec) * (60 / 2PI)
            # We calculate RPM based on the accumulated change over time dt
            rads_per_sec = abs(delta_radians) / dt # This is instantaneous, noisy
            
            # Better RPM: (Total angle change since last update) / dt
            # But for simplicity, we smooth the instantaneous value:
            
            # Calculate instantaneous RPM from angular velocity
            # velocity = rads / time
            # RPM = velocity * 9.5493
            instant_rpm = (abs(delta_radians) / dt) * 9.5493
            
            # Filter out crazy noise (e.g. > 3000 RPM is impossible for lab wheel)
            if instant_rpm < 3000:
                # Weighted Average (Smooths out jitter)
                # 80% old value, 20% new value
                self.current_rpm = (self.current_rpm * 0.8) + (instant_rpm * 0.2)
            
            self.display_rpm = int(self.current_rpm)
            self.last_time = now

# Initialize
physics = PhysicsEngine()

# --- 2. VISION PROCESSOR (Stricter Red Filter) ---
def process_frame(base64_string):
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return None
    except:
        return None

    # HSL Masking
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Range 1: Deep Red
    lower1 = np.array([0, 140, 60])   # Increased Saturation Min to 140 (Ignores Rust)
    upper1 = np.array([10, 255, 255])
    
    # Range 2: Bright Red
    lower2 = np.array([170, 140, 60]) 
    upper2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)

    # Clean Noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find Center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 30: # Filter small specs
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return cx, cy, frame.shape[1], frame.shape[0]
    return None

# --- 3. SERVER ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Reset physics on new connection
    physics.accumulated_angle = 0.0
    physics.last_raw_angle = None
    physics.current_rpm = 0.0
    
    try:
        while True:
            data = await websocket.receive_text()
            result = process_frame(data)
            
            if result:
                cx, cy, w, h = result
                rots, rpm = physics.process_coordinates(cx, cy, w, h)
                await websocket.send_text(f"{rots},{rpm},{cx},{cy}")
            else:
                await websocket.send_text("null")
                
            await asyncio.sleep(0.005) # Prevent CPU throttling
            
    except WebSocketDisconnect:
        pass