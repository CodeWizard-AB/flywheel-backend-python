import cv2 # type: ignore
import numpy as np # type: ignore
import math
import base64
import time
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect # type: ignore

app = FastAPI()

# ==============================
# YOUR PHYSICS LOGIC (PORTED)
# ==============================
class PhysicsEngine:
    def __init__(self):
        # State Variables from your script
        self.prev_angle = None
        self.total_rotation = 0.0
        self.movement_started = False
        self.stop_counter = 0
        self.STOP_THRESHOLD = 5 # 5 frames of no movement = Stop
        
        self.current_rpm = 0.0
        self.last_time = time.time()

    def process_coordinates(self, mx, my, width, height):
        # 1. DYNAMIC CENTER (Improvement: adapting to phone screen size)
        # Your code used fixed 320/240. We use the actual image center.
        CENTER_X = width // 2
        CENTER_Y = height // 2
        
        # 2. CALCULATE ANGLE (Your exact math)
        dx = mx - CENTER_X
        # Note: In computer vision, Y grows downwards. In math, Y grows upwards.
        # We invert Y here to match standard Cartesian coordinates.
        dy = -(my - CENTER_Y) 
        
        # Returns Degrees (-180 to 180)
        angle = math.degrees(math.atan2(dy, dx))
        
        delta = 0
        current_time = time.time()
        
        if self.prev_angle is not None:
            delta = angle - self.prev_angle
            
            # 3. WRAP AROUND LOGIC (Your exact math)
            if delta > 180:
                delta -= 360
            elif delta < -180:
                delta += 360
            
            # 4. MOVEMENT DETECTION
            if abs(delta) > 0.5: # Threshold from your code
                self.movement_started = True
                self.stop_counter = 0
                
                # Calculate RPM only when moving
                self.calculate_rpm(delta, current_time)
            else:
                if self.movement_started:
                    self.stop_counter += 1
            
            # 5. UPDATE TOTAL (Only if moving)
            if self.movement_started:
                self.total_rotation += delta

        self.prev_angle = angle
        self.last_time = current_time
        
        # 6. CHECK STOP CONDITION
        if self.movement_started and self.stop_counter >= self.STOP_THRESHOLD:
            # Flywheel stopped
            # We don't break the loop (server needs to stay alive), 
            # but we can signal zero RPM
            self.current_rpm = 0
            
        # Return formatted data
        display_rotations = int(abs(self.total_rotation) / 360)
        return display_rotations, int(self.current_rpm)

    def calculate_rpm(self, delta_deg, now):
        dt = now - self.last_time
        if dt > 0.05: # Prevent divide by zero / noise
            # RPM = (Degrees / time) * (60 sec / 360 deg)
            inst_rpm = (abs(delta_deg) / dt) * (60 / 360)
            
            # Smoothing (Weighted Average)
            self.current_rpm = (self.current_rpm * 0.7) + (inst_rpm * 0.3)

# Initialize Engine
physics = PhysicsEngine()

# ==============================
# YOUR VISION LOGIC (PORTED)
# ==============================
def process_frame(base64_string):
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(base64_string), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return None
    except:
        return None

    # 1. YOUR HSV RANGES
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # 2. YOUR CONTOUR LOGIC
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        # Add a small noise filter (area > 20) just in case
        if cv2.contourArea(largest) > 20: 
            M = cv2.moments(largest)
            if M["m00"] != 0:
                mx = int(M["m10"] / M["m00"])
                my = int(M["m01"] / M["m00"])
                return mx, my, frame.shape[1], frame.shape[0]
                
    return None

# ==============================
# SERVER HANDLER
# ==============================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Reset State on new connection
    physics.prev_angle = None
    physics.total_rotation = 0.0
    physics.movement_started = False
    
    try:
        while True:
            data = await websocket.receive_text()
            result = process_frame(data)
            
            if result:
                mx, my, w, h = result
                rots, rpm = physics.process_coordinates(mx, my, w, h)
                
                # Send back: Rotations, RPM, MarkerX, MarkerY
                await websocket.send_text(f"{rots},{rpm},{mx},{my}")
            else:
                # Marker lost
                await websocket.send_text("null")
            
            # Small yield to keep server responsive
            await asyncio.sleep(0.005)
            
    except WebSocketDisconnect:
        pass