import cv2
import serial
import serial.tools.list_ports
import time
import threading
import asyncio
import websockets
import json
import base64
from queue import Queue
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QualityControlWebSystem:
    def __init__(self, 
                 esp32_port='COM4',
                 camera_id=0,
                 model_path='yolo8-1240-new.pt',
                 baud_rate=115200,
                 websocket_port=8765,
                 auto_detect_port=True):
        
        # Configuration
        self.esp32_port = esp32_port
        self.camera_id = camera_id
        self.model_path = model_path
        self.baud_rate = baud_rate
        self.websocket_port = websocket_port
        self.auto_detect_port = auto_detect_port
        
        # State variables
        self.ir_detected = False
        self.defect_detected = False
        self.relay_status = True  # True = ON, False = OFF
        self.system_running = False
        self.last_classification = "IDLE"
        self.classification_changed = False
        
        # Communication
        self.serial_connection = None
        self.camera = None
        self.yolo_model = None
        self.connected_clients = set()
        
        # Data storage
        self.current_detections = []
        self.results_log = []
        
        # Serial communication lock
        self.serial_lock = threading.Lock()
        
        # Initialize components
        self.init_serial()
        self.init_camera()
        self.init_yolo()
    
    def find_esp32_port(self):
        """Auto-detect ESP32 port"""
        logger.info("🔍 Searching for ESP32 device...")
        ports = serial.tools.list_ports.comports()
        
        # Look for common ESP32 identifiers
        esp32_keywords = ['USB', 'CH340', 'CP210', 'ESP32', 'Arduino', 'Silicon Labs']
        
        for port in ports:
            logger.info(f"Found port: {port.device} - {port.description}")
            
            # Check if description contains ESP32-related keywords
            for keyword in esp32_keywords:
                if keyword.lower() in port.description.lower():
                    logger.info(f"🎯 Potential ESP32 port found: {port.device}")
                    if self.test_esp32_connection(port.device):
                        return port.device
        
        # If no keyword match, test all available ports
        logger.info("🔧 Testing all available ports...")
        for port in ports:
            if self.test_esp32_connection(port.device):
                return port.device
        
        return None
    
    def test_esp32_connection(self, port, timeout=5):
        """Test if a port has ESP32 responding"""
        try:
            logger.info(f"🧪 Testing {port}...")
            ser = serial.Serial(
                port=port,
                baudrate=self.baud_rate,
                timeout=2,
                write_timeout=2
            )
            
            time.sleep(2)  # Wait for ESP32 initialization
            
            # Clear any existing data
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            
            # Try PING command
            ser.write(b"PING\n")
            time.sleep(0.5)
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                if ser.in_waiting:
                    response = ser.readline().decode('utf-8', errors='ignore').strip()
                    logger.info(f"Response from {port}: '{response}'")
                    
                    if "PONG" in response:
                        logger.info(f"ESP32 found on {port}")
                        ser.close()
                        return True
                
                time.sleep(0.1)
            
            ser.close()
            logger.info(f"No valid response from {port}")
            return False
            
        except Exception as e:
            logger.info(f"Error testing {port}: {e}")
            return False
    
    def init_serial(self):
        """Initialize ESP32 serial communication with auto-detection"""
        try:
            # Auto-detect port if enabled
            if self.auto_detect_port:
                detected_port = self.find_esp32_port()
                if detected_port:
                    self.esp32_port = detected_port
                    logger.info(f"Auto-detected ESP32 on {self.esp32_port}")
                else:
                    logger.error("Could not auto-detect ESP32 port")
                    return False
            
            # Try to connect to the specified/detected port
            logger.info(f"Connecting to ESP32 on {self.esp32_port}...")
            
            self.serial_connection = serial.Serial(
                port=self.esp32_port,
                baudrate=self.baud_rate,
                timeout=3,
                write_timeout=3
            )
            
            time.sleep(3)  # Give ESP32 time to reset and initialize
            
            # Clear buffers
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()
            
            # Test connection with retries
            max_retries = 3
            for attempt in range(max_retries):
                logger.info(f"Connection test attempt {attempt + 1}/{max_retries}")
                
                try:
                    self.serial_connection.write(b"PING\n")
                    time.sleep(1)
                    
                    if self.serial_connection.in_waiting:
                        response = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                        logger.info(f"ESP32 response: '{response}'")
                        
                        if "PONG" in response:
                            logger.info(f"ESP32 connected successfully on {self.esp32_port}")
                            
                            # Request initial status
                            self.serial_connection.write(b"STATUS\n")
                            time.sleep(0.5)
                            
                            return True
                
                except Exception as e:
                    logger.warning(f"⚠️ Connection test {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
            
            raise Exception("ESP32 not responding to PING after multiple attempts")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect ESP32: {e}")
            self.serial_connection = None
            
            # Show troubleshooting tips
            logger.error("  Troubleshooting tips:")
            logger.error("   1. Check if ESP32 is connected via USB")
            logger.error("   2. Verify ESP32 code is uploaded correctly")
            logger.error("   3. Try pressing ESP32 reset button")
            logger.error("   4. Check if another program is using the port")
            logger.error("   5. Install ESP32 USB drivers (CH340/CP210x)")
            logger.error("   6. Try different USB cable or port")
            
            return False
    
    def init_camera(self):
        """Initialize USB camera with better error handling"""
        try:
            logger.info(f"📹 Initializing camera {self.camera_id}...")
            
            self.camera = cv2.VideoCapture(self.camera_id)
            
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {self.camera_id}")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  
            self.camera.set(cv2.CAP_PROP_FPS, 30) 
            
            # Test camera
            ret, frame = self.camera.read()
            if not ret or frame is None:
                raise Exception("Cannot read from camera")
            
            logger.info("✅ Camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize camera: {e}")
            logger.error("🔧 Camera troubleshooting tips:")
            logger.error("   1. Check if camera is connected")
            logger.error("   2. Try different camera_id (0, 1, 2...)")
            logger.error("   3. Close other applications using camera")
            logger.error("   4. Check camera permissions")
            
            self.camera = None
            return False
    
    def init_yolo(self):
        """Initialize YOLO model with better error handling"""
        try:
            logger.info(f"🤖 Loading YOLO model: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise Exception(f"Model file not found: {self.model_path}")
            
            self.yolo_model = YOLO(self.model_path)
            logger.info("✅ YOLO model loaded successfully")
            
            # Test model with dummy data
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            results = self.yolo_model(dummy_frame)
            logger.info("✅ YOLO model test successful")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            logger.error("🔧 YOLO troubleshooting tips:")
            logger.error("   1. Check if model file exists")
            logger.error("   2. Verify model file is not corrupted")
            logger.error("   3. Check YOLO/Ultralytics installation")
            
            self.yolo_model = None
            return False
    
    def send_serial_command(self, command):
        """Thread-safe serial command sending"""
        if not self.serial_connection:
            return False
        
        try:
            with self.serial_lock:
                self.serial_connection.write(f"{command}\n".encode('utf-8'))
                self.serial_connection.flush()
                return True
        except Exception as e:
            logger.error(f"Error sending command '{command}': {e}")
            return False
    
    def read_ir_sensor(self):
        """Thread untuk membaca sensor IR secara kontinyu dengan error handling"""
        logger.info("🔄 Starting IR sensor reading thread...")
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.system_running:
            if not self.serial_connection:
                time.sleep(1)
                continue
            
            try:
                if self.serial_connection.in_waiting:
                    line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line:  # Only process non-empty lines
                        # Reset error counter on successful read
                        consecutive_errors = 0
                        
                        if line == "IR:DETECTED":
                            if not self.ir_detected:  # State changed
                                self.ir_detected = True
                                logger.info("📱 HP Unit detected by IR sensor")
                        elif line == "IR:NOT_DETECTED":
                            if self.ir_detected:  # State changed
                                self.ir_detected = False
                                logger.info("🚫 No HP Unit detected by IR sensor")
                        elif "RELAY:" in line:
                            status = line.split(":")[1]
                            if status in ["ON", "ON_OK"]:
                                self.relay_status = True
                            elif status in ["OFF", "OFF_OK"]:
                                self.relay_status = False
                        elif line.startswith("ESP32:"):
                            logger.info(f"📨 ESP32 message: {line}")
                
                time.sleep(0.05)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error reading serial (attempt {consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("🚨 Too many consecutive serial errors. Attempting reconnection...")
                    self.attempt_serial_reconnect()
                    consecutive_errors = 0
                
                time.sleep(0.5)
    
    def attempt_serial_reconnect(self):
        """Attempt to reconnect to ESP32"""
        logger.info("🔄 Attempting to reconnect to ESP32...")
        
        try:
            if self.serial_connection:
                self.serial_connection.close()
                
            time.sleep(2)
            
            # Try to reinitialize serial connection
            if self.init_serial():
                logger.info("✅ ESP32 reconnection successful")
            else:
                logger.error("❌ ESP32 reconnection failed")
                
        except Exception as e:
            logger.error(f"Error during reconnection: {e}")
    
    def detect_defects(self, frame):
        """Deteksi defect menggunakan YOLO dengan error handling"""
        if self.yolo_model is None or frame is None:
            return False, frame, []
        
        try:
            results = self.yolo_model(frame)
            detections = []
            defect_found = False
            
            # Target classes: Gelembung, Gelembung Pinggir, Bintik
            target_classes = ['Gelembung', 'Gelembung Pinggir', 'Bintik']
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = box.conf[0].item()
                        if confidence > 0.5:  # Confidence threshold
                            class_id = int(box.cls[0].item())
                            class_name = self.yolo_model.names[class_id]
                            
                            # Check if detected class is a defect type
                            if class_name in target_classes:
                                defect_found = True
                                
                                # Get bounding box
                                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                                
                                detections.append({
                                    'class': class_name,
                                    'confidence': confidence,
                                    'bbox': [x1, y1, x2, y2]
                                })
                                
                                # Draw bounding box
                                color = (0, 0, 255)  # Red for defects
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(frame, f'{class_name}: {confidence:.2f}', 
                                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.6, color, 2)
            
            self.current_detections = detections
            return defect_found, frame, detections
            
        except Exception as e:
            logger.error(f"Error in defect detection: {e}")
            return False, frame, []
    
    def classify_unit(self, ir_status, defect_status):
        """Klasifikasi unit berdasarkan IR dan deteksi defect"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine classification
        if not ir_status:
            classification = "NO_DETECTION"
            status = "⏸️ No Detection - Waiting for unit"
        elif ir_status and not defect_status:
            classification = "UNIT_NORMAL"
            status = "✅ Unit Normal - No defects detected"
        elif ir_status and defect_status:
            classification = "UNIT_DEFECT"
            status = "❌ Unit Defect - Screen defects found"
        else:
            classification = "IDLE"
            status = "⏸️ System Idle"
        
        # Check if classification changed
        self.classification_changed = (classification != self.last_classification)
        self.last_classification = classification
        
        # Control relay based on classification
        if classification == "UNIT_DEFECT":
            self.control_relay(False)  # Turn OFF relay for defects
        else:
            self.control_relay(True)   # Turn ON relay for normal/no detection
        
        # Log result
        result = {
            'timestamp': timestamp,
            'ir_status': ir_status,
            'defect_status': defect_status,
            'classification': classification,
            'status': status,
            'detections': self.current_detections
        }
        
        if self.classification_changed:
            self.results_log.append(result)
            logger.info(f"{timestamp} - {status}")
        
        return result
    
    def control_relay(self, turn_on):
        """Control ESP32 relay"""
        command = "RELAY_ON" if turn_on else "RELAY_OFF"
        self.send_serial_command(command)
    
    def capture_and_process_frame(self):
        """Capture frame and process for defects"""
        if self.camera is None:
            return None, []
        
        try:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                return None, []
            
            # Detect defects
            defect_found, annotated_frame, detections = self.detect_defects(frame)
            self.defect_detected = defect_found
            
            return annotated_frame, detections
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None, []
    
    def frame_to_base64(self, frame):
        """Convert frame to base64 for web transmission"""
        if frame is None:
            return None
        
        try:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                # Convert to base64
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                return img_base64
            return None
        except Exception as e:
            logger.error(f"Error converting frame to base64: {e}")
            return None
    
    async def websocket_handler(self, websocket):
        """Handle WebSocket connections"""
        self.connected_clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.connected_clients)}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    command = data.get('command')
                    
                    if command == 'reset':
                        await self.handle_reset_command()
                    elif command == 'save':
                        await self.handle_save_command()
                    elif command == 'test_relay':
                        await self.handle_test_relay()
                    
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received from client")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            self.connected_clients.discard(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.connected_clients)}")
    
    async def handle_reset_command(self):
        """Handle system reset command"""
        self.send_serial_command("RESET")
        
        # Reset internal state
        self.ir_detected = False
        self.defect_detected = False
        self.relay_status = True
        self.current_detections = []
        
        logger.info("System reset command executed")
    
    async def handle_save_command(self):
        """Handle save results command"""
        self.save_results()
    
    async def handle_test_relay(self):
        """Handle relay test command"""
        # Toggle relay for test
        self.send_serial_command("RELAY_OFF")
        await asyncio.sleep(1)
        self.send_serial_command("RELAY_ON")
        logger.info("Relay test executed")
    
    async def broadcast_data(self):
        """Broadcast current system status to all connected clients"""
        if not self.connected_clients:
            return
        
        # Capture and process current frame
        frame, detections = self.capture_and_process_frame()
        
        # Classify current state
        result = self.classify_unit(self.ir_detected, self.defect_detected)
        
        # Prepare data for web interface
        data = {
            'ir_detected': self.ir_detected,
            'defect_detected': self.defect_detected,
            'relay_status': self.relay_status,
            'classification': result['classification'],
            'classification_changed': self.classification_changed,
            'detections': detections,
            'image_data': self.frame_to_base64(frame),
            'timestamp': result['timestamp'],
            'serial_connected': self.serial_connection is not None
        }
        
        # Send to all connected clients
        if self.connected_clients:
            message = json.dumps(data)
            disconnected_clients = []
            
            for client in self.connected_clients:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.append(client)
                except Exception as e:
                    logger.error(f"Error sending data to client: {e}")
                    disconnected_clients.append(client)
            
            # Remove disconnected clients
            for client in disconnected_clients:
                self.connected_clients.discard(client)
    
    def save_results(self, filename=None):
        """Save results to file"""
        if filename is None:
            filename = f"quality_control_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Quality Control System Log\n")
                f.write("="*50 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Summary statistics
                total_units = len(self.results_log)
                normal_count = sum(1 for r in self.results_log if r['classification'] == 'UNIT_NORMAL')
                defect_count = sum(1 for r in self.results_log if r['classification'] == 'UNIT_DEFECT')
                
                f.write("SUMMARY STATISTICS\n")
                f.write(f"Total Units Processed: {total_units}\n")
                f.write(f"Normal Units: {normal_count}\n")
                f.write(f"Defect Units: {defect_count}\n")
                if total_units > 0:
                    success_rate = (normal_count / total_units) * 100
                    f.write(f"Success Rate: {success_rate:.1f}%\n")
                f.write("\n" + "-"*50 + "\n\n")
                
                # Detailed log
                f.write("DETAILED LOG\n")
                for i, result in enumerate(self.results_log, 1):
                    f.write(f"{i}. {result['timestamp']}\n")
                    f.write(f"   IR Status: {'Detected' if result['ir_status'] else 'Not Detected'}\n")
                    f.write(f"   Defect Status: {'Found' if result['defect_status'] else 'None'}\n")
                    f.write(f"   Classification: {result['classification']}\n")
                    f.write(f"   Status: {result['status']}\n")
                    
                    if result['detections']:
                        f.write("   Detected Defects:\n")
                        for detection in result['detections']:
                            f.write(f"     - {detection['class']}: {detection['confidence']:.1%}\n")
                    f.write("\n")
            
            logger.info(f"✅ Results saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            return False
    
    async def run_async_loop(self):
        """Main async loop for broadcasting data"""
        while self.system_running:
            try:
                await self.broadcast_data()
                await asyncio.sleep(0.1)  # 10 FPS update rate
            except Exception as e:
                logger.error(f"Error in async loop: {e}")
                await asyncio.sleep(1)
    
    def run(self):
        """Start the complete system"""
        logger.info("\n🚀 Quality Control Web System Starting...")
        logger.info("="*50)
        
        # Check component initialization
        components_ok = True
        
        if not self.serial_connection:
            logger.error("❌ ESP32 not connected")
            components_ok = False
        else:
            logger.info("✅ ESP32 connected")
        
        if not self.camera:
            logger.error("❌ Camera not initialized")
            components_ok = False
        else:
            logger.info("✅ Camera initialized")
        
        if not self.yolo_model:
            logger.error("❌ YOLO model not loaded")
            components_ok = False
        else:
            logger.info("✅ YOLO model loaded")
        
        if not components_ok:
            logger.error("\n❌ System initialization failed. Cannot start.")
            logger.error("🔧 Please fix the issues above before running the system.")
            return False
        
        logger.info(f"\n✅ All components initialized successfully!")
        logger.info(f"🌐 WebSocket server will run on ws://localhost:{self.websocket_port}")
        logger.info("🖥️ Open the web interface in your browser")
        logger.info("="*50)
        
        self.system_running = True
        
        # Start IR sensor reading thread
        ir_thread = threading.Thread(target=self.read_ir_sensor, daemon=True)
        ir_thread.start()
        logger.info("🔄 IR sensor reading thread started")
        
        # Start WebSocket server and async loop
        async def main():
            try:
                # Start WebSocket server
                server = await websockets.serve(
                    self.websocket_handler,
                    "localhost",
                    self.websocket_port
                )
                logger.info(f"✅ WebSocket server started on port {self.websocket_port}")
                
                # Run the main async loop
                await self.run_async_loop()
                
            except Exception as e:
                logger.error(f"Error in async main: {e}")
        
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            logger.info("\n🛑 System interrupted by user")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up all resources"""
        logger.info("\n🧹 Cleaning up system resources...")
        
        self.system_running = False
        
        if self.camera:
            self.camera.release()
            logger.info("✅ Camera released")
        
        if self.serial_connection:
            try:
                self.send_serial_command("RESET")
                time.sleep(0.5)
                self.serial_connection.close()
                logger.info("✅ Serial connection closed")
            except:
                pass
        
        cv2.destroyAllWindows()
        logger.info("🧹 System cleanup completed")

def print_system_info():
    """Print system information and available ports"""
    print("🔧 Quality Control Web System")
    print("="*50)
    
    # Show available COM ports
    print("📡 Available COM Ports:")
    ports = serial.tools.list_ports.comports()
    if ports:
        for i, port in enumerate(ports, 1):
            print(f"  {i}. {port.device} - {port.description}")
            if any(keyword in port.description.lower() for keyword in ['usb', 'ch340', 'cp210', 'esp32', 'arduino']):
                print(f"     👆 This looks like an ESP32/Arduino device!")
    else:
        print("  ❌ No COM ports found!")
    print()

def main():
    """Main entry point"""
    print_system_info()
    
    # Configuration - SESUAIKAN DENGAN SETUP ANDA
    CONFIG = {
        'esp32_port': 'COM4',        # Port ESP32 (will auto-detect if enabled)
        'camera_id': 0,              # Camera ID (biasanya 0)
        'model_path': 'yolo8-1240-new.pt',  
        'baud_rate': 115200,         # Baud rate untuk serial
        'websocket_port': 8765,      # Port untuk WebSocket server
        'auto_detect_port': True     # Enable auto-detection of ESP32 port
    }
    
    print("⚙️ Configuration:")
    if CONFIG['auto_detect_port']:
        print(f"ESP32 Port: AUTO-DETECT (fallback: {CONFIG['esp32_port']})")
    else:
        print(f"ESP32 Port: {CONFIG['esp32_port']}")
    print(f"Camera ID: {CONFIG['camera_id']}")
    print(f"YOLO Model: {CONFIG['model_path']}")
    print(f"WebSocket Port: {CONFIG['websocket_port']}")
    print("="*50)
    
    # Ask user if they want to continue
    try:
        response = input("🚀 Start the system? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("👋 System startup cancelled")
            return False
    except KeyboardInterrupt:
        print("\n👋 System startup cancelled")
        return False
    
    # Initialize and run system
    system = QualityControlWebSystem(**CONFIG)
    success = system.run()
    
    if success:
        logger.info("✅ System completed successfully")
    else:
        logger.error("❌ System failed to start")
    
    return success

if __name__ == "__main__":
    main()