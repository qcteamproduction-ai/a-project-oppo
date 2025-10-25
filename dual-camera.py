import cv2
from ultralytics import YOLO
import threading
import queue

class DualCameraYOLO:
    def __init__(self, model_path, camera_ids=[0, 1]):
        """
        Initialize dual camera YOLO detection
        
        Args:
            model_path: Path to YOLO model file
            camera_ids: List of camera IDs to use
        """
        self.model = YOLO(model_path)
        self.camera_ids = camera_ids
        self.running = False
        self.frames = {cam_id: None for cam_id in camera_ids}
        self.frame_queues = {cam_id: queue.Queue(maxsize=2) for cam_id in camera_ids}
        
    def capture_frames(self, camera_id):
        """Capture frames from camera in separate thread"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print(f"Error: Tidak dapat membuka kamera {camera_id}")
            return
            
        print(f"Kamera {camera_id} berhasil dibuka")
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Clear old frames and add new one
                if self.frame_queues[camera_id].full():
                    try:
                        self.frame_queues[camera_id].get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queues[camera_id].put(frame)
            else:
                print(f"Error membaca frame dari kamera {camera_id}")
                break
                
        cap.release()
        print(f"Kamera {camera_id} ditutup")
    
    def process_camera(self, camera_id):
        """Process frames from camera with YOLO detection"""
        window_name = f'Camera {camera_id} - YOLO Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while self.running:
            try:
                frame = self.frame_queues[camera_id].get(timeout=1)
                
                # Run YOLO detection
                results = self.model(frame, verbose=False)
                
                # Draw results on frame
                annotated_frame = results[0].plot()
                
                # Add camera info
                cv2.putText(annotated_frame, f'Camera {camera_id}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow(window_name, annotated_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing camera {camera_id}: {e}")
                break
        
        cv2.destroyWindow(window_name)
    
    def run(self):
        """Start dual camera detection"""
        self.running = True
        threads = []
        
        # Start capture threads
        for cam_id in self.camera_ids:
            t = threading.Thread(target=self.capture_frames, args=(cam_id,))
            t.daemon = True
            t.start()
            threads.append(t)
        
        # Start processing threads
        for cam_id in self.camera_ids:
            t = threading.Thread(target=self.process_camera, args=(cam_id,))
            t.daemon = True
            t.start()
            threads.append(t)
        
        print("Dual camera detection berjalan...")
        print("Tekan 'q' pada window mana saja untuk keluar")
        
        # Wait for threads
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\nMenghentikan deteksi...")
            self.running = False
        
        cv2.destroyAllWindows()
        print("Program selesai")

def main():
    # Path ke model YOLO
    model_path = "x-ray-model.pt"
    
    # ID kamera yang akan digunakan
    camera_ids = [0, 1]
    
    # Buat instance dan jalankan
    detector = DualCameraYOLO(model_path, camera_ids)
    detector.run()

if __name__ == "__main__":
    main()
