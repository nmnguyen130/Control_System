import cv2

class CameraService:
    def __init__(self):
        self.cap = self.initialize_video_capture(0)

    def initialize_video_capture(self, camera_id=0):
        """
        Khởi tạo và kiểm tra camera từ camera_id
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print(f"Error: Could not open video device with index {camera_id}.")
            exit()
        return cap

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = self.preprocess_frame(frame)
        return frame
    
    def preprocess_frame(self, frame):
        """
        Tiền xử lý khung hình: lọc nhiễu và làm mịn hình ảnh.
        """
        frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()