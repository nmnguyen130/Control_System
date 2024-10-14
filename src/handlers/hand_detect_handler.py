import math
import cv2
import numpy as np
import mediapipe as mp

class HandDetectHandler:
    def __init__(self, static_mode=False, num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Khởi tạo HandDetectHandler với các tham số đã chỉ định.

        :param static_mode: Nếu True, phát hiện được thực hiện trên từng hình ảnh (chậm hơn).
        :param max_hands: Số lượng bàn tay tối đa để phát hiện.
        :param model_complexity: Độ phức tạp của mô hình đặc trưng bàn tay: 0 hoặc 1.
        :param detection_confidence: Ngưỡng độ tự tin phát hiện tối thiểu.
        :param min_tracking_confidence: Ngưỡng độ tự tin theo dõi tối thiểu.
        """
        self.static_mode = static_mode
        self.num_hands = num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]  # IDs cho các đầu ngón tay

    def find_hands(self, frame, isDraw=True):
        """
        Phát hiện bàn tay trong một frame.

        :param frame: Hình ảnh để tìm bàn tay trong đó.
        :param isDraw: Nếu True, vẽ kết quả lên hình ảnh.
        :return: Tuple chứa thông tin bàn tay được phát hiện và hình ảnh với hoặc không có vẽ.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        all_hands = []

        if self.results.multi_hand_landmarks:
            for hand_type, hand_lms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                hand_info = self._extract_hand_info(hand_lms, hand_type)
                all_hands.append(hand_info)

                if isDraw:
                    self._draw_hand(frame, hand_lms)

        return all_hands, frame
    
    def _extract_hand_info(self, hand_lms, hand_type):
        """Trích xuất các điểm đặc trưng bàn tay và thông tin hộp giới hạn."""

        lm_list = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]).flatten()

        # Xác định kiểu tay
        hand_type_label = hand_type.classification[0].label

        return {"lmList": lm_list, "type": hand_type_label}
    
    def _draw_hand(self, frame, hand_lms):
        """Vẽ các điểm đặc trưng bàn tay và hộp giới hạn lên hình ảnh."""
        self.mp_drawing.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

    def fingers_up(self, my_hand):
        """
        Đếm số ngón tay đang mở cho một bàn tay nhất định.

        :param my_hand: Dict chứa thông tin bàn tay.
        :return: Danh sách chỉ ra ngón tay nào đang mở.
        """
        fingers = []
        my_lm_list = my_hand["lmList"]

        # Phát hiện ngón cái
        if my_hand["type"] == "Right":
            fingers.append(1 if my_lm_list[self.tip_ids[0]][0] > my_lm_list[self.tip_ids[0] - 1][0] else 0)
        else:
            fingers.append(1 if my_lm_list[self.tip_ids[0]][0] < my_lm_list[self.tip_ids[0] - 1][0] else 0)

        # Phát hiện cho bốn ngón tay còn lại
        for id in range(1, 5):
            fingers.append(1 if my_lm_list[self.tip_ids[id]][1] < my_lm_list[self.tip_ids[id] - 2][1] else 0)

        return fingers