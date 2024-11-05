import cv2
import numpy as np
import mediapipe as mp

class HandDetectHandler:
    def __init__(self, static_mode=False, num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize HandDetectHandler with specified parameters.

        Args:
            static_mode (bool): If True, detection is performed on each image (slower)
            num_hands (int): Maximum number of hands to detect
            min_detection_confidence (float): Minimum detection confidence threshold
            min_tracking_confidence (float): Minimum tracking confidence threshold
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
        
        # Finger tip landmark indices
        self.tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    def find_hands(self, frame, isDraw=True):
        """
        Detect hands in a frame.

        Args:
            frame: Image frame to detect hands in
            isDraw (bool): If True, draw landmarks and connections on the frame

        Returns:
            tuple: (List of detected hands info, Processed frame)
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
        """
        Extract hand landmarks and information.
        
        Args:
            hand_lms: Hand landmarks from MediaPipe
            hand_type: Hand type classification from MediaPipe
            
        Returns:
            dict: Hand information including landmarks and hand type
        """
        # Convert landmarks to numpy array and flatten
        lm_list = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]).flatten()
        hand_type_label = hand_type.classification[0].label

        return {"lmList": lm_list, "type": hand_type_label}
    
    def _draw_hand(self, frame, hand_lms):
        """
        Draw hand landmarks and connections on frame.
        
        Args:
            frame: Image frame to draw on
            hand_lms: Hand landmarks to draw
        """
        self.mp_drawing.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

    def fingers_up(self, my_hand):
        """
        Count number of raised fingers for a given hand.

        Args:
            my_hand (dict): Hand information containing landmarks and type

        Returns:
            list: Binary list indicating raised fingers [thumb, index, middle, ring, pinky]
        """
        fingers = []
        my_lm_list = my_hand["lmList"]

        # Check thumb based on hand type (different logic for left/right hand)
        if my_hand["type"] == "Right":
            fingers.append(1 if my_lm_list[self.tip_ids[0]][0] > my_lm_list[self.tip_ids[0] - 1][0] else 0)
        else:
            fingers.append(1 if my_lm_list[self.tip_ids[0]][0] < my_lm_list[self.tip_ids[0] - 1][0] else 0)

        # Check other fingers by comparing y coordinates of tip and second joint
        for id in range(1, 5):
            fingers.append(1 if my_lm_list[self.tip_ids[id]][1] < my_lm_list[self.tip_ids[id] - 2][1] else 0)

        return fingers