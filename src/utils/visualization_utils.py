import cv2

def draw_label(frame, gesture_name, hand_position):
    """
    Draws gesture label on the frame (left hoặc right).
    
    Arguments:
    - frame: Khung hình đang hiển thị.
    - gesture_name: Tên của cử chỉ cần hiển thị.
    - hand_position: Vị trí của tay (left hoặc right) để xác định vị trí vẽ nhãn.
    """
    if hand_position == 'left':
        position = (10, 30)
    else:
        position = (frame.shape[1] - 300, 30)
    cv2.putText(frame, f'Gesture: {gesture_name}', position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)