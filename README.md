# Multi-Modal Control System

Dự án Multi-Modal Control System là một hệ thống tích hợp nhiều phương thức tương tác và điều khiển (cử chỉ tay, giọng nói, hành động,...) để tạo ra một hệ thống điều khiển toàn diện cho máy tính hoặc các thiết bị khác. Hệ thống được thiết kế để dễ dàng mở rộng, cho phép tích hợp nhiều module nhận diện và điều khiển khác nhau, giúp người dùng tương tác với thiết bị mà không cần sử dụng chuột hay bàn phím truyền thống.

## Chức Năng Chính

- **Điều khiển bằng cử chỉ (Gesture Control)**:
  - Di chuyển chuột.
  - Click trái/phải, cuộn trang.
  - Thao tác hệ thống (mở ứng dụng, điều chỉnh âm lượng, tắt/mở màn hình).
  - Chuyển đổi cửa sổ.
  - Thao tác copy/paste.
- **Điều khiển bằng giọng nói (Voice Control)**:
  - Nhận diện và ánh xạ lệnh giọng nói thành hành động.
  - Thao tác hệ thống (mở/đóng ứng dụng, chuyển đổi cửa sổ).

## Phát Triển Module Chính

### Module Phát Hiện Cử Chỉ (Gesture Detection)

- Nhận diện các cử chỉ đặc trưng (VD: nắm tay, giơ ngón cái, mở bàn tay).
- Theo dõi vị trí bàn tay theo thời gian thực.

### Module Ánh Xạ Cử Chỉ -> Hành Động (Gesture-to-Action Mapping)

- Định nghĩa hành vi của hệ thống dựa trên cử chỉ (VD: nắm tay để di chuyển chuột, giơ ngón cái để click chuột).

### Module Điều Khiển Hệ Thống (System Control Module)

- Tích hợp với hệ điều hành để điều khiển chuột, bàn phím, và các chức năng hệ thống.

## Thiết Lập Pipeline

1. **Tiền xử lý dữ liệu camera**:

   - Lọc nhiễu, làm mịn hình ảnh.
   - Phát hiện và tách bàn tay.

2. **Nhận diện cử chỉ**:

   - Sử dụng thư viện như MediaPipe để thu thập điểm landmarks.
   - Tạo mô hình nhận diện cử chỉ tùy biến hoặc dùng sẵn mô hình pre-trained.

3. **Ánh xạ cử chỉ -> hành động**:
   - Viết logic để ánh xạ các cử chỉ được nhận diện thành hành động cụ thể (sử dụng một lớp ánh xạ để tiện quản lý).

## Công Cụ Và Thư Viện Chính

- **Python 3.7+**
- **PyTorch**: Thư viện học sâu để xây dựng và huấn luyện mô hình học máy.
- **NumPy**: Thư viện xử lý số liệu.
- **OpenCV**: Tiền xử lý hình ảnh và phát hiện đặc trưng.
- **MediaPipe**: Thu thập điểm landmarks của bàn tay.
- **Flask**: Framework để xây dựng API và giao diện người dùng.

2. Ghi các thư viện đã cài đặt vào requirements.txt
   ```bash
      python -m pip freeze > requirements.txt
   ```

## Cài Đặt

1. Tạo và kích hoạt môi trường ảo:

   ```bash
   python -m venv multimodal_env
   source multimodal_env/bin/activate  # Trên Windows: multimodal_env\Scripts\activate
   ```

2. Cài đặt các thư viện:
   ```bash
   pip install -r requirements.txt
   ```
