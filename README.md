# Multi-Modal Control System

Dự án Multi-Modal Control System là một hệ thống tích hợp nhiều phương thức tương tác và điều khiển (cử chỉ tay, giọng nói, hành động,...) để tạo ra một hệ thống điều khiển toàn diện cho máy tính. Hệ thống được thiết kế để dễ dàng mở rộng, cho phép tích hợp nhiều module nhận diện và điều khiển khác nhau, giúp người dùng tương tác với thiết bị mà không cần sử dụng chuột hay bàn phím truyền thống.

## Chức Năng Chính

- **Điều khiển bằng Cử Chỉ (Gesture Control)**:
  - Di chuyển chuột.
  - Click trái/phải, cuộn trang.
  - Chuyển đổi cửa sổ.
- **Điều khiển bằng Giọng Nói (Voice Control)**:
  - Nhận diện và ánh xạ lệnh giọng nói thành hành động.
  - Phân tích ngữ nghĩa.
  - Thao tác hệ thống (mở/đóng ứng dụng, chuyển đổi cửa sổ).

## Phát Triển Module Chính

### Module Phát Hiện Cử Chỉ (Gesture Detection)

- Nhận diện các cử chỉ đặc trưng (VD: nắm tay, giơ ngón cái, mở bàn tay).
- Theo dõi vị trí bàn tay theo thời gian thực.
- Định nghĩa hành vi của hệ thống dựa trên cử chỉ (VD: nắm tay để di chuyển chuột, giơ ngón trỏ để click chuột...).

### Module Nhận Diện Giọng Nói (Voice Recognition Module)

- Nhận tín hiệu âm thanh: Ghi âm và xử lý tín hiệu âm thanh từ microphone.
- Mô hình học sâu: Sử dụng mô hình học sâu (như RNN hoặc Transformer) để nhận diện lệnh giọng nói từ âm thanh.
- Phân loại lệnh: Xây dựng logic để phân loại các lệnh và ánh xạ chúng đến các chức năng cụ thể trong hệ thống.
- Tương tác người dùng: Cung cấp hướng dẫn và thông báo bằng giọng nói cho người dùng về các thao tác và trạng thái của hệ thống.

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
- **SpeechRecognition**: Thư viện để nhận diện giọng nói từ âm thanh.

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

## Chạy chương trình

Để chạy ứng dụng, hãy thực hiện lệnh sau trong terminal:

```bash
python -m run
```

Lệnh này sẽ hiển thị một menu để chọn mô-đun cần chạy, bao gồm các tùy chọn cho nhận dạng cử chỉ, thu thập giọng nói và nhiều hơn nữa.

## Cấu Trúc Thư Mục Dự Án Multi-Modal Control System

control_system/
│
├── multimodal_env/ # Thư mục môi trường ảo
│
├── data/ # Thư mục dataset
│
├── trained_data/ # Thư mục chứa dữ liệu đã huấn luyện
│
├── src/ # Thư mục mã nguồn
│ ├── **init**.py
│ ├── main.py
│ ├── config/ # Thư mục cấu hình
│ │ ├── **init**.py
│ │
│ ├── core/ # Các thành phần chính của hệ thống
│ │ ├── **init**.py
│ │
│ ├── handlers/ # Xử lý các sự kiện và yêu cầu
│ │ ├── **init**.py
│ │
│ ├── modules/ # Các module chức năng
│ │ ├── gesture_detection/ # Module phát hiện cử chỉ
│ │ │ ├── **init**.py
│ │ │
│ │ ├── voice_recognition/ # Module nhận diện giọng nói
│ │ │ ├── **init**.py
│ │ │
│ │ ├── system_control/ # Module điều khiển hệ thống
│ │ │ ├── **init**.py
| |
│ ├── services/ # Dịch vụ hỗ trợ cho hệ thống
│ │ ├── **init**.py
│ │ ├── settings.py # Cấu hình cho hệ thống
│ │
│ ├── utils/ # Tiện ích hỗ trợ
│ │ ├── **init**.py
│
├── requirements.txt # Danh sách các thư viện cần thiết
├── README.md # Tài liệu dự án
└── .gitignore # Các tệp không cần theo dõi
