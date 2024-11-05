import os
import numpy as np
import pandas as pd
import torch

def load_data(csv_path, labels):
    """
    Hàm load dữ liệu đã xử lý từ file CSV.
    Arguments:
    - csv_path: Đường dẫn đến file CSV chứa dữ liệu đã xử lý.
    - labels: Ánh xạ các gesture labels thành số nguyên.

    Returns:
    - data: Tensor chứa các điểm đặc trưng của cử chỉ.
    - target: Tensor chứa nhãn tương ứng với dữ liệu.
    """

    data, target = [], []
    
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_path)
    points = df.iloc[:, :-1].values  # Bỏ cột label (cột cuối cùng)
    label_column = df.iloc[:, -1].values  # Cột label

    for i, label in enumerate(label_column):
        if label in labels:
            data.append(points[i])
            target.append(labels[label])

    # Chuyển đổi data và target thành Tensor
    data = torch.tensor(np.array(data), dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.long)
    return data, target

def initialize_dataframe(columns, csv_path):
    """
    Khởi tạo DataFrame hoặc đọc dữ liệu từ file CSV nếu đã tồn tại.
    """
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # If file exists but is empty or corrupted, create new DataFrame
            if df.empty or list(df.columns) != columns:
                return pd.DataFrame(columns=columns)
            return df
        else:
            return pd.DataFrame(columns=columns)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        # Handle empty or corrupted CSV files
        return pd.DataFrame(columns=columns)

def save_dataframe_to_csv(df, csv_path):
    """
    Lưu DataFrame vào file CSV.
    """
    df.to_csv(csv_path, index=False)