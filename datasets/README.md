# Dữ liệu cho notebook

Đặt CSV tại đây hoặc giữ file ở chỗ khác và nhập đường dẫn tuyệt đối trong
[notebook](../notebooks/parameter_sensitivity.ipynb). CSV phải có header; các feature
phải là số hữu hạn, không NaN. Thư mục này mặc định bỏ qua dữ liệu người dùng khi Git add;
chỉ dữ liệu mẫu tổng hợp và tài liệu được cho phép đưa vào Git.

Trong ô cấu hình notebook:

```python
DATASET_PATH = "/home/your-name/data/my_dataset.csv"
N_CLUSTERS = 3
LABEL_COLUMN = None  # hoặc "species", "class", ... để tách nhãn khỏi features
```

Đường dẫn tương đối tính từ `ba-clustering/`, ví dụ `datasets/my_dataset.csv`.
Chọn K phù hợp bài toán; notebook không tự suy ra số cụm tối ưu. Loại cột ID trước
khi chạy. Nếu nhãn là số cũng phải chỉ rõ LABEL_COLUMN để tránh coi nhãn là feature.
Hiện hỗ trợ CSV; xuất Excel sang CSV trước. Đường dẫn URL không phải file local.
Sau khi đổi dataset, dùng Run All để các biểu đồ khớp dữ liệu mới.

`example_blobs.csv` gồm 90 mẫu, 2 feature, 3 cụm sinh từ `make_blobs`, std=0.8,
data seed=42. Không lưu nhãn sinh; metadata nằm trong `example_blobs.provenance.json`.
Đây là dữ liệu tổng hợp nhỏ để kiểm tra pipeline, không phải bằng chứng nghiên cứu chính thức.
