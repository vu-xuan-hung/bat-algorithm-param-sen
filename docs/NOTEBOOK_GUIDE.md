# Notebook: dữ liệu riêng, so sánh và độ nhạy

Mở [parameter_sensitivity.ipynb](../notebooks/parameter_sensitivity.ipynb) từ project root
hoặc `notebooks/`. Chọn Python `.venv`. Sửa DATASET_PATH, N_CLUSTERS, LABEL_COLUMN ở ô
cấu hình rồi Run All. Ví dụ CSV và quy ước cột nằm trong [datasets](../datasets/README.md).
Có thể nhập địa chỉ tuyệt đối; không bắt buộc chuyển file vào project.

Notebook gồm kiểm tra SSE/velocity toy, tải/chuẩn hóa dữ liệu, OFAT sáu tham số,
grid A0/r0, so sánh BA reference/K-Means, hội tụ, sample SD/mean CI, bảng paired
Wilcoxon/Holm, đếm accepted global/local, kiểm tra determinism/SSE/bounds, và hình cụm.
Hai feature đầu chỉ là hình chiếu khi D>2; với D=1 dùng chỉ số mẫu. Nhãn external chỉ
được dùng đánh giá. Tham số baseline, SSE, dấu velocity và random draws không đổi.

Mặc định P=12,T=24, seed kỹ thuật 0,1: 28 OFAT records +20 grid records; 11 PNG và
bảng được nhúng vào notebook. OFAT có 13 BA conditions; không gọi cấu hình tốt nhất là
kết luận. Hai seed không đủ cho xác nhận thống kê. K-Means dùng cùng processed data/K,
10 restarts theo config. Silhouette dùng toàn bộ dữ liệu; chi phí có thể lớn với CSV lớn.
Xem ngân sách in trước ô chạy và tăng quy mô có chủ ý. Không dùng seed formal để debug.

Mỗi lần chạy tạo `results/notebook_<UTC>_<random-id>/`, không ghi đè phiên trước.
Trong đó có notebook_settings.json, notebook_evidence.json, ofat/ và grid/ với
manifest/raw JSONL/CSV/summary/comparisons; PNG nằm trong figures/ của phiên/design.
Output hiện có trong notebook thuộc dataset/cấu hình được in trong chính notebook.

## Cách thực thi đã kiểm chứng

Môi trường không có Jupyter và tải nbformat/nbclient/ipykernel thất bại do DNS.
Vì vậy notebook được kiểm tra bằng **sequential Python cell runner**, không phải kernel
Jupyter. Runner thực thi code thật trong cùng namespace, thu stdout, bảng HTML và ảnh
PNG thật; dừng và lưu traceback nếu lỗi, xóa output cũ ở các cell chưa chạy để tránh
nhầm bằng chứng. Không hỗ trợ IPython magic. File output phải chưa tồn tại:

```bash
.venv/bin/python scripts/run_notebook.py --input notebooks/parameter_sensitivity.ipynb --output /tmp/ba-notebook-new.ipynb
```

Đã chạy toàn bộ 10 code cells trên dữ liệu mẫu và test CSV riêng một feature với cột nhãn
văn bản, K=2. Cả hai tạo 48 records và 11 ảnh, không lỗi. Hai ảnh main-effects và acceptance
đã được xem trực tiếp. Test còn kiểm tra từ chối ghi đè và giữ lỗi/xóa output stale.
Mở/Run All trong giao diện VS Code/Jupyter chưa được kiểm chứng trong môi trường này.

Nhóm dependency tùy chọn `.[notebooks]` khai báo ipykernel, nbformat, nbclient để cài
cho môi trường có mạng; cài mới chưa kiểm chứng do DNS, không cập nhật lock bằng phiên bản
đoán. Base requirements-lock.txt vẫn là snapshot runtime/dev cũ. Không cần nhóm này để
chạy sequential runner. Có thể truyền biến môi trường BA_NOTEBOOK_DATASET,
BA_NOTEBOOK_CLUSTERS, BA_NOTEBOOK_LABEL_COLUMN và BA_NOTEBOOK_OUTPUT_ROOT khi tự động hóa;
đường dẫn có khoảng trắng được hỗ trợ vì đọc trực tiếp bằng pathlib, không qua shell.

Sau khi notebook chạy thành công, bằng chứng chứng minh các assertion/pipeline trên các
input đã thử. Nó không chứng minh BA vượt K-Means, tìm tối ưu toàn cục, hay mọi dataset
đều có cùng độ nhạy. Nghiên cứu formal cần giao thức freeze trong EXPERIMENT_DESIGN.
