# Notebook: dữ liệu riêng, so sánh và độ nhạy

Mở [parameter_sensitivity.ipynb](../notebooks/parameter_sensitivity.ipynb) từ project root
hoặc `notebooks/`. Chọn Python `.venv`. Sửa DATASET_PATH, N_CLUSTERS, LABEL_COLUMN ở ô
cấu hình rồi Run All. Ví dụ CSV và quy ước cột nằm trong [datasets](../datasets/README.md).
Có thể nhập địa chỉ tuyệt đối; không bắt buộc chuyển file vào project.
Với Rice, Dry Bean, HTRU2 và Letter Recognition, xem lệnh tải và cấu hình tương ứng
trong [bảng dữ liệu UCI](../datasets/README.md#uci-benchmarks).

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

## Phiên kiểm tra ban đầu bằng sequential runner (lịch sử)

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


## Kiểm chứng mới: kernel Jupyter thực tế và cài đặt sạch

Sau khi mạng được mở, cài đặt sạch đã qua với lock gốc và .[notebooks]. Đã chạy notebook
qua nbclient với kernel ipykernel dùng chính Python của venv sạch: schema nbformat hợp lệ,
10 code cells, 48 records thành công, 11 inline PNGs, không lỗi; chỉ seed 0,1. Artifact
riêng: [notebook Jupyter](verification/parameter_sensitivity.jupyter.ipynb),
[audit](verification/jupyter_kernel_audit.json), [môi trường](verification/clean_install_verification.json).
Notebook gốc và output nghiên cứu được giữ nguyên. Chưa kiểm tra tương tác giao diện
VS Code; đó là phần acceptance thủ công còn lại, không phải thiếu kiểm chứng kernel.

Tạo venv và cài theo README, sau đó `.venv/bin/python -m pip install -e ".[notebooks]"`.
Để kiểm tra lại bằng kernel thật mà không ghi đè notebook/results cũ, dùng đoạn Python
bên dưới từ project root với interpreter của venv. Nó tạo kernelspec/output tạm riêng và
chạy code notebook thật, không dùng sequential runner:

```python
import json
import sys
import tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

root = Path.cwd()
work = Path(tempfile.mkdtemp(prefix="ba-jupyter-check-"))
kernel_dir = work / "kernels" / "ba-verify"
kernel_dir.mkdir(parents=True)
(kernel_dir / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "BA verified environment", "language": "python",
    "env": {"BA_NOTEBOOK_OUTPUT_ROOT": str(work / "results"),
            "MPLCONFIGDIR": str(work / "matplotlib"),
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
}))
manager = KernelManager(kernel_name="ba-verify",
    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(work / "kernels")]))
notebook = nbformat.read(root / "notebooks/parameter_sensitivity.ipynb", as_version=4)
nbformat.validate(notebook)
NotebookClient(notebook, km=manager, timeout=180, allow_errors=False,
    resources={"metadata": {"path": str(root)}}).execute()
nbformat.validate(notebook)
# Metadata inherited from the source refers to its historical sequential run.
notebook.metadata["execution_verification"] = {"runner": "real nbclient/ipykernel",
    "status": "passed", "note": "GUI not tested"}
with (work / "executed.ipynb").open("x") as handle:
    nbformat.write(notebook, handle)
print(work)
```

Đoạn này dành cho notebook mặc định; thay dataset/K/label theo hướng dẫn ở đầu khi cần.
Không dùng formal seed để kiểm tra môi trường. Cài dependency mới qua mạng đã được kiểm
chứng, nhưng không có offline wheel bundle hay bảo đảm mọi OS/Python version.
