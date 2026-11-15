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

## UCI benchmarks

From the repository root, run `python scripts/fetch_uci_datasets.py` once. The script
saves UCI's headered CSVs byte for byte and checks their SHA-256 hashes. Running it again
verifies the existing files. The CSVs are local and ignored by Git.

| UCI dataset | CLI name / CSV stem | Rows | Features | K | Label column |
|---|---|---:|---:|---:|---|
| [Rice (Cammeo and Osmancik)](https://doi.org/10.24432/C5MW4Z) | `rice` | 3,810 | 7 | 2 | `Class` |
| [Dry Bean](https://doi.org/10.24432/C50S4B) | `dry_bean` | 13,611 | 16 | 7 | `Class` |
| [HTRU2](https://doi.org/10.24432/C5DK6R) | `htru2` | 17,898 | 8 | 2 | `class` |
| [Letter Recognition](https://doi.org/10.24432/C5ZP40) | `letter_recognition` | 20,000 | 16 | 26 | `lettr` |

HTRU2 retains its original class imbalance of 1,639 pulsar candidates and 16,259
negative examples. The configured K values match the class counts. Class labels are used
for adjusted Rand index (ARI). The optimizers receive the z-scored numeric features.

With the project environment active, run a short BA and K-Means check from the repository
root:

```sh
python -m ba_clustering.cli run --dataset rice --seed 0 --set population_size=3 --set max_iterations=1
```

For the notebook, set `DATASET_PATH` to `datasets/<stem>.csv` and use the matching
`N_CLUSTERS` and case-sensitive `LABEL_COLUMN` from the table. For example:

```python
DATASET_PATH = "datasets/htru2.csv"
N_CLUSTERS = 2
LABEL_COLUMN = "class"
```

Full-data silhouette computes pairwise distances, so runtime grows roughly quadratically
with the row count. A notebook sweep repeats that cost for each run.
