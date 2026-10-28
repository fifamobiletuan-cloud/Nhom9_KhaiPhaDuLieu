# ĐỒ ÁN MÔN HỌC: KHAI PHÁ DỮ LIỆU (DATA MINING)

## ĐỀ TÀI: KHAI PHÁ DỮ LIỆU VÀ PHÂN LOẠI CẢM XÚC ĐÁNH GIÁ SẢN PHẨM AMAZON THEO 5 MỨC SAO KẾT HỢP KHAI PHÁ KHÍA CẠNH VÀ CHẨN ĐOÁN VẤN ĐỀ SẢN PHẨM

*(Amazon Reviews Fine-Grained 5-Class Sentiment Analysis with Aspect-Based Product Diagnosis)*

---

### THÔNG TIN HỌC PHẦN VÀ NHÓM THỰC HIỆN

* **Cơ sở đào tạo:** Trường Đại học Thủy Lợi (Thuyloi University)
* **Khoa:** Công nghệ Thông tin
* **Bộ môn:** Khoa học Dữ liệu và Trí tuệ Nhân tạo
* **Môn học:** Khai phá Dữ liệu (Data Mining)
* **Học kỳ:** Học kỳ 1 — Năm học 2026 - 2027
* **Giảng viên hướng dẫn:** ThS. Vũ Thị Hạnh
* **Nhóm sinh viên thực hiện:** **Nhóm 9** (Lớp: S26-65CNTT)
  1. **Nguyễn Huỳnh Anh Tuấn** — MSSV: `2351067119`
  2. **Lê Văn Tính** — MSSV: `2351067117`
  3. **Nguyễn Anh Vũ** — MSSV: `2351067121`

---

### TRUY CẬP VÀ XEM TRỰC TIẾP TÀI LIỆU (ONLINE VIEWER - KHÔNG CẦN TẢI VỀ)

Quý Thầy/Cô và các bạn có thể bấm trực tiếp vào các liên kết dưới đây để **xem toàn bộ Báo cáo, Slide thuyết trình và Mã nguồn trực tiếp trên trình duyệt** mà không cần tải tệp về máy tính:

| Tài liệu | Định dạng | Xem trực tiếp trên Trình duyệt (1 Click) | Ghi chú |
| :--- | :---: | :--- | :--- |
| **Báo cáo Đồ án chi tiết** | **PDF** | 📄 [**Xem Báo cáo PDF trực tiếp trên GitHub**](BaoCao_KhaiPhaDuLieuNhom9.pdf) | Đầy đủ 68 trang, mục lục, bảng biểu và đồ thị sắc nét |
| **Báo cáo Đồ án gốc** | **DOCX** | 📘 [**Xem Báo cáo Word trên Microsoft Office Online**](https://view.officeapps.live.com/op/view.aspx?src=https://raw.githubusercontent.com/fifamobiletuan-cloud/Nhom9_KhaiPhaDuLieu/main/BaoCao_KhaiPhaDuLieuNhom9.docx) | Xem qua trình duyệt Microsoft Word Online chính chủ |
| **Slide Thuyết trình** | **PDF** | 📊 [**Xem Slide PDF trực tiếp trên GitHub**](Nhom9.pdf) | 29 slide thuyết trình đồ họa trực quan |
| **Slide Thuyết trình gốc** | **PPTX** | 📽️ [**Xem Slide PowerPoint trên Microsoft Office Online**](https://view.officeapps.live.com/op/view.aspx?src=https://raw.githubusercontent.com/fifamobiletuan-cloud/Nhom9_KhaiPhaDuLieu/main/Nhom9.pptx) | Trình chiếu slide trực tuyến qua Microsoft PowerPoint Online |
| **Mã nguồn Thực nghiệm** | **IPYNB** | 🐍 [**Xem Notebook Code trực tiếp trên GitHub**](AmazonReviews_Nhom9.ipynb) | 63 cell thực thi đầy đủ kèm kết quả và đồ thị tương tác |

---

---

### TỔNG QUAN VÀ KẾT QUẢ ĐẠT ĐƯỢC

Đồ án tập trung nghiên cứu, thiết kế và hiện thực hóa một hệ thống toàn diện theo chuẩn quy trình công nghiệp **CRISP-DM** nhằm giải quyết bài toán phân loại cảm xúc đa mức độ (5 mức sao) trên tập dữ liệu lớn **Amazon Reviews Full Score Dataset** (3.65 triệu bản ghi):

1. **Quản lý dữ liệu lớn & Tối ưu hóa bộ nhớ:**
   * Sử dụng **PyArrow Columnar Memory Engine** nạp 3.65 triệu dòng dữ liệu trong **5.1 giây** (nhanh gấp 9 lần so với Pandas C Engine).
   * Ứng dụng cấu trúc **Ma trận thưa nén CSR** (Compressed Sparse Row) của Scipy với độ thưa 99.89%, nén không gian 50,000 chiều của 600,000 bài nhận xét xuống chỉ còn **294 MB RAM** (thay vì 240 GB ở dạng mảng dày đặc).
   * Lấy mẫu phân tầng ngẫu nhiên (Stratified Sampling) 600,000 mẫu Train và 97,500 mẫu Test, bảo toàn xác suất tiên nghiệm $P(Y=k) = 0.20$ với sai số biên $MoE < 0.126\%$.

2. **Tiền xử lý văn bản sâu & Trích xuất đặc trưng hình thái:**
   * Xây dựng danh sách bảo toàn **10 từ phủ định cốt lõi** (`cannot`, `neither`, `never`, `no`, `nobody`, `none`, `nor`, `not`, `nothing`, `without`), khắc phục lỗi xóa mù quáng Stop Words.
   * Chuyển hóa dấu câu cảm thán thành token từ vựng (`!` thành `exclam`, `?` thành `quest`).
   * Trích xuất 7 đặc trưng hình thái văn bản và phát hiện quy luật chữ U của dấu cảm thán (1 sao và 5 sao dùng nhiều gấp 3 lần 3 sao).

3. **Khai phá dữ liệu không giám sát (Unsupervised Mining):**
   * **Phân tích chủ đề LSA (TruncatedSVD 100 chiều):** Nén không gian từ vựng và nhận diện 5 chủ đề cốt lõi (Sách, Phim, Nhạc, Thiết bị công nghệ, Nhận xét chung) trong 16.2 giây.
   * **Phân cụm MiniBatchKMeans:** Khảo sát hệ số Silhouette trên mẫu 10,000 quan sát và xác định số cụm tối ưu **k = 6** (đạt đỉnh cực đại $s = 0.0480$).
   * **Khai phá luật kết hợp (Association Rules):** Khai phá 27,575 luật kết hợp với độ đo **Lift** trong 0.7 giây. Luật mạnh nhất cho 1 sao là `{zero stars} -> {1 sao}` (Lift = 4.81, Confidence = 93.70%).
   * **Khám phá hiện tượng cụm từ phủ định (Negation Bigrams):** Phát hiện cụm từ `not bad` có tới **49.30% rơi vào 3 sao (trung lập)** và 18.91% ở 4 sao, khẳng định ưu thế vượt trội của N-grams so với Unigram.

4. **Học máy có giám sát & Tinh chỉnh siêu tham số:**
   * Tinh chỉnh siêu tham số điều chuẩn $\alpha$ trên tập kiểm định độc lập (120,000 train / 30,000 validation) cho 3 mô hình: **Complement Naive Bayes (CNB)**, **SGD Linear SVM**, và **SGD Logistic Regression**.
   * Đánh giá khách quan trên tập kiểm thử độc lập **97,500 bài đánh giá** (cân bằng 19,500 mẫu/lớp).

| Mô hình học máy | Accuracy (%) | Macro F1 (%) | Off-by-one Accuracy (%) | MAE (sao) | Thời gian Huấn luyện (s) | Tốc độ Suy luận |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Complement Naive Bayes (CNB)** | 50.26% | 47.99% | 84.26% | 0.738 sao | **0.30 s** | 1,500,000 mẫu/s |
| **SGD Linear SVM (Modified Huber)** | 52.10% | 49.27% | 86.83% | 0.667 sao | 11.75 s | 1,550,000 mẫu/s |
| **SGD Logistic Regression (Quán quân)** | **54.48%** | **53.80%** | **89.14%** | **0.612 sao** | 12.33 s | **1,600,000 mẫu/s** |

> *Ghi chú:* Độ chính xác **54.48%** cao gấp **2.72 lần** so với Baseline đoán ngẫu nhiên (20.0%), tiệm cận kết quả công bố của Yann LeCun tại NeurIPS 2015 (54.64%). Đặc biệt, chỉ số **Off-by-one đạt 89.14%** (gần 90% dự đoán đúng hoặc chỉ lệch 1 sao) và **MAE chỉ 0.612 sao**.

5. **Hệ thống ứng dụng ABSA & Chẩn đoán vấn đề sản phẩm:**
   * Xây dựng từ điển **9 nhóm khía cạnh sản phẩm** (Giao hàng, Chất lượng, Độ bền, Hoạt động kém, Giá cả, Đúng mô tả, Kích cỡ, Dịch vụ khách hàng, Nội dung).
   * Tự động tính toán sao trung bình, khoảng tin cậy Bootstrap 95%, bóc tách nguyên nhân bị chê và đề xuất khuyến nghị hành động kinh doanh cụ thể.
   * **Kiểm chứng khoa học trên 500 sản phẩm mô phỏng (25,000 bài nhận xét):** Hệ số tương quan Pearson giữa sao thật và sao dự đoán đạt **r = 0.978**, sai số MAE cấp sản phẩm chỉ **0.178 sao**, và độ chính xác phân loại sản phẩm Tốt / Không tốt đạt **95.4% - 99.7%**.

---

### CẤU TRÚC THƯ MỤC DỰ ÁN

```text
├── AmazonReviews_Nhom9.ipynb        # Jupyter Notebook hoàn chỉnh (63 cell thực thi đầy đủ)
├── BaoCao_KhaiPhaDuLieuNhom9.docx   # Báo cáo đồ án chi tiết Word (~100 trang, 30,258 từ, 23 bảng)
├── BaoCao_KhaiPhaDuLieuNhom9.pdf    # Báo cáo đồ án định dạng PDF (xem trực tiếp trên GitHub)
├── Nhom9.pptx                       # Slide trình chiếu bảo vệ đồ án PowerPoint
├── Nhom9.pdf                        # Slide trình chiếu định dạng PDF (xem trực tiếp trên GitHub)
├── figures/                         # Thư mục đồ thị tổng quan của đề tài
│   ├── fig1_class_distribution.png
│   ├── fig2_linguistic_features.png
│   ├── fig3_topic_modeling.png
│   ├── fig4_model_comparison.png
│   ├── fig5_confusion_matrix.png
│   └── fig6_top_words_importance.png
├── figures_extracted/               # 13 đồ thị thực nghiệm trích xuất trực tiếp từ Notebook
│   ├── cell_6_img_2.png             # Phân phối nhãn dữ liệu Train & Test
│   ├── cell_17_img_2.png            # Tương quan đặc trưng hình thái theo số sao
│   ├── cell_22_img_1.png            # Phân tích 5 cụm chủ đề LSA
│   ├── cell_24_img_2.png            # Biểu đồ Silhouette chọn k=6 cụm KMeans
│   ├── cell_28_img_0.png            # Biểu đồ độ nâng Lift cho 1 sao và 5 sao
│   ├── cell_32_img_5.png            # Đường cong điều chuẩn siêu tham số alpha
│   ├── cell_36_img_2.png            # So sánh hiệu năng 3 mô hình học máy
│   ├── cell_37_img_1.png            # Ma trận nhầm lẫn số lượng và chuẩn hóa %
│   ├── cell_38_img_0.png            # Trọng số từ khóa quyết định của Logistic Regression
│   ├── cell_53_img_1.png            # Chẩn đoán tự động Sản phẩm B (Lỗi giao hàng)
│   ├── cell_54_img_1.png            # Chẩn đoán tự động Sản phẩm C (Lỗi chất lượng)
│   ├── cell_57_img_1.png            # Chẩn đoán sản phẩm thực tế "Tai nghe không dây XYZ"
│   └── cell_59_img_1.png            # Kiểm chứng tương quan Pearson trên 500 sản phẩm giả lập
├── amazon_review_fine-grained_5_classes_csv/
│   └── readme.txt                   # Mô tả nguồn gốc và hướng dẫn tải tập dữ liệu gốc
├── .gitignore                       # Bỏ qua các tệp CSV kích thước lớn (>100MB) theo quy định GitHub
└── README.md                        # Tài liệu hướng dẫn tổng quan của kho lưu trữ
```

---

### HƯỚNG DẪN CÀI ĐẶT VÀ TÁI LẬP THỰC NGHIỆM (REPRODUCIBILITY GUIDE)

#### 1. Yêu cầu môi trường
* Hệ điều hành: Windows 10/11, Linux Ubuntu hoặc macOS
* Python 3.9 trở lên
* Bộ nhớ RAM: Tối thiểu 8 GB (Khuyến nghị 16 GB)

#### 2. Cài đặt các thư viện phụ thuộc
Mở terminal hoặc command prompt và cài đặt các thư viện cần thiết:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn pyarrow joblib jupyter
```

#### 3. Chuẩn bị tập dữ liệu
Tải tập dữ liệu chuẩn **Amazon Review Full Score Dataset** từ Kaggle hoặc kho dữ liệu mở:
* Đặt 2 tệp `train.csv` và `test.csv` vào thư mục: `amazon_review_fine-grained_5_classes_csv/`

#### 4. Chạy thực nghiệm
Khởi động Jupyter Notebook và mở tệp `AmazonReviews_Nhom9.ipynb`:
```bash
jupyter notebook AmazonReviews_Nhom9.ipynb
```
Chọn **Kernel -> Restart & Run All** để tái lập lại toàn bộ các kết quả và biểu đồ thực nghiệm.

---

### LIÊN HỆ VÀ BẢN QUYỀN

Đồ án được thực hiện phục vụ mục đích học tập và nghiên cứu khoa học thuộc khuôn khổ học phần **Khai phá Dữ liệu** tại **Trường Đại học Thủy Lợi**. Mọi ý kiến đóng góp hoặc thắc mắc vui lòng liên hệ nhóm sinh viên thực hiện qua email:
* Email liên hệ: `fifamobiletuan@gmail.com`
