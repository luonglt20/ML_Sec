# 📋 BÁO CÁO PHÂN TÍCH DANH SÁCH CÁC CÂU LỖI (FAILED CASES ANALYSIS)

**Dataset:** MedQA USMLE Test Set ($N = 1,270$ câu)  
**Nguồn dữ liệu:** [`eval_results_full_1270_log.txt`](file:///Users/toilaluongg/Desktop/UIT%20-SDH/ML-Sec/GK/eval_results_full_1270_log.txt)

---

## 1. TỔNG QUAN PHÂN LOẠI CÂU LỖI

| Nhóm Ca Lỗi | Số lượng | Nguyên nhân Cốt lõi | Hướng Khắc phục của V3 |
|:---|:---:|:---|:---|
| **PieWrongs** *(Cả V0 và V3 đều sai)* | **68 câu** ($5.35\%$) | Câu hỏi y khoa USMLE Step 2/3 cực khó, kết hợp đa triệu chứng hiếm gặp hoặc đòi hỏi tính toán liều lượng dược lý phức tạp. | Cần mở rộng RAG Corpus chuyên sâu (UpToDate, Harrison's Internal Medicine). |
| **Losses** *(V0 đúng, V3 sai)* | **18 câu** ($1.41\%$) | Nhiễu ngữ cảnh (Context Noise) do đoạn văn RAG retrieved chứa thông tin đánh đố khiến Reasoner bị lệch hướng. | Đã khắc phục bằng **Verifier Agent High-Precision Rule** ở phiên bản tối ưu mới nhất. |

---

## 2. DANH SÁCH CÁC CA LỖI NỔI BẬT (SAMPLE FAILED CASES FROM LOG)

### 🔴 Ca Lỗi 1: Tác dụng phụ chống chỉ định Vòng tránh thai Vòng Đồng (Copper IUD)
* **Chủ đề Y khoa:** Dược lý & Bệnh học Phụ khoa (Gynecology / Pharmacology)
* **Từ khóa Truy vấn:** `copper IUD contraindications Wilson disease copper allergy pelvic infection`
* **Triệu chứng Lâm sàng:** Bệnh nhân nữ muốn đặt vòng tránh thai đồng nhưng có tiền sử bệnh lý tự miễn hoặc rối loạn chuyển hóa đồng (Bệnh Wilson).
* **Lý do Lỗi:** Đoạn văn RAG thô chưa quét trúng phần chống chỉ định tuyệt đối của Bệnh Wilson (Wilson's disease), dẫn đến việc chọn lầm giải pháp tránh thai nội tiết thay vì vòng tránh thai không đồng.

---

### 🔴 Ca Lỗi 2: Rối loạn Ngoại tháp & Tăng Prolactin do Risperidone
* **Chủ đề Y khoa:** Tâm thần học & Thuốc Chống loạn thần (Psychiatry / Antipsychotics)
* **Từ khóa Truy vấn:** `risperidone hyperprolactinemia extrapyramidal symptoms young male schizophrenia`
* **Triệu chứng Lâm sàng:** Bệnh nhân nam trẻ tuổi điều trị Tâm thần phân liệt xuất hiện chứng vú to ở nam giới (gynecomastia) và giảm ham muốn tình dục.
* **Lý do Lỗi:** Mô hình V0 chọn nhầm tác dụng phụ tăng cân của Olanzapine, trong khi V3 do bị trích dẫn RAG tổng quát nên chưa tách biệt rõ ràng mức độ gây tăng Prolactin máu cao nhất của Risperidone so với các thuốc chống loạn thần thế hệ 2 khác.

---

### 🔴 Ca Lỗi 3: Ngộ độc Tai & Tổn thương Thính giác do Cisplatin
* **Chủ đề Y khoa:** Ung thư học & Thuốc Hóa trị (Oncology / Chemotherapy Toxicity)
* **Từ khóa Truy vấn:** `cisplatin ototoxicity DNA cross-linking sensorineural hearing loss bladder cancer`
* **Triệu chứng Lâm sàng:** Bệnh nhân ung thư bàng quang sau điều trị hóa trị bằng Cisplatin xuất hiện ù tai và giảm thính lực tiếp nhận 2 bên (sensorineural hearing loss).
* **Lý do Lỗi:** Cả 2 mô hình đều phân vân giữa cơ chế gây độc cho thận (Nephrotoxicity) và ngộ độc tai (Ototoxicity) do thiếu thông tin định lượng liều tích lũy của Cisplatin.

---

### 🔴 Ca Lỗi 4: Quy trình Chẩn đoán Xác nhận HIV ở Phụ nữ Mang thai
* **Chủ đề Y khoa:** Truyền nhiễm & Sản khoa (Infectious Disease / Obstetrics)
* **Từ khóa Truy vấn:** `HIV maternal confirmatory testing Western blot immunofluorescence assay pregnant woman`
* **Triệu chứng Lâm sàng:** Phụ nữ mang thai có xét nghiệm ELISA HIV dương tính lần đầu, cần bước xét nghiệm chẩn đoán xác định tiếp theo.
* **Lý do Lỗi:** Sự thay đổi giữa hướng dẫn cũ (Western blot) và hướng dẫn CDC mới (HIV-1/HIV-2 differentiation immunoassay & Nucleic Acid Test RNA) khiến cả V0 và V3 bị xung đột giữa các mốc thời gian kiến thức.

---

### 🔴 Ca Lỗi 5: Độc tính Gan & Thần kinh của Galantamine trong Điều trị Alzheimer
* **Chủ đề Y khoa:** Thần kinh học & Dược lý Thuốc Ức chế Cholinesterase (Neurology / Pharmacology)
* **Từ khóa Truy vấn:** `galantamine cholinergic toxicity diarrhea vomiting management antimuscarinic atropine`
* **Triệu chứng Lâm sàng:** Bệnh nhân Alzheimer dùng quá liều Galantamine bị tiêu chảy, nôn mửa, tụt huyết áp và chậm nhịp tim.
* **Lý do Lỗi:** V0 nhầm lẫn giữa xử trí bằng Atropine (độc tính muscarinic) và Pralidoxime, trong khi V3 bị RAG kéo sang hội chứng nôn chu kỳ (Cyclic vomiting syndrome).

---

### 🔴 Ca Lỗi 6: Biến chứng Bệnh Thận Đa Nang Di truyền Trỗi (ADPKD)
* **Chủ đề Y khoa:** Thận học & Di truyền học (Nephrology / Genetics)
* **Từ khóa Truy vấn:** `autosomal dominant polycystic kidney disease intracranial aneurysm screening MRA`
* **Triệu chứng Lâm sàng:** Bệnh nhân nam có thận đa nang kèm tiền sử gia đình có người vỡ túi mạch não (intracranial aneurysm).
* **Lý do Lỗi:** Cả V0 và V3 đều dự đoán đúng bệnh lý thận nhưng chọn sai chỉ định tầm soát mạch máu não (chọn CT thay vì MRA sọ não không dựng quang).

---

## 3. TỔNG KẾT & ĐỀ XUẤT NÂNG CẤP

1. **68 ca PieWrongs:** Đều là những câu hỏi đòi hỏi tri thức y khoa ngách hoặc quy trình chẩn đoán CDC/USMLE mới cập nhật.
2. **Giải pháp tối ưu:** Bổ sung **Hybrid Retrieval (BM25 + MedCPT Dense Retriever)** kết hợp **Reranker (BGE-Reranker-Large)** để đảm bảo các đoạn văn trích dẫn luôn chứa đúng 100% hướng dẫn chẩn đoán y khoa chuẩn xác.
