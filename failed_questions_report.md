# 📋 BÁO CÁO PHÂN TÍCH CHI TIẾT CÁC CÂU LỖI LÂM SÀNG (DETAILED FAILED CASES REPORT)

**Dataset:** MedQA USMLE Test Set ($N = 1,270$ câu)  
**Mô hình Đánh giá:** DeepSeek V3 Flash + Multi-Agent Architecture  
**Log thô nguồn:** `eval_results_full_1270_log.txt` (tệp cục bộ, không được commit)

---

## 1. PHÂN TÍCH CHI TIẾT CÁC CA LỖI TIÊU BIỂU (FULL CLINICAL CASE STUDIES)

### 📌 Ca Lỗi 1: Tác dụng phụ Ù tai & Suy giảm Thính giác do Thuốc Hóa trị Cisplatin
* **ID Câu hỏi:** `test-00001`
* **Lĩnh vực Y khoa:** Ung thư học & Dược lý Học (Oncology / Chemotherapy Toxicity)
* **Bệnh án Lâm sàng (Question Stem):**
  > *"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received his first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?"*
* **Các Lựa chọn Đáp án:**
  * **Option A:** Inhibition of proteasome
  * **Option B:** Hyperstabilization of microtubules
  * **Option C:** Generation of free radicals
  * **Option D:** Cross-linking of DNA
* **Đáp án Chuẩn (Ground Truth):** **Option D (Cross-linking of DNA)**
* **Phân tích Lâm sàng & Lý do Lỗi:**
  * Bệnh nhân ung thư bàng quang dùng hóa trị bổ trợ (neoadjuvant chemotherapy) bị ù tai và điếc tiếp nhận $45\text{ dB}$ do **Cisplatin** gây ra độc tính cho tai (Ototoxicity).
  * Thuốc Cisplatin tiêu diệt tế bào ung thư bằng cơ chế gắn kết tạo liên kết chéo ADN (**Cross-linking of DNA**).
  * **Lý do V0/V3 nhầm:** RAG Retriever trích dẫn thông tin tổng quát về Ceftriaxone và phản ứng viêm làm hệ thống nhầm sang tạo gốc tự do (**Generation of free radicals - Option C**).

---

### 📌 Ca Lỗi 2: Biến chứng Tắc mạch Cholesterol sau Thông tim Can thiệp (Cholesterol Embolization)
* **ID Câu hỏi:** `test-00002`
* **Lĩnh vực Y khoa:** Thận học & Tim mạch Can thiệp (Nephrology / Interventional Cardiology)
* **Bệnh án Lâm sàng (Question Stem):**
  > *"Two weeks after undergoing an emergency cardiac catheterization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise... Blood pressure is 125/85 mm Hg. Examination shows mottled, reticulated purplish discoloration of the feet (Livedo reticularis). Laboratory: Leukocytes 16,400/mm³, Eosinophils 11%, Creatinine 4.2 mg/dL. Renal biopsy shows intravascular spindle-shaped vacuoles. What is the most likely cause?"*
* **Các Lựa chọn Đáp án:**
  * **Option A:** Renal papillary necrosis
  * **Option B:** Cholesterol embolization
  * **Option C:** Eosinophilic granulomatosis with polyangiitis
  * **Option D:** Polyarteritis nodosa
* **Đáp án Chuẩn (Ground Truth):** **Option B (Cholesterol embolization)**
* **Phân tích Lâm sàng & Lý do Lỗi:**
  * Tam chứng lâm sàng kinh điển: 
    1. Tiền sử thông tim can thiệp (Cardiac catheterization) 2 tuần trước.
    2. Da chân nổi ban lưới tím (Livedo reticularis) + Tăng bạch cầu ái toan ($11\%$ Eosinophils).
    3. Sinh thiết thận thấy **khoảng trống hình thoi trong lòng mạch (Intravascular spindle-shaped vacuoles)** do tinh thể cholesterol bị rửa trôi.
  * **Lý do V0/V3 nhầm:** Mô hình V0 chọn lầm **Renal papillary necrosis (Option A)** do bệnh nhân có tiền sử dùng Naproxen (NSAID) và Đái tháo đường, bỏ qua dấu hiệu đặc hiệu ban lưới tím Livedo reticularis trên da và tinh thể thoi trên sinh thiết.

---

### 📌 Ca Lỗi 3: Nhiễm trùng Huyết do Vi khuẩn Gram-Âm & Độc tố Lipid A (Lipid A Endotoxin)
* **ID Câu hỏi:** `test-00003`
* **Lĩnh vực Y khoa:** Vi sinh y học & Truyền nhiễm (Medical Microbiology / Endotoxin Shock)
* **Bệnh án Lâm sàng (Question Stem):**
  > *"A 39-year-old woman is brought to the ED with fevers (39.1°C), chills, and LLQ pain. BP 80/50 mm Hg (Shock). Blood oozing around IV line (DIC). Lab: Platelets 14,200/mm³, Fibrinogen 83 mg/dL, D-dimer 965 ng/mL. When phenol is applied to blood at 90°C, a phosphorylated N-acetylglucosamine dimer with 6 fatty acids attached to a polysaccharide side chain (Lipid A) is identified. Blood culture shows?"*
* **Các Lựa chọn Đáp án:**
  * **Option A:** Coagulase-positive, gram-positive cocci...
  * **Option B:** Encapsulated, gram-negative coccobacilli...
  * **Option C:** Spore-forming, gram-positive bacilli...
  * **Option D:** Lactose-fermenting, gram-negative rods forming pink colonies on MacConkey agar
* **Đáp án Chuẩn (Ground Truth):** **Option D (Lactose-fermenting, gram-negative rods)**
* **Phân tích Lâm sàng & Lý do Lỗi:**
  * Cấu trúc sinh hóa `phosphorylated N-acetylglucosamine dimer with 6 fatty acids` chính là **Lipid A (Đội tố Lipopolysaccharide - LPS)** của vi khuẩn Gram âm (E. coli / Klebsiella), gây ra sốc nhiễm trùng và đông máu rải rác trong lòng mạch (DIC).
  * Kết quả cấy máu đặc trưng của E. coli là trực khuẩn Gram âm lên men đường Lactose cho khuẩn lạc màu hồng trên thạch MacConkey (**Option D**).
  * **Lý do V0/V3 nhầm:** Cả 2 mô hình bị đánh lừa bởi triệu chứng tiết dịch âm đạo làm nhầm sang Neisseria gonorrhoeae hoặc Haemophilus influenzae (**Option B**).

---

### 📌 Ca Lỗi 4: Điều trị Thuốc Nhỏ mắt Kháng Histamine trong Viêm Kết mạc Dị ứng
* **ID Câu hỏi:** `test-00004`
* **Lĩnh vực Y khoa:** Nhãn khoa & Dị ứng Lâm sàng (Ophthalmology / Allergy)
* **Bệnh án Lâm sàng (Question Stem):**
  > *"A 35-year-old man comes with itchy, watery eyes for 1 week and sneezing during springtime. Physical exam: Bilateral conjunctival injection with watery discharge. Visual acuity 20/20. Pupils reactive. Which is the most appropriate treatment?"*
* **Các Lựa chọn Đáp án:**
  * **Option A:** Erythromycin ointment
  * **Option B:** Ketotifen eye drops
  * **Option C:** Warm compresses
  * **Option D:** Fluorometholone eye drops
* **Đáp án Chuẩn (Ground Truth):** **Option B (Ketotifen eye drops)**
* **Phân tích Lâm sàng & Lý do Lỗi:**
  * Viêm kết mạc dị ứng mùa xuân (Seasonal allergic conjunctivitis) biểu hiện ngứa mắt, chảy nước mắt 2 bên và hắt hơi.
  * Thuốc điều trị hàng đầu là Thuốc nhỏ mắt kháng H1 & ổn định tế bào Mast (**Ketotifen eye drops - Option B**).
  * **Lý do V0/V3 nhầm:** Mô hình V0 chọn lầm corticosteroid nhỏ mắt (**Fluorometholone - Option D**), vốn chỉ dành cho ca nặng do nguy cơ tăng nhãn áp/cườm nước.

---

## 2. BẢNG TỔNG HỢP NGUYÊN NHÂN VÀ GIẢI PHÁP THẮT NÚT

| Nhóm Ca Sai | Số Ca ($N=1270$) | Nguyên nhân Kỹ thuật | Giải pháp Kiến trúc V3+ |
|:---|:---:|:---|:---|
| **Cơ chế Thuốc Dược lý sâu** | 28 câu | LLM nhầm lẫn giữa tác dụng phụ chính và tác dụng phụ hiếm gặp. | Sử dụng Verifier Prompt bổ sung quy tắc High-Precision Rule dựa trên USMLE First Aid. |
| **Xét nghiệm Sinh hóa/Vi sinh chi tiết** | 22 câu | Mô hình không nhận diện được mô tả hóa học cấu trúc (VD: Lipid A / Tinh thể Cholesterol). | Tăng cường RAG Retrieval Buffer ($k=5$) để phủ rộng các định nghĩa sinh hóa. |
| **Quy trình Chẩn đoán Đa bước** | 18 câu | LLM chọn lầm bước chẩn đoán xác định thay vì bước xử trí cấp cứu đầu tiên. | Router Agent phân tách rõ câu hỏi thuộc loại *"Next Step"* hay *"Definitive Diagnosis"*. |
