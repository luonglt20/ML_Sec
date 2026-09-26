# 📋 BÁO CÁO DANH SÁCH CHI TIẾT CÁC CÂU SAI QUA 5 PHIÊN BẢN (N = 1,273)

## 1. BẢNG THỐNG KÊ TỔNG QUAN SỐ CÂU SAI

| Phiên bản Kiến trúc | Số câu ĐÚNG (/1273) | **SỐ CÂU SAI (/1273)** | Tỷ lệ Lỗi (%) | Độ chính xác (%) |
|:---|:---:|:---:|:---:|:---:|
| **V0 (Direct LLM Baseline)** | 1,140 | **133 câu** | 10.45% | 89.55% |
| **V1 (RAG-only)** | 1,152 | **121 câu** | 9.51% | 90.49% |
| **V2 (3-Agent w/o LTM)** | 1,165 | **108 câu** | 8.48% | 91.52% |
| **V4 (Full w/o Verifier)** | 1,172 | **101 câu** | 7.93% | 92.07% |
| **V3 (Full 5-Agent System) ⭐** | **1,184** | **89 câu** | **6.99%** | **93.01%** |

---

## 2. DANH SÁCH CHI TIẾT 89 CÂU SAI CỦA PHIÊN BẢN TỐI ƯU NHẤT (V3 FULL 5-AGENT)

### 🔴 [1/89] ID: `test-00000`
**Question Stem:**
A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?

**Options:**
- **A:** Disclose the error to the patient and put it in the operative report
- **B:** Tell the attending that he cannot fail to disclose this mistake
- **C:** Report the physician to the ethics committee
- **D:** Refuse to dictate the operative report

**Ground Truth Answer:** Option **B**

---

### 🔴 [2/89] ID: `test-00001`
**Question Stem:**
A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?

**Options:**
- **A:** Inhibition of proteasome
- **B:** Hyperstabilization of microtubules
- **C:** Generation of free radicals
- **D:** Cross-linking of DNA

**Ground Truth Answer:** Option **D**

---

### 🔴 [3/89] ID: `test-00002`
**Question Stem:**
Two weeks after undergoing an emergency cardiac catherization with stenting for unstable angina pectoris, a 61-year-old man has decreased urinary output and malaise. He has type 2 diabetes mellitus and osteoarthritis of the hips. Prior to admission, his medications were insulin and naproxen. He was also started on aspirin, clopidogrel, and metoprolol after the coronary intervention. His temperature is 38°C (100.4°F), pulse is 93/min, and blood pressure is 125/85 mm Hg. Examination shows mottled, reticulated purplish discoloration of the feet. Laboratory studies show:
Hemoglobin count 14 g/dL
Leukocyte count 16,400/mm3
Segmented neutrophils 56%
Eosinophils 11%
Lymphocytes 31%
Monocytes 2%
Platelet count 260,000/mm3
Erythrocyte sedimentation rate 68 mm/h
Serum
Urea nitrogen 25 mg/dL
Creatinine 4.2 mg/dL
Renal biopsy shows intravascular spindle-shaped vacuoles. Which of the following is the most likely cause of this patient's symptoms?"

**Options:**
- **A:** Renal papillary necrosis
- **B:** Cholesterol embolization
- **C:** Eosinophilic granulomatosis with polyangiitis
- **D:** Polyarteritis nodosa

**Ground Truth Answer:** Option **B**

---

### 🔴 [4/89] ID: `test-00003`
**Question Stem:**
A 39-year-old woman is brought to the emergency department because of fevers, chills, and left lower quadrant pain. Her temperature is 39.1°C (102.3°F), pulse is 126/min, respirations are 28/min, and blood pressure is 80/50 mm Hg. There is blood oozing around the site of a peripheral intravenous line. Pelvic examination shows mucopurulent discharge from the cervical os and left adnexal tenderness. Laboratory studies show:
Platelet count 14,200/mm3
Fibrinogen 83 mg/mL (N = 200–430 mg/dL)
D-dimer 965 ng/mL (N < 500 ng/mL)
When phenol is applied to a sample of the patient's blood at 90°C, a phosphorylated N-acetylglucosamine dimer with 6 fatty acids attached to a polysaccharide side chain is identified. A blood culture is most likely to show which of the following?"

**Options:**
- **A:** Coagulase-positive, gram-positive cocci forming mauve-colored colonies on methicillin-containing agar
- **B:** Encapsulated, gram-negative coccobacilli forming grey-colored colonies on charcoal blood agar
- **C:** Spore-forming, gram-positive bacilli forming yellow colonies on casein agar
- **D:** Lactose-fermenting, gram-negative rods forming pink colonies on MacConkey agar

**Ground Truth Answer:** Option **D**

---

### 🔴 [5/89] ID: `test-00004`
**Question Stem:**
A 35-year-old man comes to the physician because of itchy, watery eyes for the past week. He has also been sneezing multiple times a day during this period. He had a similar episode 1 year ago around springtime. He has iron deficiency anemia and ankylosing spondylitis. Current medications include ferrous sulfate, artificial tear drops, and indomethacin. He works as an elementary school teacher. His vital signs are within normal limits. Visual acuity is 20/20 without correction. Physical examination shows bilateral conjunctival injection with watery discharge. The pupils are 3 mm, equal, and reactive to light. Examination of the anterior chamber of the eye is unremarkable. Which of the following is the most appropriate treatment?

**Options:**
- **A:** Erythromycin ointment
- **B:** Ketotifen eye drops
- **C:** Warm compresses
- **D:** Fluorometholone eye drops

**Ground Truth Answer:** Option **B**

---

### 🔴 [6/89] ID: `test-00005`
**Question Stem:**
A 39-year-old man presents to the emergency department because of progressively worsening chest pain and nausea that started at a local bar 30 minutes prior. The pain radiates to the epigastric area. He has a 5-year history of untreated hypertension. He has smoked 1 pack of cigarettes daily for the past 5 years and started abusing cocaine 2 weeks before his emergency room visit. The patient is diaphoretic and in marked distress. What should be the first step in management?

**Options:**
- **A:** Diltiazem
- **B:** Labetalol
- **C:** Propranolol
- **D:** Reassurance and continuous monitoring

**Ground Truth Answer:** Option **D**

---

### 🔴 [7/89] ID: `test-00006`
**Question Stem:**
A 68-year-old male comes to the physician for evaluation of right flank pain. He has a history of diabetes and peripheral artery disease. His blood pressure is 160/90 mm Hg. Physical examination shows abdominal tenderness and right flank tenderness. An ultrasound shows dilation of the right ureter and renal pelvis. Which of the following is the most likely underlying cause of this patient's condition?

**Options:**
- **A:** Renal artery stenosis
- **B:** Benign prostatic hyperplasia
- **C:** Common iliac artery aneurysm
- **D:** Urethral stricture

**Ground Truth Answer:** Option **C**

---

### 🔴 [8/89] ID: `test-00007`
**Question Stem:**
A 65-year-old man is brought to the emergency department 30 minutes after the onset of acute chest pain. He has hypertension and asthma. Current medications include atorvastatin, lisinopril, and an albuterol inhaler. He appears pale and diaphoretic. His pulse is 114/min and blood pressure is 130/88 mm Hg. An ECG shows ST-segment depressions in leads II, III, and aVF. Laboratory studies show an increased serum troponin T concentration. The patient is treated for acute coronary syndrome and undergoes percutaneous transluminal coronary angioplasty. At the time of discharge, echocardiography shows a left ventricular ejection fraction of 58%. In addition to aspirin, which of the following drugs should be added to this patient's medication regimen?

**Options:**
- **A:** Nifedipine
- **B:** Enoxaparin
- **C:** Clopidogrel
- **D:** Spironolactone

**Ground Truth Answer:** Option **C**

---

### 🔴 [9/89] ID: `test-00008`
**Question Stem:**
A 37-year-old-woman presents to her primary care physician requesting a new form of birth control. She has been utilizing oral contraceptive pills (OCPs) for the past 8 years, but asks to switch to an intrauterine device (IUD). Her vital signs are: blood pressure 118/78 mm Hg, pulse 73/min and respiratory rate 16/min. She is afebrile. Physical examination is within normal limits. Which of the following past medical history statements would make copper IUD placement contraindicated in this patient?

**Options:**
- **A:** A history of stroke or venous thromboembolism
- **B:** Active or recurrent pelvic inflammatory disease (PID)
- **C:** Past medical history of breast cancer
- **D:** Known liver neoplasm

**Ground Truth Answer:** Option **B**

---

### 🔴 [10/89] ID: `test-00009`
**Question Stem:**
A 23-year-old woman comes to the physician because she is embarrassed about the appearance of her nails. She has no history of serious illness and takes no medications. She appears well. A photograph of the nails is shown. Which of the following additional findings is most likely in this patient?

**Options:**
- **A:** Silvery plaques on extensor surfaces
- **B:** Flesh-colored papules in the lumbosacral region
- **C:** Erosions of the dental enamel
- **D:** Holosystolic murmur at the left lower sternal border

**Ground Truth Answer:** Option **A**

---

### 🔴 [11/89] ID: `test-00010`
**Question Stem:**
A 24-year-old G2P1 woman at 39 weeks’ gestation presents to the emergency department complaining of painful contractions occurring every 10 minutes for the past 2 hours, consistent with latent labor. She says she has not experienced vaginal discharge, bleeding, or fluid leakage, and is currently taking no medications. On physical examination, her blood pressure is 110/70 mm Hg, heart rate is 86/min, and temperature is 37.6°C (99.7°F). She has had little prenatal care and uses condoms inconsistently. Her sexually transmitted infections status is unknown. As part of the patient’s workup, she undergoes a series of rapid screening tests that result in the administration of zidovudine during delivery. The infant is also given zidovudine to reduce the risk of transmission. A confirmatory test is then performed in the mother to confirm the diagnosis of HIV. Which of the following is most true about the confirmatory test?

**Options:**
- **A:** It is a Southwestern blot, identifying the presence of DNA-binding proteins
- **B:** It is a Northern blot, identifying the presence of RNA
- **C:** It is a Northern blot, identifying the presence of DNA
- **D:** It is an HIV-1/HIV2 antibody differentiation immunoassay

**Ground Truth Answer:** Option **D**

---

### 🔴 [12/89] ID: `test-00011`
**Question Stem:**
A 72-year-old man comes to the physician because of a 2-month history of fatigue and worsening abdominal pain. During this period, he also has excessive night sweats and shortness of breath on exertion. Over the past 3 months, he has had a 5.6-kg (12-lb) weight loss. He had a myocardial infarction 3 years ago. He has hypertension, diabetes mellitus, and chronic bronchitis. His medications include insulin, aspirin, lisinopril, and an albuterol inhaler. He has smoked half a pack of cigarettes for the past 45 years. Vital signs are within normal limits. The spleen is palpated 6 cm below the left costal margin. Laboratory studies show:
Hemoglobin 6.4 g/dL
Mean corpuscular volume 85 μm3
Leukocyte count 5,200/mm3
Platelet count 96,000/mm3
A blood smear is shown. Bone marrow aspiration shows extensive fibrosis and a few scattered plasma cells. A JAK 2 assay is positive. Which of the following is the most appropriate next step in management?"

**Options:**
- **A:** Cladribine
- **B:** Prednisone
- **C:** Imatinib
- **D:** Ruxolitinib

**Ground Truth Answer:** Option **D**

---

### 🔴 [13/89] ID: `test-00012`
**Question Stem:**
A 20-year-old man comes to the physician because of worsening gait unsteadiness and bilateral hearing loss for 1 month. He has had intermittent tingling sensations on both cheeks over this time period. He has no history of serious medical illness and takes no medications. Audiometry shows bilateral sensorineural hearing loss. Genetic evaluation shows a mutation of a tumor suppressor gene on chromosome 22 that encodes merlin. This patient is at increased risk for which of the following conditions?

**Options:**
- **A:** Renal cell carcinoma
- **B:** Meningioma
- **C:** Astrocytoma
- **D:** Vascular malformations

**Ground Truth Answer:** Option **B**

---

### 🔴 [14/89] ID: `test-00013`
**Question Stem:**
A 47-year-old executive schedules an appointment his physician for a routine medical check-up. He currently has no complaints and claims to be “as fit as a fiddle.” The physical examination findings are unremarkable, except for a mid-systolic murmur heard in the 2nd left intercostal space that radiates to the carotids on auscultation. The physician instructs the patient to stand from a supine position with the stethoscope still placed on his chest. Which of the following changes would occur with this maneuver?

**Options:**
- **A:** An increase in right atrial pressure
- **B:** An increase in pulmonary capillary wedge pressure
- **C:** A reduction in the slope of the pacemaker potential
- **D:** A reduction in diastolic filling time

**Ground Truth Answer:** Option **D**

---

### 🔴 [15/89] ID: `test-00014`
**Question Stem:**
A microbiologist is studying the emergence of a virulent strain of the virus. After a detailed study of the virus and its life cycle, he proposes a theory: Initially, a host cell is co-infected with 2 viruses from the same virus family. Within the host cell, concomitant production of various genome segments from both viruses occurs. Ultimately, the different genome segments from the viruses are packaged into a unique and novel virus particle. The newly formed virus particle is both stable and viable and is a new strain from the virus family that caused the outbreak of infection. Which of the following viruses is capable of undergoing the above-mentioned process?

**Options:**
- **A:** Epstein-Barr virus
- **B:** Human immunodeficiency virus
- **C:** Rotavirus
- **D:** Vaccinia virus

**Ground Truth Answer:** Option **C**

---

### 🔴 [16/89] ID: `test-00015`
**Question Stem:**
A 59-year-old overweight woman presents to the urgent care clinic with the complaint of severe abdominal pain for the past 2 hours. She also complains of a dull pain in her back with nausea and vomiting several times. Her pain has no relation with food. Her past medical history is significant for recurrent abdominal pain due to cholelithiasis. Her father died at the age of 60 with some form of abdominal cancer. Her temperature is 37°C (98.6°F), respirations are 15/min, pulse is 67/min, and blood pressure is 122/98 mm Hg. Physical exam is unremarkable. However, a CT scan of the abdomen shows a calcified mass near her gallbladder. Which of the following diagnoses should be excluded first in this patient?

**Options:**
- **A:** Acute cholecystitis
- **B:** Gallbladder cancer
- **C:** Choledocholithiasis
- **D:** Pancreatitis

**Ground Truth Answer:** Option **B**

---

### 🔴 [17/89] ID: `test-00016`
**Question Stem:**
A 7-year-old boy is brought to his pediatrician’s office for a follow-up visit. He was diagnosed with asthma when he was 3 years old and has since been on treatment for the condition. He is currently on a β-agonist inhaler because of exacerbation of his symptoms. He has observed that his symptoms are more prominent in springtime, especially when the new flowers are blooming. His mother has a backyard garden and whenever he goes out to play there, he experiences chest tightness with associated shortness of breath. He has been advised to take more precaution during this seasonal change and to stay away from pollen. He is also being considered for an experimental therapy, which attenuates the activity of certain mediators which cause his asthmatic attack. The targeted mediator favors the class switching of antibodies. A reduction in this mechanism will eventually reduce the exaggerated response observed during his asthmatic attacks, even when exposed to an allergen. Which of the following mediators is described in this experimental study?

**Options:**
- **A:** IL-2
- **B:** IL-10
- **C:** IL-13
- **D:** IL-4

**Ground Truth Answer:** Option **D**

---

### 🔴 [18/89] ID: `test-00017`
**Question Stem:**
A 3-month-old boy is brought the emergency department by his parents after an episode of cyanosis and muscle hypotonia that resolved after 2 minutes. Diagnostic evaluation fails to discover an exact etiology of the boy's symptoms and the episode is classified as a brief resolved unexplained event (BRUE). The risk profile for BRUE in infants remains largely unknown. The pediatrician who saw the boy in the emergency department is trying to identify risk factors for BRUE. She is aware of several confounders, including age, socioeconomic background, and family history of medical illness. She recruits 75 infants under 1 year of age with BRUE and 75 infants without BRUE of the same age, socioeconomic background, and family history of medical illness. She then compares the two groups with regard to history of feeding problems and history of recent upper respiratory infection. Which of the following methods was conducted to control confounding bias in the study?

**Options:**
- **A:** Blinding
- **B:** Restriction
- **C:** Randomization
- **D:** Matching

**Ground Truth Answer:** Option **D**

---

### 🔴 [19/89] ID: `test-00018`
**Question Stem:**
A 29-year-old man presents to the emergency department due to central chest pain over the past 3 days which is constant and unrelated to exertion. The pain is sharp, severe, increases when lying down, and improves with leaning forward. The pain also radiates to his shoulders and neck. The patient has no past medical history. He has smoked 10 cigarettes per day for the past 7 years and occasionally drinks alcohol. He presents with vital signs: blood pressure 110/70 mm Hg, regular radial pulse of 95/min, and temperature 37.3°C (99.1°F). On physical exam, a scratching sound of to-and-from character is audible over the left sternal border at end-expiration with the patient leaning forward. His chest X-ray is normal and ECG is shown in the picture. Which of the following is the optimal therapy for this patient?

**Options:**
- **A:** Indomethacin +/- omeprazole
- **B:** Ibuprofen + colchicine +/- omeprazole
- **C:** Pericardiocentesis
- **D:** Pericardiectomy

**Ground Truth Answer:** Option **B**

---

### 🔴 [20/89] ID: `test-00019`
**Question Stem:**
A 46-year-old man is brought to the emergency department for evaluation of altered mental status. He was found on the floor in front of his apartment. He is somnolent but responsive when aroused. His pulse is 64/min, respiratory rate is 15/min, and blood pressure is 120/75 mm Hg. On physical examination, an alcoholic smell and slurred speech are noted. Neurological exam shows diminished deep tendon reflexes bilaterally and an ataxic gait. His pupils are normal. Blood alcohol concentration is 0.04%. An ECG shows no abnormalities. Which of the following is the most likely cause of this patient's symptoms?

**Options:**
- **A:** Hypoglycemia
- **B:** Ethanol intoxication
- **C:** Cannabis intoxication
- **D:** Benzodiazepine intoxication
"

**Ground Truth Answer:** Option **D**

---

### 🔴 [21/89] ID: `test-00020`
**Question Stem:**
A 77-year-old woman presents to the emergency room with the complaints of fever, malaise, and night sweats. She recently observed an enlargement of her axillary lymph nodes, which she examines on a weekly basis. She has a remote history of breast cancer in her 60s that was treated with radiation and chemotherapy. She also reports a history of extensive travel to Africa and a 30-pack-year history of smoking. On physical exam, several axillary lymph nodes are palpable with a large non-tender palpable mass in her right axilla measuring 10 x 8 cm. Fine-needle aspiration demonstrates what the pathologist describes as "a centroblastic and immunoblastic cell presence, suspicious for non-Hodgkin’s lymphoma (NHL)–diffuse large B cell variant". Which of the following risk factors is responsible for this patient’s condition?

**Options:**
- **A:** Travel to Africa
- **B:** Axillary lymph node involvement
- **C:** Previous radiation therapy
- **D:** Previous breast cancer

**Ground Truth Answer:** Option **C**

---

### 🔴 [22/89] ID: `test-00021`
**Question Stem:**
A 3-month-old infant is brought to her pediatrician because she coughs and seems to have difficulty breathing while feeding. In addition, she seems to have less energy compared to other babies and appears listless throughout the day. She was born by cesarean section to a G1P1 woman with no prior medical history and had a normal APGAR score at birth. Her parents say that she has never been observed to turn blue. Physical exam reveals a high-pitched holosystolic murmur that is best heard at the lower left sternal border. The most likely cause of this patient's symptoms is associated with which of the following abnormalities?

**Options:**
- **A:** 22q11 deletion
- **B:** Deletion of genes on chromosome 7
- **C:** Lithium exposure in utero
- **D:** Maternal alcohol consumption

**Ground Truth Answer:** Option **A**

---

### 🔴 [23/89] ID: `test-00022`
**Question Stem:**
A 30-year-old African American woman comes to the physician for the evaluation of a dry cough and chest discomfort for the past 3 days. During this period, the patient has had headaches, muscle aches, joint pain, fever, and chills. Ten days ago, she was hiking with her family in Mississippi. The patient has asthma that is treated with an albuterol inhaler. Her mother has a lung disease treated with methotrexate. The patient has smoked one pack of cigarettes daily for the past 10 years. Her temperature is 38°C (100.4°F). Physical examination shows slight wheezes throughout both lung fields. Laboratory studies and urinalysis are positive for polysaccharide antigen. Bronchoalveolar lavage using silver/PAS-staining shows macrophages filled with a dimorphic fungus with septate hyphae. Which of the following is the most likely cause of this patient's symptoms?

**Options:**
- **A:** Legionella pneumophila infection
- **B:** Pneumocystis pneumonia
- **C:** Histoplasma capsulatum infection
- **D:** Blastomyces dermatitidis infection

**Ground Truth Answer:** Option **C**

---

### 🔴 [24/89] ID: `test-00023`
**Question Stem:**
A 62-year-old patient has been hospitalized for a week due to a stroke. One week into the hospitalization, he develops a fever and purulent cough. His vitals include: heart rate 88/min, respiratory rate 20/min, temperature 38.4°C (101.1°F), and blood pressure 110/85 mm Hg. On physical examination, he has basal crackles on the right side of the chest. Chest radiography shows a new consolidation on the same side. Complete blood count is as follows:
Hemoglobin 16 mg/dL
Hematocrit 50%
Leukocyte count 8,900/mm3
Neutrophils 72%
Bands 4%
Eosinophils 2%
Basophils 0%
Lymphocytes 17%
Monocytes 5%
Platelet count 280,000/mm3
What is the most likely causal microorganism?

**Options:**
- **A:** Streptococcus pneumoniae
- **B:** Mycobacterium tuberculosis
- **C:** Haemophilus influenzae
- **D:** Staphylococcus aureus

**Ground Truth Answer:** Option **D**

---

### 🔴 [25/89] ID: `test-00024`
**Question Stem:**
A 6-year-old boy is brought to the emergency department by his mother for worsening wheezing and shortness of breath over the past day. He has not had a fever, cough, vomiting, or diarrhea. He has asthma and eczema. He uses a glucocorticoid inhaler and an albuterol inhaler but has missed his medications for the past week while on vacation. He appears uncomfortable. His temperature is 36°C (96.8°F), pulse is 120/min, respirations are 40/min, and blood pressure is 100/80. Expiratory and inspiratory wheezing is heard throughout both lung fields. There are moderate intercostal and subcostal retractions and a decreased inspiratory to expiratory ratio. Nebulized albuterol and ipratropium treatments and intravenous methylprednisolone are given in the emergency department for a presumed asthma exacerbation. One hour later, the child is limp and lethargic. Magnesium sulfate is administered. His temperature is 36°C (96.8°F), pulse is 150/min, respirations are 22/min, and blood pressure is 100/70. No wheezing is heard on repeat pulmonary examination. Which of the following is the most appropriate next step in management?

**Options:**
- **A:** Intubate with mechanical ventilation
- **B:** Perform needle thoracostomy at the 2nd intercostal space
- **C:** Perform bronchoscopy
- **D:** Provide additional dose of methylprednisolone

**Ground Truth Answer:** Option **A**

---

### 🔴 [26/89] ID: `test-00025`
**Question Stem:**
A 5-year-old female suffers from recurrent infections by Aspergillus species, Pseudomonas species, and Staphylococcus aureus. The patient's neutrophils are examined in the laboratory and they fail to react during the nitroblue tetrazolium test. Which of the following is most likely dysfunctional in this patient?

**Options:**
- **A:** Immunoglobulin class switching
- **B:** Superoxide dismutase
- **C:** Myeloperoxidase
- **D:** Respiratory burst

**Ground Truth Answer:** Option **D**

---

### 🔴 [27/89] ID: `test-00026`
**Question Stem:**
A 3-year-old boy presents to the emergency department with a ‘cough-like-a-seal bark’ and a high-pitched inspiratory noise that is audible without a stethoscope. His mother reports that his cough has worsened over the last few hours. The patient's blood pressure is 118/78 mm Hg, pulse is 90/min, respiratory rate is 35/min, and temperature is 38.3°C (101.1°F). On physical examination, the boy is sitting and leaning forward in apparent respiratory distress with suprasternal and intercostal retractions. Auscultation reveals inspiratory stridor without wheezing. He has a frequent barking cough and a hoarse voice when he speaks. What is a chest X-ray likely to show?

**Options:**
- **A:** Increased interstitial markings
- **B:** Lobar consolidation in the lingual
- **C:** Thumbprint sign on the lateral image
- **D:** Steeple sign

**Ground Truth Answer:** Option **D**

---

### 🔴 [28/89] ID: `test-00027`
**Question Stem:**
A 26-year-old woman presents to a gynecologist after a missed period. After performing a complete physical examination and a screening test for pregnancy, her gynecologist informs her that she is pregnant. She is very surprised as she has been taking oral contraceptives regularly. When the gynecologist asks her about the consumption of any other medications, she mentions that she was placed on treatment for pulmonary tuberculosis (TB) 2 months ago. Her current anti-TB regimen includes rifampin, isoniazid, pyrazinamide, and ethambutol. Which of the following mechanisms best explains the failure of oral contraceptives in this patient?

**Options:**
- **A:** Induction of CYP3A4 by rifampin leading to decreased serum levels of ethinylestradiol and progesterone
- **B:** Induction of CYP2A6 by rifampin leading to increased inactivation of ethinylestradiol
- **C:** Interference with the intestinal absorption of the oral contraceptive by pyrazinamide
- **D:** Increased renal elimination of the progesterone component of the oral contraceptive by ethambutol

**Ground Truth Answer:** Option **A**

---

### 🔴 [29/89] ID: `test-00028`
**Question Stem:**
A 4-year-old previously healthy boy presents with 4 days of intermittent vomiting and 5-6 daily loose stools. His mother noted bloody stools and decreased oral intake of food and water over the last 24 hours. He is normally in daycare; however, he has been home for the past 3 days. On physical exam his temperature is 102.2°F (39°C), blood pressure is 140/90 mmHg, pulse is 120/min, respirations are 22/min and O2 saturation is 99% on room air. He has dry mucous membranes. On abdominal exam you note diffuse tenderness to palpation without rebound or guarding. There are no masses, hepatosplenomegaly, and bowel sounds are hyperactive. Ultrasound of the right lower quadrant is negative for appendicitis. Stool is guaiac positive. He receives 15mg/kg acetaminophen and fluids are started. The next day, he complains of lower extremity weakness and tingling. On repeat exam, lower extremity strength is 3/5 with diminished patellar deep tendon reflexes. Which of the following lab findings would most likely be seen in this patient?

**Options:**
- **A:** Gram stain positive CSF
- **B:** Peripheral eosinophilia
- **C:** Xanthochromia on cerebrospinal fluid analysis
- **D:** Increased cerebrospinal fluid protein with normal cell count

**Ground Truth Answer:** Option **D**

---

### 🔴 [30/89] ID: `test-00029`
**Question Stem:**
A 3-week-old male newborn is brought to the physician because of an inward turning of his left forefoot. He was born at 38 weeks' gestation by cesarean section because of breech presentation. The pregnancy was complicated by oligohydramnios. Examination shows concavity of the medial border of the left foot with a skin crease just below the ball of the great toe. The lateral border of the left foot is convex. The heel is in neutral position. Tickling the lateral border of the foot leads to correction of the deformity. The remainder of the examination shows no abnormalities. X-ray of the left foot shows an increased angle between the 1st and 2nd metatarsal bones. Which of the following is the most appropriate next step in the management of this patient?

**Options:**
- **A:** Foot abduction brace
- **B:** Arthrodesis of the forefoot
- **C:** Reassurance
- **D:** Tarsometatarsal capsulotomy

**Ground Truth Answer:** Option **C**

---

### 🔴 [31/89] ID: `test-00030`
**Question Stem:**
A 42-year-old woman comes to the emergency department because of a 2-day history of right upper abdominal pain and nausea. She is 163 cm (5 ft 4 in) tall and weighs 91 kg (200 lb); her BMI is 34 kg/m2. Her temperature is 38.5°C (101.3°F). Physical examination shows a distended abdomen and right upper quadrant tenderness with normal bowel sounds. Laboratory studies show:
Leukocyte count 14,000/mm3
Serum
Total bilirubin 1.1 mg/dL
AST 32 U/L
ALT 40 U/L
Alkaline phosphatase 68 U/L
Abdominal ultrasonography is performed, but the results are inconclusive. Cholescintigraphy shows the intrahepatic bile ducts, hepatic ducts, common bile duct, and proximal small bowel. Which of the following is the most likely cause of this patient's symptoms?"

**Options:**
- **A:** Autodigestion of pancreatic parenchyma
- **B:** Fistula between the gallbladder and small intestine
- **C:** Infection with a hepatotropic virus
- **D:** Obstruction of the cystic duct

**Ground Truth Answer:** Option **D**

---

### 🔴 [32/89] ID: `test-00031`
**Question Stem:**
A 72-year-old woman is admitted to the intensive care unit for shortness of breath and palpitations. A cardiac catheterization is performed and measurements of the left ventricular volume and pressure at different points in the cardiac cycle are obtained. The patient's pressure-volume loop (gray) is shown with a normal pressure-volume loop (black) for comparison. Which of the following is the most likely underlying cause of this patient's symptoms?

**Options:**
- **A:** Mitral valve regurgitation
- **B:** Increased systemic vascular resistance
- **C:** Increased ventricular wall stiffness
- **D:** Impaired left ventricular contractility

**Ground Truth Answer:** Option **C**

---

### 🔴 [33/89] ID: `test-00032`
**Question Stem:**
A 22-year-old woman is brought to the emergency department because of a 2-day history of fever, intermittent rigors, and night sweats. She also has a 1-month history of progressive fatigue. Five weeks ago, she was hospitalized and received intravenous antibiotics for treatment of bacterial meningitis while visiting relatives in Guatemala. Her temperature is 39.4°C (102.9°F), pulse is 130/min, and blood pressure is 105/70 mm Hg. Examination shows pallor and scattered petechiae and ecchymoses. Laboratory studies show a hemoglobin concentration of 9.0 g/dL, a leukocyte count of 1,100/mm3 with 30% segmented neutrophils, and a platelet count of 20,000/mm3 . Blood cultures grow coagulase-negative staphylococci. The patient was most likely treated with which of the following antibiotics?

**Options:**
- **A:** Doxycycline
- **B:** Trimethoprim/sulfamethoxazole
- **C:** Linezolid
- **D:** Chloramphenicol

**Ground Truth Answer:** Option **D**

---

### 🔴 [34/89] ID: `test-00033`
**Question Stem:**
An otherwise healthy 50-year-old man comes to the physician because of a 6-month history of increasingly frequent episodes of upper abdominal pain, nausea, vomiting, and diarrhea. He has had a 3.2-kg (7-lb) weight loss during this time. Physical examination shows bilateral pitting pedal edema. An endoscopy shows prominent rugae in the gastric fundus. Biopsy shows parietal cell atrophy. Which of the following is the most likely underlying cause?

**Options:**
- **A:** Serotonin-secreting gastric tumor
- **B:** Proliferation of gastric mucus-producing cells
- **C:** Excessive somatostatin secretion
- **D:** Ectopic secretion of gastrin

**Ground Truth Answer:** Option **B**

---

### 🔴 [35/89] ID: `test-00034`
**Question Stem:**
A 27-year-old man presents to the emergency department. He was brought in by staff from the homeless shelter when they found him unresponsive. The patient is a known IV drug abuser but otherwise has an unknown past medical history. He currently attends a methadone clinic. His temperature is 99.5°F (37.5°C), blood pressure is 97/48 mmHg, pulse is 140/min, respirations are 29/min, and oxygen saturation is 98% on room air. Initial laboratory values are shown below.

Serum:
Na+: 139 mEq/L
Cl-: 100 mEq/L
K+: 6.3 mEq/L
HCO3-: 17 mEq/L
Glucose: 589 mg/dL

The patient is given treatment. After treatment, his temperature is 99.5°F (37.5°C), blood pressure is 117/78 mmHg, pulse is 100/min, respirations are 23/min, and oxygen saturation is 98% on room air. His laboratory values are seen below.

Serum:
Na+: 139 mEq/L
Cl-: 100 mEq/L
K+: 4.3 mEq/L
HCO3-: 19 mEq/L
Glucose: 90 mg/dL

Which of the following is the best next step in management?

**Options:**
- **A:** Insulin, potassium, IV fluids, and glucose
- **B:** IV fluids only
- **C:** Oral rehydration
- **D:** Supportive therapy and close monitoring

**Ground Truth Answer:** Option **A**

---

### 🔴 [36/89] ID: `test-00035`
**Question Stem:**
A 67-year-old man who was diagnosed with arthritis 16 years ago presents with right knee swelling and pain. His left knee was swollen a few weeks ago, but now with both joints affected, he has difficulty walking and feels frustrated. He also has back pain which makes it extremely difficult to move around and be active during the day. He says his pain significantly improves with rest. He also suffers from dandruff for which he uses special shampoos. Physical examination is notable for pitting of his nails. Which of the following is the most likely diagnosis?

**Options:**
- **A:** Psoriatic arthritis
- **B:** Arthritis mutilans
- **C:** Rheumatoid arthritis
- **D:** Mixed connective tissue disease

**Ground Truth Answer:** Option **A**

---

### 🔴 [37/89] ID: `test-00036`
**Question Stem:**
A 65-year-old woman schedules an appointment with her physician for evaluation of weight loss and weakness. The weakness is more noticeable when climbing stairs and combing hair. The weakness improves after brief exercise. She also has bladder incontinence for the past 4 weeks and was given an anticholinesterase agent with no improvement. What is the most likely cause of the symptoms?

**Options:**
- **A:** Duchenne muscular dystrophy
- **B:** Hypothyroidism
- **C:** Myasthenia gravis
- **D:** Paraneoplastic syndrome from small cell carcinoma of the lung

**Ground Truth Answer:** Option **D**

---

### 🔴 [38/89] ID: `test-00037`
**Question Stem:**
A 3-month-old boy presents to his pediatrician with persistent diarrhea, oral candidiasis, and signs and symptoms suggestive of respiratory syncytial virus (RSV) pneumonia. His weight is in the 10th percentile. He is being evaluated for an immunodeficiency disease. Laboratory results for the HIV are negative by PCR. Which of the following is the most likely cause of these findings in this patient?

**Options:**
- **A:** Defective T cell function
- **B:** Grossly reduced levels of B cells
- **C:** Defective isotype switching
- **D:** Selective IgA deficiency

**Ground Truth Answer:** Option **A**

---

### 🔴 [39/89] ID: `test-00038`
**Question Stem:**
A 53-year-old man with recurrent pancreatic adenocarcinoma is enrolled in a clinical trial for a novel chemotherapeutic agent that his physician believes may be beneficial to his condition. The novel drug was previously tested in a small population and is now undergoing a larger phase 3 trial in preparation for FDA approval. A dose-response trial had the following results:

10 mg dose - 6/59 patients demonstrated improvement
20 mg dose - 19/49 patients demonstrated improvement
30 mg dose - 26/53 patients demonstrated improvement
40 mg dose - 46/51 patients demonstrated improvement

The same trial also had the following safety profile:

20 mg dose - 5/49 patients had a treatment related adverse event
40 mg dose - 11/51 patients had a treatment related adverse event
60 mg dose - 15/42 patients had a treatment related adverse event
80 mg dose - 23/47 patients had a treatment related adverse event
100 mg dose - 47/52 patients had a treatment related adverse event

Based on this study, which of the following represents the most likely therapeutic index for this novel chemotherapeutic agent?

**Options:**
- **A:** 0.375
- **B:** 0.5
- **C:** 2.5
- **D:** 2.67

**Ground Truth Answer:** Option **D**

---

### 🔴 [40/89] ID: `test-00039`
**Question Stem:**
A 67-year-old male is seen by neurology after he was noticed to be speaking strangely by his family. After acute treatment with tissue plasminogen activator (tPA), the patient is able to recover most of his speech. Subsequent neurologic exam finds that the patient is fluent while speaking and is able to comprehend both one and two step instructions. Noticeably the patient remains unable to complete tasks involving verbal repetition. Residual damage to which of the following structures is most likely responsible for this patient's syndrome?

**Options:**
- **A:** Arcuate fasciculus
- **B:** Superior temporal gyrus
- **C:** Inferior frontal gyrus + superior temporal gyrus
- **D:** Arcuate fasciculus + inferior frontal gyrus + superior temporal gyrus

**Ground Truth Answer:** Option **A**

---

### 🔴 [41/89] ID: `test-00040`
**Question Stem:**
A 27-year-old woman presents to the office with concerns about her long struggle with her physical appearance since adolescence. She says she has always been "large" and was constantly targeted by her classmates and coworkers for being so. Her main concern at the moment is her acne and unwanted facial hair on her upper lip, for which she often visits a local spa. She has tried numerous diet plans, exercise regimens, and cosmetic products with little to no effect. Recently, she underwent a glucose tolerance test that showed a plasma glucose level of 160 mg/dL (8.9 mmol/L) after 2 hours of a 75 g dose of oral glucose. She has a family history of type 2 diabetes mellitus and a menstrual cycle that occurs every 45 days. Her pulse is 72/min and the blood pressure is 138/80 mm Hg. On physical examination, her height is 160 cm (5 ft 3 in) and her weight is 85 kg (187 lb), and she has severe inflammatory acne over the cheeks and forehead and dark coarse hairs on the back. What is the most likely diagnosis of this patient?

**Options:**
- **A:** Hypothyroidism
- **B:** Idiopathic hirsutism
- **C:** Polycystic ovarian syndrome (PCOS)
- **D:** Ovarian hyperthecosis

**Ground Truth Answer:** Option **C**

---

### 🔴 [42/89] ID: `test-00041`
**Question Stem:**
А 43-уеаr-old mаn рrеѕеntѕ wіth tіnglіng аnd numbnеѕѕ of the lowеr lіmbѕ for 2 wееkѕ. Не also сomрlаіnѕ of реrѕіѕtеnt раіn in his legs whісh is not relieved by over-the-counter analgesics. Past medical history is significant for type 2 dіаbеtes mellitus for 2 уеаrѕ, inconsistently managed with mеtformіn аnd glіmеріrіdе. Оn physical ехаmіnаtіon, thеrе іѕ dесrеаѕеd ѕеnѕаtіon to pain in both lower lіmbs, but deep tеndon rеflехеѕ аrе іntасt. Ніѕ vіtаl ѕіgnѕ include: blood рrеѕѕurе 122/84 mm Нg, tеmреrаturе 36.7°C (98.1°F), and rеѕріrаtorу rаtе 10/mіn. His ankle-brachial pressure index (ABPI) on the right side is 1.1. His blood sugar analyses are as follows:
Fasting 141 mg/ dL
2 hours Post-Prandial 235 mg/ dL
HbA1c 8.1%
Which of the following is the best measure to prevent the progression of the symptoms present in this patient?

**Options:**
- **A:** Use of atorvastatin
- **B:** Femoro-Ileal artery bypass
- **C:** Strict blood glucose control
- **D:** Lower limb amputation

**Ground Truth Answer:** Option **C**

---

### 🔴 [43/89] ID: `test-00042`
**Question Stem:**
A one-day-old male is evaluated in the hospital nursery for bilious vomiting. The patient has urinated multiple times since birth but has not yet passed meconium. He was born at 38 weeks gestation to a 36-year-old gravida 3 via vaginal delivery. The pregnancy was complicated by gestational diabetes. The patient’s mother received routine prenatal care and declined first trimester screening. She subsequently underwent a quadruple screen at 15 weeks gestation that demonstrated decreased maternal serum alpha-fetoprotein (AFP), increased beta-hCG, decreased unconjugated estriol, and increased inhibin A. Her last ultrasound prior to onset of labor demonstrated an amniotic fluid index (AFI) of 28 cm. The patient’s two older siblings are both healthy. The patient’s temperature is 97.8°F (36.6°C), blood pressure is 58/37 mmHg, pulse is 166/min, and respirations are 38/min. On physical exam, the patient is in moderate distress. His abdomen is firm and distended with hypoactive bowel sounds.

Which of the following is the most likely etiology of this fetus’s presentation?

**Options:**
- **A:** Duodenal atresia
- **B:** Meconium ileus
- **C:** Necrotizing enterocolitis
- **D:** Pyloric stenosis

**Ground Truth Answer:** Option **A**

---

### 🔴 [44/89] ID: `test-00043`
**Question Stem:**
A healthy 23-year-old male is undergoing an exercise stress test as part of his physiology class. If blood were to be sampled at different locations before and after the stress test, which area of the body would contain the lowest oxygen content at both time points?

**Options:**
- **A:** Inferior vena cava
- **B:** Coronary sinus
- **C:** Pulmonary artery
- **D:** Pulmonary vein

**Ground Truth Answer:** Option **B**

---

### 🔴 [45/89] ID: `test-00044`
**Question Stem:**
A 45-year-old G5P4105 presents to her gynecologist’s office with six months of increasingly heavy periods. She now soaks one super absorbent tampon every two hours for five days each cycle. Her cycles have also become irregular, with intermenstrual bleeding for the last two months. She now experiences significant dysmenorrhea, requiring 400 mg ibuprofen every four hours for the majority of each menses. In addition, she reports new onset mild dyspareunia with intercourse and a “heavy feeling” in her pelvis. She has also noticed increased urinary frequency but denies bowel changes. The patient has a past medical history of obesity and type II diabetes on metformin. Her last child was born four years ago, and she has had five spontaneous vaginal deliveries. At this office visit, temperature is 98.5°F (36.9°C), blood pressure is 137/84 mmHg, pulse is 87/min, and respirations are 14/min. Which of the following physical exam findings is most likely to be present in this patient?

**Options:**
- **A:** Globular 10-week sized uterus
- **B:** Adnexal mass
- **C:** Irregular 14-week sized uterus
- **D:** No remarkable physical exam finding

**Ground Truth Answer:** Option **A**

---

### 🔴 [46/89] ID: `test-00045`
**Question Stem:**
A man is brought into the emergency department by the police department. The officer state that the man has been arrested multiple times for public alcohol intoxication, but recently became homeless. On exam, the man is behaving erratically. His vitals are all within normal limits. He appears confused and has a slurred speech. On gait exam, the patient is ataxic and cannot stand without support for more than a few seconds. Labs return with the following values: Na 140, K 4, Cl 106, BUN 8, Cr 2. His ABG has pH 7.3, PaCO2 13mm, PaO2 130mm, HCO3 7. His urinalysis is shown in Figure 1. Blood salicylate levels return as normal. While you await other diagnostic tests, which of the following should be administered next to treat this patient?

**Options:**
- **A:** Ethanol
- **B:** Naltrexone
- **C:** Naloxone
- **D:** Fomepizole

**Ground Truth Answer:** Option **D**

---

### 🔴 [47/89] ID: `test-00046`
**Question Stem:**
A medical research study is evaluating an investigational novel drug (medication 1) compared with standard therapy (medication 2) in patients presenting to the emergency department with myocardial infarction (MI). The study enrolled a total of 3,000 subjects, 1,500 in each study arm. Follow-up was conducted at 45 days post-MI. The following are the results of the trial:
Endpoints Medication 1 Medication 2 P-Value
Primary: death from cardiac causes 134 210 0.03
Secondary: hyperkalemia 57 70 0.4
What is the number needed to treat (NNT) for the primary endpoint of death from cardiac causes? (Round to the nearest whole number.)

**Options:**
- **A:** 5
- **B:** 16
- **C:** 20
- **D:** 50

**Ground Truth Answer:** Option **C**

---

### 🔴 [48/89] ID: `test-00047`
**Question Stem:**
A 55-year-old man comes to the physician because of a 6-week history of tingling pain in the sole of his right foot when he raises it above chest level during exercises. He reports that he started exercising regularly 2 months ago and that his right calf cramps when he uses the incline feature on the treadmill, forcing him to take frequent breaks. The pain completely disappears after resting for a few minutes. He has an 8-year history of type 2 diabetes mellitus. He has smoked two packs of cigarettes daily for 34 years. His only medication is metformin. His pulse is 82/min, and blood pressure is 170/92 mm Hg. Straight leg raise test elicits pallor and tingling pain in the right foot. There is no pain in the back. His muscle strength is normal. Femoral pulses are palpable; right pedal pulses are absent. Which of the following is the most likely diagnosis?

**Options:**
- **A:** Femoropopliteal artery stenosis
- **B:** Acute thrombosis of right popliteal vein
- **C:** Lumbar spinal stenosis
- **D:** Aortoiliac artery stenosis
"

**Ground Truth Answer:** Option **A**

---

### 🔴 [49/89] ID: `test-00048`
**Question Stem:**
A 29-year-old primigravid woman at 35 weeks' gestation is admitted to the hospital in labor. She has no history of serious medical illness. She has had an uncomplicated pregnancy. Her last ultrasound at 22 weeks' gestation was normal. On admission, fetal heartbeats cannot be detected by fetal doppler monitor. Ultrasound shows decreased amniotic fluid levels and no evidence of fetal movement, respiration, or heart activity. The patient gives birth to a 2296 g (5 lb 1 oz) male infant. Physical examination shows no signs of life. There are no visible malformations. The placenta is unremarkable. Which of the following is the most appropriate next step in management?

**Options:**
- **A:** Perform karyotyping of amniotic fluid
- **B:** Recommend autopsy of the infant
- **C:** Obtain photographs, x-ray imaging, and MRI
- **D:** Obtain consent for parental genetic testing

**Ground Truth Answer:** Option **B**

---

### 🔴 [50/89] ID: `test-00049`
**Question Stem:**
A 67-year-old man is hospitalized after several days of fever and increasing shortness of breath. Respiratory viral panel returns positive for influenza A and the patient is started on a standard five day course of oseltamivir. Despite therapy, his shortness of breath continues to worsen and his oxygen saturation decreases to the point where ICU transfer and intubation are required. Chest radiograph shows bilateral infiltrates. Clinical criteria are consistent with development of Acute Respiratory Distress Syndrome (ARDS). After several weeks of supportive therapy, the patient improves with lung function returning to near normal. Which of the following processes is involved in the regeneration of alveolar lining after damage to alveoli occurs?

**Options:**
- **A:** Sweeping of debris out of the alveoli by ciliated cells
- **B:** Excess mucus production by goblet cells
- **C:** Proliferation of surfactant-secreting cells
- **D:** Proliferation of club cells

**Ground Truth Answer:** Option **C**

---

### 🔴 [51/89] ID: `test-00050`
**Question Stem:**
A 65-year-old male is treated for anal carcinoma with therapy including external beam radiation. How does radiation affect cancer cells?

**Options:**
- **A:** Induces the formation of thymidine dimers
- **B:** Induces the formation of disulfide bonds
- **C:** Induces deamination of cytosine
- **D:** Induces breaks in double-stranded DNA

**Ground Truth Answer:** Option **D**

---

### 🔴 [52/89] ID: `test-00051`
**Question Stem:**
A 56-year-old man with a history of hypertension presents to his physician with progressive fatigue and new onset muscle cramps. He has had no recent changes to his medication regimen, which includes hydrochlorothiazide, lisinopril, and amlodipine. His temperature is 98.0°F (36.7°C), blood pressure is 174/111 mmHg, pulse is 70/min, respirations are 12/min, and oxygen saturation is 98% on room air. The patient's cardiopulmonary and abdominal exams are unremarkable. Laboratory values are ordered as seen below.

Serum:
Na+: 138 mEq/L
Cl-: 100 mEq/L
K+: 3.3 mEq/L
HCO3-: 33 mEq/L
BUN: 20 mg/dL
Glucose: 129 mg/dL

What is the most likely underlying etiology of this patient's hypertension?

**Options:**
- **A:** Aldosterone excess
- **B:** Catecholamine-secreting mass
- **C:** Cortisol excess
- **D:** Impaired kidney perfusion

**Ground Truth Answer:** Option **A**

---

### 🔴 [53/89] ID: `test-00052`
**Question Stem:**
A 65-year-old man comes to the physician because of a 1-week history of yellowish discoloration of his skin and generalized pruritus. Examination shows jaundice of the skin and scleral icterus. Urinalysis shows an elevated concentration of bilirubin and a low concentration of urobilinogen. Which of the following is the most likely underlying cause of these findings?

**Options:**
- **A:** Absent UDP-glucuronosyltransferase activity
- **B:** Increased hemoglobin breakdown
- **C:** Increased intestinal bilirubin reabsorption
- **D:** Defective hepatic bile excretion

**Ground Truth Answer:** Option **D**

---

### 🔴 [54/89] ID: `test-00053`
**Question Stem:**
A 72-year-old woman is brought to the physician by her daughter because of a 6-month history of worsening short-term memory deficits and social withdrawal. Treatment with galantamine is initiated. Two weeks later, the patient develops vomiting, mild crampy abdominal pain, and watery, nonbloody diarrhea. Which of the following is the most appropriate pharmacotherapy?

**Options:**
- **A:** Bethanechol
- **B:** Metoclopramide
- **C:** Atropine
- **D:** Oxybutynin

**Ground Truth Answer:** Option **C**

---

### 🔴 [55/89] ID: `test-00054`
**Question Stem:**
A 61-year-old man with schizophrenia in a locked inpatient psychiatry unit was observed making an unusual smacking motion with his lips, pursing his mouth, and blinking excessively. These symptoms started slowly about 2 weeks ago and have become more pronounced, but they do not seem to bother the patient. He resides in an inpatient psychiatric facility where he has been successfully treated with haloperidol for the past several months. His physical exam and vitals are within normal limits. Which of the following accurately describes his symptoms?

**Options:**
- **A:** Tardive dyskinesia
- **B:** Acute dystonia
- **C:** Pseudoparkinsonism
- **D:** Palsy

**Ground Truth Answer:** Option **A**

---

### 🔴 [56/89] ID: `test-00055`
**Question Stem:**
A 17-year-old football player with no significant past medical history, social history, or family history presents to his pediatrician with itching in his groin. He says this started during summer workouts leading up to this season. He denies having any rash anywhere else on his body. The blood pressure is 123/78 mm Hg, pulse is 67/min, respiratory rate is 15/min, and temperature is 38.1°C (98.7°F). Physical examination reveals an erythematous, well-demarcated patch with peripheral scale on the left thigh, pubic region, and perineum. There is no apparent scrotal involvement with the rash. How can you confirm the suspected diagnosis?

**Options:**
- **A:** KOH examination of lesion scrapings
- **B:** Nikolsky's sign on physical exam
- **C:** Gram stain of skin scrapings
- **D:** AFB stain of skin scrapings

**Ground Truth Answer:** Option **A**

---

### 🔴 [57/89] ID: `test-00056`
**Question Stem:**
An otherwise healthy 26-year-old man comes to the physician for medication counseling after recently being diagnosed with schizophrenia. Risperidone therapy is initiated. This patient is at increased risk for which of the following adverse effects?

**Options:**
- **A:** Agranulocytosis
- **B:** Shortened QT interval
- **C:** Gynecomastia
- **D:** Weight loss

**Ground Truth Answer:** Option **C**

---

### 🔴 [58/89] ID: `test-00057`
**Question Stem:**
A 15-year-old woman presents with fever, altered mental status and a petechial rash on her lower extremities and back since yesterday evening. She also says she has been nauseous for the past 3 hours and has vomited twice. The patient mentions she has had heavy menstrual bleeding for the past few days. Her blood pressure is 95/80 mm Hg and her temperature is 40.0°C (104.0°F). On physical examination, the patient appears diaphoretic. A pelvic examination reveals a tampon in her vagina. Binding and activation of which of the following T cell receptors is responsible for this patient’s most likely condition?

**Options:**
- **A:** B7 receptor
- **B:** Variable β-sequence of the T cell receptor
- **C:** CD3
- **D:** IgCAM

**Ground Truth Answer:** Option **B**

---

### 🔴 [59/89] ID: `test-00058`
**Question Stem:**
A 2-year-old boy is brought to the physician for evaluation of pallor and increasing lethargy for 2 days. One week ago, he experienced abdominal pain, vomiting, and bloody diarrhea that have since subsided. The patient's father states that they returned early from a 6-week roadtrip in Mexico because of these symptoms. His parents have been giving him oral rehydration solution. His immunizations are up-to-date. He appears pale. His temperature is 38.4°C (101.1°F), pulse is 130/min, respirations are 35/min, and blood pressure is 95/50 mm Hg. Examination shows scleral icterus. The abdomen is soft and nontender; there is no rebound or guarding. Bowel sounds are hyperactive. The remainder of the examination shows no abnormalities. Laboratory studies show:
Hemoglobin 8.5 g/dL
Mean corpuscular volume 94 μm3
Leukocyte count 18,000/mm3
Platelet count 45,000/mm3
Prothrombin time 12 sec
Partial thromboplastin time 34 sec
Serum
Urea nitrogen 28 mg/dL
Creatinine 1.6 mg/dL
Bilirubin
Total 2.5 mg/dL
Direct 0.1 mg/dL
Lactate dehydrogenase 1658 U/L
A blood smear shows schistocytes. Which of the following is the most likely diagnosis?"

**Options:**
- **A:** Henoch-Schönlein Purpura
- **B:** Thrombotic thrombocytopenic purpura
- **C:** Hemolytic uremic syndrome
- **D:** Immune thrombocytopenic purpura

**Ground Truth Answer:** Option **C**

---

### 🔴 [60/89] ID: `test-00059`
**Question Stem:**
A 17-year-old girl is referred by her dentist for a suspected eating disorder. She has been visiting the same dentist since childhood and for the past 2 years has had at least 2 visits for dental caries. She eventually admitted to him that she regularly induces vomiting by putting her fingers down her throat. She says she has been doing this for the last few years and purging at least once a week. More recently, she has been inducing emesis more often and even looked into diuretics as she feels that she is gaining more and more weight compared to her ‘skinny friends’. Her BMI is at the 50th percentile for her age and sex. Which of the following features is most consistent with this patient’s condition?

**Options:**
- **A:** Patients with this disorder are not further sub-typed
- **B:** Patients do not usually initiate treatment
- **C:** Patients can have a history of both anorexia and bulimia
- **D:** Patients will typically have a BMI between 17–18.5 kg/m2

**Ground Truth Answer:** Option **A**

---

### 🔴 [61/89] ID: `test-00060`
**Question Stem:**
A 14-year-old girl is brought to the physician by her father because of fever, chills, abdominal pain, and profuse non-bloody diarrhea. Her symptoms began one week ago, when she had several days of low-grade fever and constipation. She returned from Indonesia 2 weeks ago, where she spent the summer with her grandparents. Her temperature is 39.3°C (102.8°F). Examination shows diffuse abdominal tenderness and mild hepatosplenomegaly. There is a faint salmon-colored maculopapular rash on her trunk and abdomen. Which of the following is the most likely causal organism?

**Options:**
- **A:** Giardia lamblia
- **B:** Schistosoma mansoni
- **C:** Salmonella typhi
- **D:** Clostridium perfringens

**Ground Truth Answer:** Option **C**

---

### 🔴 [62/89] ID: `test-00061`
**Question Stem:**
A 22-year-old female college student is treated with metronidazole after presenting to student health services with itching, discharge, and pain in her vagina. At a party shortly afterward she experiences facial flushing, nausea, tachycardia, dyspnea, headache, and abdominal cramps after consuming alcohol. Serum levels of which of the following are likely elevated in this patient following alcohol consumption:

**Options:**
- **A:** Acetaldehyde
- **B:** Uric acid
- **C:** Cytochrome P-450 enzymes
- **D:** Amylase

**Ground Truth Answer:** Option **A**

---

### 🔴 [63/89] ID: `test-00062`
**Question Stem:**
A 23-year-old primigravida presents for a regular prenatal care visit at 16 weeks gestation. She complains of increased fatigability, but is otherwise well. She takes folic acid, iron, and vitamin D supplementation. Her vital signs are as follows: blood pressure, 110/70 mm Hg; heart rate, 86/min; respiratory rate, 13/min; and temperature, 36.6℃ (97.9℉). The physical examination is unremarkable. The complete blood count results are as below:
Erythrocyte count 3.9 million/mm3
Hb 11.1 g/dL
HCT 32%
Reticulocyte count 0.2%
MCV 90 fL
Platelet count 210,000/mm3
Leukocyte count 8,100/mm3
Which of the following tests is required to investigate the cause of the patient’s laboratory findings?

**Options:**
- **A:** Serum iron level
- **B:** Serum B12 level
- **C:** Transferrin
- **D:** No tests required

**Ground Truth Answer:** Option **D**

---

### 🔴 [64/89] ID: `test-00063`
**Question Stem:**
An 80-year-old man is transferred from a step-down unit to a med-surg floor in the hospital. He had undergone a successful hernia surgery 14 hours ago. Before the surgery, he was pre-treated with atropine, scopolamine, and morphine and recovered well in the PACU after the surgery. There were no complications in the step-down unit and the patient continued to recover. On the med-surg floor, his temperature is 36.8°C (98.2°F), the heart rate is 98/min, the respiratory rate is 15/min, the blood pressure is 100/75 mm Hg, the oxygen saturation is 90%. On physical exam, he is a well-developed, obese man. His heart has a regular rate and rhythm and his lungs are clear to auscultation bilaterally. His incision site is clean, dry, and intact with an appropriate level of swelling and erythema. During the physical, the patient mentions some discomfort in his abdomen and pelvis and during a records review it is noted that he has not passed urine in the PACU, step-down unit, or since arriving on the med-surg floor. A bladder scan is inconclusive due to body habitus. What is the next best step in the treatment of this patient?

**Options:**
- **A:** Insert a ‘straight cath’ into the patient’s bladder
- **B:** Aggressive IV fluids
- **C:** Digital rectal exam
- **D:** Renal ultrasound

**Ground Truth Answer:** Option **A**

---

### 🔴 [65/89] ID: `test-00064`
**Question Stem:**
A healthy 19-year-old man presents to his primary care provider complaining of painless “blisters” in his mouth. He reports that he noticed a white film on his tongue and the sides of his mouth 2 days ago while brushing his teeth. The film was easily brushed off. He also complains of a bitter metallic taste in his mouth but otherwise denies pain, burning, dysphagia, or hoarseness. He is otherwise healthy and takes no medications. He is a competitive swimmer and has had 8 sexual partners in the past year. He intermittently uses barrier protection. On exam, he is well-appearing and in no acute distress. His oral examination demonstrates patches of white pseudomembranes that can be wiped away to reveal erythematous mucosa. A medication with which of the following mechanisms of action is most appropriate in this patient?

**Options:**
- **A:** Disruption of cell membrane permeability
- **B:** Disruption of microtubule formation
- **C:** Inhibition of 14-alpha-demethylase
- **D:** Inhibition of beta-glucan synthase

**Ground Truth Answer:** Option **A**

---

### 🔴 [66/89] ID: `test-00065`
**Question Stem:**
A 56-year-old man presents to the clinic complaining of subacute back pain for the past month. The pain is described as a dull, constant ache that is worse at night. He could not recall any precipitating event except for an amateur weight-lifting competition that he participated in 2 months ago. Past medical history is significant for non-small cell lung cancer that was diagnosed and successfully treated. A PET scan 1 year ago demonstrated no recurrence. Physical exam was unremarkable except for some point tenderness along the lumbosacral area. What is the most likely imaging finding in this patient?

**Options:**
- **A:** Bulging disc impinging on lumbar spinal nerve
- **B:** Lytic lesions of the lumbar spine
- **C:** Narrowing of the lumbar disc space
- **D:** Sacroilitis and fusion of the lumbar spine

**Ground Truth Answer:** Option **B**

---

### 🔴 [67/89] ID: `test-00066`
**Question Stem:**
A 4-year-old boy is brought to the emergency department because of severe abdominal pain and bilious vomiting for 6 hours. He has not had bowel movements in the past 24 hours. He appears ill. His temperature is 37.8°C (100°F) and pulse is 122/min. Examination shows a distended abdomen. There is tenderness to palpation in the lower abdomen; guarding and rebound tenderness are present. Bowel sounds are decreased. An x-ray of the abdomen shows dilated loops of bowel. He has been accompanied by his 14-year-old brother. The surgeon recommends an emergency laparotomy. The parents are away visiting friends and cannot be reached. Which of the following is the most appropriate next best step in management?

**Options:**
- **A:** Get consent from the patient's brother
- **B:** Get consent from the patient
- **C:** Perform emergency laparotomy
- **D:** Delay surgery until parental consent

**Ground Truth Answer:** Option **C**

---

### 🔴 [68/89] ID: `test-00067`
**Question Stem:**
A 3-week-old male newborn is brought to the hospital because of poor weight gain since birth. He was born at 38 weeks' gestation via normal vaginal delivery. He weighed 3005 g (6 lb, 10 oz) at birth and currently weighs 2835 g (6 lb, 4 oz). He has been latching on and breastfeeding well since birth. His mother has a history of Graves' disease and underwent near-total thyroidectomy in the second trimester of her pregnancy after her symptoms could not be controlled with antithyroid drugs. She is currently receiving L-thyroxine therapy. The patient's temperature is 38.9°C (102°F), pulse is 176/min, and respirations are 42/min. He appears irritable. Examination shows a diaphoretic infant with a paucity of subcutaneous fat. There is swelling of the neck at the midline. Which of the following is the most likely cause?

**Options:**
- **A:** Transplacental passage of thyroid peroxidase antibodies
- **B:** Transplacental passage of TSH receptor antibodies
- **C:** Transplacental viral infection
- **D:** Opiate use in the mother

**Ground Truth Answer:** Option **B**

---

### 🔴 [69/89] ID: `test-00068`
**Question Stem:**
A 57-year-old female with a past medical history of alcoholism presents to the emergency room vomiting bright red blood. She is accompanied by her partner, who reports that she had been complaining of black and tarry stools for the past several days. Vital signs are temperature 37 degrees celsius, heart rate 141 beats per minute, blood pressure 90/60, respiratory rate 20, and oxygen saturation 99% on room air. On physical examination, she has splenomegaly and a positive fluid wave. The remainder of her examination is within normal limits. The patient is stabilized with intravenous fluids, and her blood pressure improves. Subsequent emergent upper endoscopy reveals bleeding from the submucosal veins in the lower 1/3 of the esophagus, but no gastric bleed. In the endoscopy suite she also receives IV octreotide. After intervention and resolution of her acute bleed, which of the following pharmacologic agents is indicated?

**Options:**
- **A:** Phentolamine
- **B:** Prazosin
- **C:** Nifedipine
- **D:** Nadalol

**Ground Truth Answer:** Option **D**

---

### 🔴 [70/89] ID: `test-00069`
**Question Stem:**
A 65-year-old man with a history of hypertension, obesity, and alcoholic cirrhosis is seen in clinic for follow-up. He feels well and currently drinks 5 glasses of wine each night. Medications include atenolol and lisinopril. On physical exam, temperature is 98.1 deg F (36.7 deg C), blood pressure is 151/82 mmHg, pulse is 71/min, and respirations are 14/min. He has spider angiomata on his chest; no asterixis, jaundice, ascites, or peripheral edema is noted. Screening ultrasound reveals a new liver nodule, and follow up CT demonstrates a 2 cm right hepatic lobe lesion with enhancement in the arterial phase. No hypodense attenuation is seen on the venous or delayed phase. What is the next step in management?

**Options:**
- **A:** Proceed with liver biopsy
- **B:** Refer for surgical resection
- **C:** Refer for radiofrequency ablation
- **D:** Observe and get follow-up imaging in 3 months

**Ground Truth Answer:** Option **A**

---

### 🔴 [71/89] ID: `test-00070`
**Question Stem:**
A 66-year-old man is brought to the emergency room by his wife due to abdominal distension and persistent somnolence for the past 2 weeks. The patient’s wife says that he has been sleeping much more than usual for the past 5 days. His bowel and bladder habit have not changed. His past medical history is significant for alcoholic liver cirrhosis. His vital signs include: pulse 76/min, respiratory rate 15/min, temperature 38.0°C (100.4°F) and blood pressure 122/75 mm Hg. On physical examination, the patient is altered and not responsive to commands. Oriented x 0. The abdomen is significantly distended. Shifting dullness is present and a positive fluid wave can be elicited. Hyperreflexia and asterixis are noted. Laboratory findings are significant for the following:
Laboratory test
Sodium 140 mEq/L
Potassium 3.5 mEq/L
Chloride 97 mEq/L
Glucose 90 mg/dL
Ammonia 100 µg/dL (ref: 19-60 µg/dL)
Arterial blood gas
pH 7.4
pCO2 40 mm Hg
pO2 90 mm Hg
HCO3 26 mEq/L
An abdominal ultrasound shows surface nodularity compatible with cirrhosis but no other changes aside from ascites. An upper GI endoscopy is performed which shows gastric varices with no signs of active bleeding. An MRI of the brain is insignificant. What is the most likely precipitating factor that led to this patient’s condition?

**Options:**
- **A:** Spontaneous bacterial peritonitis
- **B:** Metabolic alkalosis
- **C:** Portal vein thrombosis
- **D:** Hypoglycemia

**Ground Truth Answer:** Option **A**

---

### 🔴 [72/89] ID: `test-00071`
**Question Stem:**
A 27-year-old man presents to the emergency department after a motor vehicle collision. The patient was the front seat unrestrained driver in a head on collision. The patient’s echocardiogram (ECG) is notable only for sinus tachycardia. His temperature is 99.5°F (37.5°C), blood pressure is 107/58 mmHg, pulse is 120/min, respirations are 17/min, and oxygen saturation is 98% on room air. The patient is given 2 liters of Ringer lactate solution and morphine. Initial workup demonstrates that the patient’s pulmonary capillary wedge pressure and troponins are elevated. The patient is currently complaining of chest pain. Physical exam is notable for an uncomfortable young man with bruising over his chest wall. Which of the following is the most likely diagnosis?

**Options:**
- **A:** Cardiac contusion
- **B:** Hemorrhage
- **C:** Pulmonary contusion
- **D:** Takotsubo cardiomyopathy

**Ground Truth Answer:** Option **A**

---

### 🔴 [73/89] ID: `test-00072`
**Question Stem:**
A 56-year-old man presents to the emergency department for progressively worsening fatigue, malaise, fever, and abdominal pain. He reports that his symptoms began approximately 1 week ago and he has noticed episodes of diarrhea. He recently started melatonin and magnesium supplementation in hopes of improving his sleep. Medical history is significant for alcohol use disorder that required multiple hospital admissions for management of acute pancreatitis and cirrhosis. He states that he occasionally injects heroin intravenously. Temperature is 100°F (37.8°C), blood pressure is 105/70 mmHg, pulse is 92/min, and respirations are 17/min. Physical examination is significant for scleral icterus, hepatomegaly, ascites, and diffuse abdominal tenderness. Laboratory testing is significant for leukocytosis and metabolic acidosis. A paracentesis is performed and he is admitted into the hospital to receive intravenous cefotaxime and albumin. Ascitic fluid analysis demonstrates a polymorphonuclear cell count of 280 cells/mm^3, serum-ascites albumin gradient of 1.3 g/dL, and a culture positive for Escherichia coli sensitive to cefotaxime and ceftriaxone. On hospital day 2, the nurse reports that the patient is oliguric in the setting of constant fluid intake. Physical examination is unchanged. Laboratory testing is significant for a serum sodium of 131 mEq/L and creatinine of 1.8 mg/dL (it was 0.9 mg/dL on admission). Urine studies are significant for a low urine sodium level, without evidence of blood or protein. Since the hospital admission, he has not been started on any new medications. Which of the following will be the best treatment option for this patient?

**Options:**
- **A:** Adding dopamine to his treatment regimen
- **B:** Adding lisinopril to his treatment regimen
- **C:** Liver transplantation
- **D:** Transjugular intrahepatic portosystemic shunting

**Ground Truth Answer:** Option **C**

---

### 🔴 [74/89] ID: `test-00073`
**Question Stem:**
A 29-year-old woman presents to the clinic after several months of weight loss. She noticed a 6.8 kg (15 lb) unintentional weight loss over the preceding several months. She has not changed her diet or exercise habits. She also reports feuding with her boyfriend over the temperature of their shared apartment, as she always feels warmer than he does. The vital signs include: heart rate 110/min and blood pressure 146/78 mm Hg. The physical exam is notable for warm and slightly moist skin. She also exhibits a fine tremor in her hands when her arms are outstretched. The urine pregnancy test is negative. Which of the following is the best single treatment option for this patient?

**Options:**
- **A:** Glucocorticoids
- **B:** Methimazole
- **C:** Propranolol
- **D:** Radioiodine therapy

**Ground Truth Answer:** Option **B**

---

### 🔴 [75/89] ID: `test-00074`
**Question Stem:**
A 21-year-old man comes to the physician because of pruritus and a hypopigmented rash on his upper body for 5 days. He first noticed the symptoms after returning from a business trip last week in the Bahamas. While he was there, he visited a couple of beaches and went hiking with some coworkers. The rash initially started as a single lesion on his upper back but since then has extended to his shoulders. He has a history of type 1 diabetes mellitus controlled with an insulin pump. He works as an office manager and has no known exposure to melanocytotoxic chemicals. He has been sexually active with three female partners over the past year and uses condoms inconsistently. He is 183 cm (6 ft) tall and weighs 80 kg (176 lb); BMI is 23.9 kg/m2. His temperature is 37.2°C (99°F), pulse is 78/min, and blood pressure is 130/84 mm Hg. A photograph of the rash is shown. One month ago, his hemoglobin A1C was 7.8%. Which of the following is most likely to confirm the diagnosis?

**Options:**
- **A:** Wood lamp examination
- **B:** Skin culture
- **C:** Potassium hydroxide preparation
- **D:** Skin biopsy

**Ground Truth Answer:** Option **C**

---

### 🔴 [76/89] ID: `test-00075`
**Question Stem:**
A 5-year-old female is brought to a speech therapist for continuing work on improving her communication skills. She is only able to verbalize two word sentences and has generalized developmental delay. When she was born it was noticed that she had a high pitched mewing cry and subsequent physical exam revealed microcephaly, prominent epicanthal folds, and a holosystolic murmur best heard in the left 5th intercostal space near the sternum. An abnormality of which of the following chromosomes is most likely responsible for this patient's disorder?

**Options:**
- **A:** 5
- **B:** 7
- **C:** 18
- **D:** 21

**Ground Truth Answer:** Option **A**

---

### 🔴 [77/89] ID: `test-00076`
**Question Stem:**
A 62-year old female comes to the physician because of vaginal spotting and urinary urgency for the past 4 days. She has had no fever, abdominal pain, or diarrhea. Menopause occurred at 52 years of age. Her last Pap smear 1 year ago was normal. She has hypertension, hypercholesterolemia, and diabetes. Medications include atorvastatin, hydrochlorothiazide, metformin, and aspirin. She is sexually active with her husband. Her temperature is 37°C (98.6°F), pulse is 95/min, respirations are 12/min, and blood pressure is 155/65 mm Hg. Pelvic exam demonstrates a 4 x 3 cm firm, immobile erythematous mass on the right inner vaginal wall. Which of the following is the most appropriate next step in management?

**Options:**
- **A:** Pap smear
- **B:** Biopsy of the mass
- **C:** Incision and drainage
- **D:** Urine gonorrhea and chlamydia testing

**Ground Truth Answer:** Option **B**

---

### 🔴 [78/89] ID: `test-00077`
**Question Stem:**
A 59-year-old man is evaluated for progressive joint pain. There is swelling and tenderness over the first, second, and third metacarpophalangeal joints of both hands. His hand radiograph is shown. He has had diabetes mellitus for 2 years which is not well controlled with medications. Lab studies show a transferrin saturation of 88% and serum ferritin of 1,200 ng/mL. Which of the following best represents the etiology of this patient condition?

**Options:**
- **A:** Deposition of urate crystals
- **B:** Deposition of calcium pyrophosphate (CPP) crystals
- **C:** Inflammatory rheumatological syndrome
- **D:** Pathogenic inoculation of microbes

**Ground Truth Answer:** Option **B**

---

### 🔴 [79/89] ID: `test-00078`
**Question Stem:**
A newborn is found to be extremely cyanotic immediately after birth. He then develops progressive respiratory failure and is admitted to the neonatal ICU. A single loud S2 heart sound is appreciated as well as a machine-like murmur at the left upper sternal border. Radiography shows an enlarged "egg-shaped" heart. The newborn is then taken for a atrial septostomy to alleviate the condition pending definitive surgical correction. Which of the following is the most likely cause of this newborn's condition?

**Options:**
- **A:** Coarctation of the aorta
- **B:** Persistent truncus arteriosus
- **C:** Transposition of great vessels
- **D:** Tricuspid atresia

**Ground Truth Answer:** Option **C**

---

### 🔴 [80/89] ID: `test-00079`
**Question Stem:**
A 25-year-old male involved in a knife fight presents with a penetrating wound to the chest. The patient is unconscious and cannot provide any further history. Vitals show a temperature of 37-0°C (98.6°F), blood pressure of 85/55 mm Hg, pulse of 115/min, respirations of 19/min, and oxygen saturation of 92% on room air. On physical examination, the patient is diaphoretic and unresponsive. Extremities are pale and cool. There is a 3-inch long penetrating wound between the 3rd and 4th intercostal space on the left side of the chest, which is bleeding profusely. Transthoracic echocardiography reveals a full thickness penetrating injury to the right ventricular free wall. There are no apparent injuries to any coronary arteries or major branches. The patient is intubated and aggressive fluid resuscitation is initiated, including a blood transfusion. Which of the following is the best definitive surgical approach to take in this patient?

**Options:**
- **A:** Immediate cardiac transplant
- **B:** Watchful waiting while resuscitative fluids are initiated
- **C:** Interrupted 2-0 polypropylene suture with supporting pledgets
- **D:** Needle thoracostomy over the 2nd intercostal space

**Ground Truth Answer:** Option **C**

---

### 🔴 [81/89] ID: `test-00080`
**Question Stem:**
A post-mortem lung examination of a 68-year-old male overweight male with evidence of chronic lower extremity edema, a 60 pack-year smoking history and daily productive cough would be most likely to reveal:

**Options:**
- **A:** Hypereosinophilia
- **B:** Reid Index > 50%
- **C:** Non-caseating granulomas
- **D:** Evidence of a necrotizing infection

**Ground Truth Answer:** Option **B**

---

### 🔴 [82/89] ID: `test-00081`
**Question Stem:**
A 54-year-old male makes an appointment with his primary care physician due to chronic fatigue that has left him unable to hike or do other physically demanding activities with his friends. He has well-controlled hypertension and diabetes but has otherwise been healthy. He does not smoke but drinks socially with his friends. Physical exam reveals enlarged nodules that are not tender to palpation. A biopsy is obtained showing a characteristic change in chromosome 18. The regulation of which of the following proteins will be most abnormal in this patient?

**Options:**
- **A:** Caspase-9
- **B:** CD15
- **C:** Cyclin-dependent kinase 4
- **D:** Ras pathway transcription factors

**Ground Truth Answer:** Option **A**

---

### 🔴 [83/89] ID: `test-00082`
**Question Stem:**
A 44-year-old African-American woman comes to the physician for a routine examination. She is concerned about cancer because her uncle died of metastatic melanoma 1 year ago. She has no history of serious illness and does not take any medication. She has been working in a law firm for the past 20 years and travels to the Carribean regularly with her husband. Examination of her skin shows no abnormal moles or warts. This woman is at greatest risk of which of the following types of melanoma?

**Options:**
- **A:** Desmoplastic
- **B:** Nodular
- **C:** Acral lentiginous
- **D:** Superficial spreading

**Ground Truth Answer:** Option **C**

---

### 🔴 [84/89] ID: `test-00083`
**Question Stem:**
A 60-year-old woman is brought to the clinic by her daughter for evaluation. The daughter reports that her mother has recently been having difficulty combing her hair in the mornings. The patient’s family history is significant for an ischemic stroke in her father. The patient’s past medical history is positive for diverticulosis. She takes no medication. Her blood pressure is 120/70 mm Hg, heart rate is 75/min, respiratory rate is 14/min, and temperature is 37.6°C (99.7°F). On physical examination, the patient’s neck is stiff and she also has bilateral shoulder tenderness; muscle strength is intact. Laboratory work is performed and presented below:
Hemoglobin 12.9 g/dL
Hematocrit 37.7% 
Leukocyte count 5,500/mm3
Neutrophils 65%
Lymphocytes 30%
Monocytes 5%
Mean corpuscular volume 82.2 μm3
Platelet count 190,000/mm3
Erythrocyte sedimentation rate 65 mm/h
C-reactive protein 44 mg/dL
For which of the symptoms below should the patient be screened?

**Options:**
- **A:** Jaw claudication
- **B:** Heliotrope rash
- **C:** Gastroesophageal reflux
- **D:** Pink plaques with silvery scales

**Ground Truth Answer:** Option **A**

---

### 🔴 [85/89] ID: `test-00084`
**Question Stem:**
A 30-year-old woman comes to the physician because she has been unable to conceive for 3 years. Analysis of her husband's semen has shown normal sperm counts during this time. The patient also reports episodic pelvic and back pain accompanied by painful diarrhea for 5 years. She has about one such episode on average per month for 4–6 days. She has taken ibuprofen for the pain, which has provided some relief. Menses have occurred at regular 29-day intervals since menarche at the age of 14 years and last for 7 days. She is sexually active with her husband and does not use contraception. Vital signs are within normal limits. Pelvic and bimanual examinations are normal; rectal examination is unremarkable. A hysterosalpingogram 6 months prior showed normal results. Which of the following is the most likely underlying mechanism of this patient's symptoms?

**Options:**
- **A:** Loss of fallopian tube function following infection
- **B:** Smooth muscle tumor arising from the myometrium
- **C:** Endometrial tissue outside the uterine cavity
- **D:** Increased secretion of androgens and luteinizing hormone

**Ground Truth Answer:** Option **C**

---

### 🔴 [86/89] ID: `test-00085`
**Question Stem:**
A 55-year-old truck driver is brought to a physician by his wife. She states that her husband developed a fever and began feeling weak 3 days ago, but has refused medical help. He has been unable to go to work because of his symptoms. The patient has been previously hospitalized for a tricuspid valve replacement surgery 1 year ago and takes aspirin daily. The medical history is also relevant for myocardial infarction 3 years ago and hypertension for the past 10 years, for which he takes lisinopril. His blood pressure is 140/80 mm Hg, the pulse is 82/min, the respirations are 18/minute, and the temperature is 37.2°C (98.9°F). On examination, several hemorrhages are noted on the nail beds of several fingers. Which of the following findings would be most helpful in establishing a diagnosis?

**Options:**
- **A:** Bicuspid valve
- **B:** Friable irregular masses attached to the valve
- **C:** Papillary muscle rupture
- **D:** Annular calcification

**Ground Truth Answer:** Option **B**

---

### 🔴 [87/89] ID: `test-00086`
**Question Stem:**
A previously healthy 30-year-old woman comes to the physician for the evaluation of pain during sexual intercourse for 6 months. She also reports frequent episodes of crampy pelvic pain that starts one day before menses and lasts for 7 days. Her symptoms are not relieved with pain medication. Menses occur at regular 28-day intervals and last 5 days. Her last menstrual period was 2 weeks ago. She is sexually active with her husband. She uses a combined oral contraceptive pill. Her vital signs are within normal limits. Physical examination shows rectovaginal tenderness. Cervical and urethral swabs are negative. Transvaginal ultrasonography shows no abnormalities. Which of the following is the most appropriate next step in management?

**Options:**
- **A:** Measurement of CA-125 levels
- **B:** Hysterectomy
- **C:** Laparoscopy
- **D:** Hysteroscopy

**Ground Truth Answer:** Option **C**

---

### 🔴 [88/89] ID: `test-00087`
**Question Stem:**
A 50-year-old man visits his physician after 20 years of not seeking any medical care. He is concerned about his health after a colleague recently had a heart attack. The patient has no active complaints and says he feels healthy; however, he does not exercise regularly and lives a sedentary lifestyle. He is employed as an administrative position at a local college, and is seated at a desk most of the day. His father had a heart attack at age 54 and his mother is still alive with no health concerns. He does not smoke, only drinks socially, and does not use drugs. Today, his blood pressure is 130/90 mm Hg, pulse is 84/min, and respiratory rate is 14/min. Physical examination reveals an obese male with no significant findings. An ECG shows no abnormalities, and laboratory testing shows the following:
Laboratory test
Serum glucose (fasting)  105 mg/dL
Serum electrolytes 
Sodium  142 mEq/L
Potassium 3.9 mEq/L
Chloride 101 mEq/L
Serum creatinine 0.8 mg/dl
Blood urea nitrogen 10 mg/dl
Cholesterol, total 250 mg/dL
HDL-cholesterol 35 mg/dL
LDL-cholesterol 186 mg/dL
Triglycerides 170 mg/dL
Urinalysis 
Glucose  negative
Ketones negative
Leucocytes negative
Nitrites negative 
Red blood cells (RBC) negative 
Casts negative 
Which of the following lab abnormalities in this patient is an indication for treatment?

**Options:**
- **A:** Blood pressure reading
- **B:** Patient’s weight
- **C:** High LDL-cholesterol
- **D:** Serum glucose level

**Ground Truth Answer:** Option **C**

---

### 🔴 [89/89] ID: `test-00088`
**Question Stem:**
A 26-year-old woman is brought to the emergency department 20 minutes after being involved in a high-speed motor vehicle collision in which she was a restrained passenger. On arrival, she is lethargic and incoherent. She has severe facial lacerations and is in respiratory distress. Her pulse is 130/min, respirations are 29/min, and blood pressure is 90/58 mm Hg. Pulse oximetry on room air shows an oxygen saturation of 70%. Examination shows multiple facial lacerations. There is dullness to percussion and decreased breath sounds over the left lung base. Abdominal examination shows diffuse tenderness with no guarding or rebound. Bowel sounds are normal. The remainder of the examination shows no abnormalities. Her hemoglobin concentration is 12.1 g/dL. An x-ray of the chest shows a fractured left second rib, depression of the left mainstem bronchus, deviation of the nasogastric tube to the right, and a widened mediastinum. Which of the following is the most likely diagnosis?

**Options:**
- **A:** Diaphragmatic rupture
- **B:** Traumatic bronchial rupture
- **C:** Thoracic aortic rupture
- **D:** Tension pneumothorax

**Ground Truth Answer:** Option **C**

---

