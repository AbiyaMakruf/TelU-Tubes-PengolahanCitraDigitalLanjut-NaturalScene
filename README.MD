# 🏞️ Klasifikasi Pemandangan Alam dengan Deep Learning & Pengolahan Citra Digital

## 🌟 Tujuan Proyek (Project Goal)

Mengembangkan solusi untuk masalah pengenalan citra yang menantang, yaitu klasifikasi pemandangan alam, dengan mengintegrasikan alat pengolahan citra digital fundamental dan menerapkan metode *state-of-the-art* (SOTA) dalam *deep learning* untuk klasifikasi.

---

## 📌 Ikhtisar (Overview)

Proyek ini menggali konsep inti dalam **Pengolahan Citra Digital (PCD)**, yang diperkuat dengan **teknik klasifikasi citra modern**. Tujuannya adalah membangun sistem *end-to-end* yang tidak hanya meningkatkan kualitas citra melalui teknik pra-pemrosesan tetapi juga secara akurat mengklasifikasikan citra pemandangan alam ke dalam kategori yang relevan berdasarkan fitur yang diekstraksi secara otomatis oleh model *deep learning*. Pendekatan ini bertujuan untuk memperdalam pemahaman tentang aspek teoretis dan praktis dari aplikasi pengolahan citra dalam skenario dunia nyata, khususnya menggunakan dataset Intel Image Classification dan arsitektur MobileNetV3.

---

## 🛠️ Implementasi Spesifik Proyek Ini

* **Masalah yang Dipecahkan:** Klasifikasi otomatis gambar pemandangan alam ke dalam 6 kategori: bangunan, hutan, gletser, gunung, laut, dan jalan.
* **Dataset yang Digunakan:** [Intel Image Classification dari Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
* **Model Utama:** MobileNetV3 (Small & Large) dengan pendekatan *fine-tuning* dari bobot *pre-trained* ImageNet.
* **Tools & Frameworks:** Python, TensorFlow, Keras, OpenCV, Scikit-learn, Streamlit.
* **Aplikasi Web Demo:** Sebuah aplikasi interaktif berbasis Streamlit untuk demonstrasi klasifikasi gambar secara *real-time* oleh pengguna.

---

## 📦 Komponen Proyek (Project Components) Sesuai Tugas Awal

Berikut adalah komponen umum yang menjadi acuan dalam pengembangan proyek ini, beserta adaptasinya:

### 1. Pra-pemrosesan (Pre-processing)

* **Pengurangan Noise (Noise reduction)**: Implementasi filter seperti Gaussian dan Median filter disediakan dan dapat dieksperimenkan pengaruhnya terhadap performa model *deep learning*.
* **Peningkatan Kualitas Citra (Image enhancement)**: Teknik seperti *histogram equalization* disediakan. Normalisasi citra (seperti yang dilakukan oleh `preprocess_input` pada MobileNetV3) juga merupakan bagian krusial dari tahap ini.

### 2. Ekstraksi Fitur (Feature Extraction)

* Ekstraksi fitur relevan dilakukan secara **otomatis oleh lapisan-lapisan konvolusi pada model MobileNetV3**. Ini merupakan pendekatan modern yang menggantikan ekstraksi fitur manual seperti analisis tekstur atau deskriptor bentuk untuk tugas klasifikasi citra kompleks.

### 3. Klasifikasi (Classification)

* Mengimplementasikan algoritma klasifikasi berbasis *neural network* yang merupakan bagian integral dari arsitektur MobileNetV3 (lapisan *fully connected* dan *softmax* di bagian *head*).
* Melatih *classifier* (fine-tuning MobileNetV3) menggunakan fitur yang diekstraksi secara otomatis dan dataset Intel Image Classification yang berlabel.

### 4. Pasca-pemrosesan (Post-processing) & Evaluasi

* Mengevaluasi hasil klasifikasi menggunakan metrik standar seperti **akurasi (accuracy)**, **presisi (precision)**, **recall**, dan **F1-score**.
* Melakukan *fine-tuning* model klasifikasi (MobileNetV3) untuk meningkatkan performa, termasuk penyesuaian *learning rate* dan strategi *unfreezing layer*.
* Visualisasi hasil seperti *confusion matrix* dan **Grad-CAM** untuk interpretasi keputusan model.

---

## 🚀 Struktur Direktori Proyek (Contoh)
proyek_klasifikasi_alam/
├── dataset/                     # Dataset Intel Image Classification
│   ├── train/
│   └── test/
├── saved_models/                # Model .keras yang sudah dilatih
│   ├── mobilenet_v3_small_finetune_best.keras
│   └── mobilenet_v3_large_finetune_best.keras
├── results/                     # Tempat menyimpan hasil evaluasi (gambar, csv, dll.)
│   ├── images/                  # Contoh: confusion matrices, training histories
│   │   ├── cm_small_baseline.png
│   │   └── history_large_finetune.png
│   └── reports/                 # (Opsional) laporan dalam format lain
├── sample_test_images/          # Gambar contoh untuk demo aplikasi Streamlit
│   ├── buildings.jpg
│   └── ...
├── app_utils/                   # (Opsional) Skrip bantuan Python
│   ├── init.py
│   └── grad_cam_utils.py
├── streamlit_app.py             # Skrip utama aplikasi Streamlit
├── requirements.txt             # Daftar dependensi Python
└── README.md                    # File ini

---

## ⚙️ Cara Menjalankan Proyek

1.  **Clone Repository (Jika Ada):**
    ```bash
    git clone [URL_REPOSITORY_ANDA]
    cd proyek_klasifikasi_alam
    ```
2.  **Siapkan Dataset:**
    * Unduh dataset [Intel Image Classification dari Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
    * Ekstrak dan letakkan isinya (folder `seg_train`, `seg_test`, `seg_pred`) ke dalam direktori `dataset/`. Ubah nama `seg_train` menjadi `train` dan `seg_test` menjadi `test` jika kode Python Anda menggunakan nama tersebut.
3.  **Buat Virtual Environment (Direkomendasikan):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
4.  **Instal Dependensi:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Pastikan file `requirements.txt` berisi semua pustaka yang dibutuhkan seperti `tensorflow`, `opencv-python`, `matplotlib`, `scikit-learn`, `streamlit`, `numpy`, `pillow`)*
5.  **Latih Model (Jika Belum Ada Model Tersimpan):**
    * Jalankan skrip pelatihan Python Anda (misalnya, `train_model.py` - *Anda perlu membuat skrip ini atau menyertakan notebook Jupyter jika pelatihan belum dilakukan*).
    * Pastikan model terbaik disimpan di direktori `saved_models/`.
6.  **Jalankan Aplikasi Web Streamlit:**
    * Pastikan path ke model (`.keras`) dan gambar contoh di `streamlit_app.py` sudah benar.
    * Pastikan path ke gambar hasil evaluasi (confusion matrix, grafik training) di `streamlit_app.py` pada halaman "Performa Model" juga sudah benar.
    ```bash
    streamlit run streamlit_app.py
    ```
    Aplikasi akan terbuka di browser Anda.

---

## 📈 Hasil Utama (Contoh Berdasarkan Data Anda)

| Model                                         | Test Accuracy | Test Loss | Catatan                                    |
| :-------------------------------------------- | :------------ | :-------- | :----------------------------------------- |
| MobileNetV3 Small (Baseline - Feat. Extract.) | 89.27%        | 0.2700    | Performa awal hanya dengan melatih *head*.   |
| MobileNetV3 Large (Baseline - Feat. Extract.) | 91.53%        | 0.2310    | Model lebih besar menunjukkan dasar lebih baik. |
| **MobileNetV3 Small (Fine-tuned)** | **92.93%** | **0.2106**| Peningkatan signifikan setelah fine-tuning. |
| **MobileNetV3 Large (Fine-tuned)** | **93.63%** | **0.2376**| Performa terbaik, meskipun loss sedikit naik. |

*Fine-tuning secara konsisten meningkatkan performa kedua model. MobileNetV3 Large (Fine-tuned) mencapai akurasi tertinggi.*

---

##  deliverables Project Deliverables (Sesuai Silabus)

* ✅ **Aplikasi Web Fungsional:** Implementasi menggunakan Streamlit untuk klasifikasi citra dan visualisasi Grad-CAM.
* 📝 **Laporan Komprehensif** yang merinci:
    * Latar Belakang Teori (PCD, CNN, MobileNetV3, Transfer Learning, Grad-CAM)
    * Detail Implementasi (Dataset, Pra-pemrosesan, Arsitektur Model, Proses Training & Fine-tuning, Pengembangan Aplikasi Web)
    * Hasil Eksperimen (Metrik evaluasi, Analisis Performa, Visualisasi)
* 🎤 **Presentasi** yang merangkum:
    * Tujuan dan Ruang Lingkup Proyek
    * Metodologi yang Digunakan
    * Hasil dan Pembahasan
    * Kesimpulan dan Wawasan Utama

🔗 Contoh ide proyek (Referensi Umum): [Stanford EE368 Project List](https://web.stanford.edu/class/ee368/Project)

---

## 📅 Milestones & Grading Policy (Sesuai Silabus)

| Deliverable            | Weight | Deadline      | Week |
| :--------------------- | :----- | :------------ | :--- |
| 📝 Proposal            | 10%    | 19 April 2025 | W8   |
| 📈 Progress Report     | 20%    | 10 Mei 2025   | W11  |
| 📦 Final Report & Code | 45%    | 31 Mei 2025   | W14  |
| 🎤 Final Presentation  | 25%    | 07 Juni 2025  | W15  |

---

## ✅ Proposal (200–400 words) - Panduan

Proposal Anda harus menjawab:

* Apa masalah yang Anda selidiki dan mengapa itu menarik? (Contoh: klasifikasi pemandangan alam untuk aplikasi XYZ, menarik karena kompleksitas visual dan relevansi).
* Literatur apa yang akan Anda tinjau? (Contoh: Paper tentang MobileNetV3, Grad-CAM, teknik pra-pemrosesan citra).
* Dataset apa yang akan digunakan atau dikumpulkan? (Contoh: Intel Image Classification dari Kaggle).
* Algoritma atau metode apa yang akan Anda terapkan? Apakah Anda akan menggunakan atau mengadaptasi implementasi yang sudah ada? (Contoh: Fine-tuning MobileNetV3 dari Keras, implementasi Grad-CAM berdasarkan contoh).
* Bagaimana Anda akan mengevaluasi hasil (kualitatif: plot/gambar seperti confusion matrix, Grad-CAM; kuantitatif: metrik seperti akurasi, F1-score)?

---

## ✅ Progress Report (2–3 pages) - Panduan Struktur

1.  **Judul & Penulis**
2.  **Pendahuluan** – Deskripsi singkat masalah dan rencana pendekatan Anda.
3.  **Pernyataan Masalah** – Detail tentang dataset yang digunakan, hasil yang diharapkan, dan strategi evaluasi.
4.  **Pendekatan Teknis** – Metode yang diusulkan untuk menyelesaikan masalah (misalnya, arsitektur model spesifik, langkah pra-pemrosesan).
5.  **Desain Antarmuka Pengguna (UI)** – Tata letak dan desain interaksi untuk aplikasi (misalnya, sketsa atau deskripsi alur aplikasi Streamlit).
6.  **Hasil Awal/Sementara** – Hasil yang diperoleh sejauh ini, beserta evaluasinya (misalnya, akurasi dari model baseline).

---

## ✅ Final Report (6–8 pages) - Panduan Struktur

1.  **Judul & Penulis**
2.  **Abstrak** – Maksimal 300 kata yang merangkum masalah, metode, dan hasil utama.
3.  **Pendahuluan (10%)** – Deskripsikan masalah yang Anda kerjakan, mengapa itu penting, dan gambaran umum hasil Anda.
4.  **Karya Terkait (10%)** – Diskusikan publikasi yang berkaitan dengan proyek Anda. Bagaimana pendekatan Anda mirip atau berbeda dari yang lain?
5.  **Data (10%)** – Deskripsikan data yang Anda gunakan. Jenis datanya apa? Dari mana asalnya? Berapa banyak data yang Anda gunakan? Apakah Anda melakukan pra-pemrosesan, pemfilteran, atau perlakuan khusus lainnya?
6.  **Metode (30%)** – Diskusikan pendekatan Anda untuk menyelesaikan masalah. Mengapa pendekatan Anda tepat? Apakah Anda mempertimbangkan pendekatan alternatif? Anda harus menunjukkan bahwa Anda telah menerapkan ide dan keterampilan yang dibangun selama perkuliahan. Sertakan gambar, diagram, atau tabel untuk menjelaskan metode Anda atau membandingkannya dengan metode lain.
7.  **Eksperimen (30%)** – Diskusikan eksperimen yang Anda lakukan untuk menunjukkan bahwa pendekatan Anda menyelesaikan masalah. Ini bisa berupa perbandingan dengan metode lain, studi ablasi, eksperimen dengan hyperparameter atau arsitektur yang berbeda, penggunaan teknik visualisasi (seperti Grad-CAM), diskusi mode kegagalan umum model Anda, dll. Sertakan grafik, tabel, atau gambar lain untuk mengilustrasikan hasil eksperimen Anda.
8.  **Kesimpulan (5%)** – Rangkum hasil utama Anda - apa yang telah Anda pelajari?
9.  **Saran untuk Pengembangan Selanjutnya** atau aplikasi baru dari ide Anda.
10. **Penulisan / Format (5%)** – Apakah laporan Anda ditulis dengan jelas dan diformat dengan baik?

---

## 💭 Refleksi dan Pembelajaran (Contoh untuk Ditambahkan di Laporan Akhir)

* Pentingnya *fine-tuning* dibandingkan hanya *feature extraction* pada model *pre-trained*.
* Peran Grad-CAM dalam memahami keputusan model dan melakukan *debugging* kualitatif.
* Tantangan dalam memilih hyperparameter yang optimal.
* Efektivitas MobileNetV3 sebagai arsitektur yang efisien namun tetap akurat.
* Pengalaman dalam membangun aplikasi web interaktif untuk model *machine learning*.