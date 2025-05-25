import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import os

try:
    from app_utils.grad_cam_utils import make_gradcam_heatmap, generate_gradcam_overlay
except ImportError:
    st.error("file app_utils/grad_cam_utils.py tidak ditemukan. Pastikan file tersebut ada di direktori yang benar.")

# --- Konfigurasi Aplikasi ---
st.set_page_config(page_title="Klasifikasi Pemandangan Alam", layout="wide")

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMG_WIDTH, IMG_HEIGHT = 224, 224

# Path ke model
MODEL_PATH_SMALL = 'results/model/finetune/mobilenet_v3_small_finetune_best.keras'
MODEL_PATH_LARGE = 'results/model/finetune/mobilenet_v3_large_finetune_best.keras'

# Path ke gambar contoh
SAMPLE_IMAGE_DIR = 'sample_test_images/'
sample_images_data = {
    "Bangunan": os.path.join(SAMPLE_IMAGE_DIR, "buildings.jpg"),
    "Hutan": os.path.join(SAMPLE_IMAGE_DIR, "forest.jpg"),
    "Gletser": os.path.join(SAMPLE_IMAGE_DIR, "glacier.jpg"),
    "Gunung": os.path.join(SAMPLE_IMAGE_DIR, "mountain.jpg"),
    "Laut": os.path.join(SAMPLE_IMAGE_DIR, "sea.jpg"),
    "Jalan": os.path.join(SAMPLE_IMAGE_DIR, "street.jpg"),
}
# Filter sample images
valid_sample_images = {name: path for name, path in sample_images_data.items() if os.path.exists(path)}

# --- PATH UNTUK GAMBAR HASIL EVALUASI ---
PATH_CONFUSION_MATRIX_SMALL_BASELINE = "results/graph/baseline/mobilenetv3small_baseline_featureextract_confusion_matrix.png"
PATH_TRAINING_HISTORY_SMALL_BASELINE = "results/graph/baseline/mobilenetv3small_baseline_featureextract_accuracy_loss.png"
PATH_CONFUSION_MATRIX_LARGE_BASELINE = "results/graph/baseline/mobilenetv3large_baseline_featureextract_confusion_matrix.png"
PATH_TRAINING_HISTORY_LARGE_BASELINE = "results/graph/baseline/mobilenetv3large_baseline_featureextract_accuracy_loss.png"

PATH_CONFUSION_MATRIX_SMALL_FINETUNE = "results/graph/finetune/mobilenetv3_small_confusion_matrix.png"
PATH_TRAINING_HISTORY_SMALL_FINETUNE = "results/graph/finetune/MobileNetV3_Small.png"
PATH_CONFUSION_MATRIX_LARGE_FINETUNE = "results/graph/finetune/mobilenetv3_large_confusion_matrix.png"
PATH_TRAINING_HISTORY_LARGE_FINETUNE = "results/graph/finetune/MobileNetV3_Large.png"
# --- ----------------------------------------------------------- ---


# --- Fungsi Bantuan ---
@st.cache_resource # Cache resource agar model tidak di-load ulang setiap interaksi
def load_keras_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error memuat model dari {model_path}: {e}")
        return None

def preprocess_image_for_model(pil_image):
    """ Preprocess PIL image untuk input MobileNetV3 """
    img_resized = pil_image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return mobilenet_v3_preprocess_input(img_array_expanded), img_array

def get_last_conv_layer_name(model):
    """ Mencari nama layer konvolusi terakhir sebelum GlobalAveragePooling """
    if model == 'small':
        return 'activation_17'
    elif model == 'large':
        return 'activation_37'

# --- Muat Model (hanya jika diperlukan di halaman demo) ---
model_small = None
model_large = None

# --- Navigasi / Menu ---
st.sidebar.title("Navigasi Proyek")
menu_options = ["🏠 Home", "🖼️ Demo Klasifikasi", "📊 Performa Model", "📚 Tentang Dataset"]
page = st.sidebar.radio("Pilih Halaman:", menu_options)

# --- Konten Halaman ---
if page == "🏠 Home":
    st.title("Selamat Datang di Aplikasi Klasifikasi Pemandangan Alam! 🏞️")
    st.markdown("""
    Aplikasi ini mendemonstrasikan kemampuan model *deep learning* (MobileNetV3 Small & Large)
    untuk mengklasifikasikan gambar pemandangan alam ke dalam enam kategori:
    **bangunan, hutan, gletser, gunung, laut, dan jalan.**

    **Tujuan Proyek:**
    Proyek ini bertujuan untuk menyelesaikan permasalahan klasifikasi pemandangan alam
    menggunakan dataset Intel Image Classification. Pengenalan alam secara otomatis merupakan
    tantangan penting dalam berbagai aplikasi dunia nyata. Proyek ini menggabungkan teknik
    fundamental pengolahan citra (secara konseptual dalam pra-pemrosesan dataset) dengan
    metode klasifikasi modern berbasis *deep learning* untuk meningkatkan performa pengenalan.

    **Cara Menggunakan Halaman Demo:**
    1.  Pilih halaman **"🖼️ Demo Klasifikasi"** dari menu navigasi di sidebar.
    2.  Pilih model yang ingin Anda gunakan (MobileNetV3 Small atau Large).
    3.  Unggah gambar Anda sendiri melalui *drag and drop* atau tombol *browse*, ATAU
    4.  Pilih salah satu gambar contoh yang disediakan.
    5.  Hasil klasifikasi beserta visualisasi Grad-CAM akan ditampilkan.

    Silakan jelajahi menu lain untuk informasi lebih lanjut tentang dataset dan performa model.
    """)


elif page == "🖼️ Demo Klasifikasi":
    st.header("🖼️ Demo Klasifikasi Gambar Pemandangan Alam")

    model_choice = st.selectbox("Pilih Model (Fine-tuned):", ("MobileNetV3 Small (Fine-tuned)", "MobileNetV3 Large (Fine-tuned)"))

    active_model = None
    model_name_for_gradcam = "" 
    if model_choice == "MobileNetV3 Small (Fine-tuned)":
        if model_small is None: 
            model_small = load_keras_model(MODEL_PATH_SMALL)
        active_model = model_small
        model_name_for_gradcam = "small" 
    else:
        if model_large is None: 
            model_large = load_keras_model(MODEL_PATH_LARGE)
        active_model = model_large
        model_name_for_gradcam = "large"

    if active_model is None:
        st.error("Model yang dipilih gagal dimuat. Silakan periksa path model atau coba lagi.")
    else:
        st.subheader("Unggah Gambar Anda atau Pilih Contoh:")
        uploaded_file = st.file_uploader("Seret dan lepas gambar di sini, atau klik untuk memilih file", type=["jpg", "jpeg", "png"])
        
        st.markdown("<h5 style='text-align: center; color: grey;'>ATAU</h5>", unsafe_allow_html=True)
        
        st.subheader("Pilih Gambar Contoh:")
        if not valid_sample_images:
            st.warning("Tidak ada gambar contoh yang valid ditemukan. Pastikan path dan file benar.")
        else:
            cols = st.columns(min(len(valid_sample_images), 6))
            selected_sample_path = None
            for i, (name, path) in enumerate(valid_sample_images.items()):
                with cols[i % len(cols)]:
                    try:
                        sample_img_pil = Image.open(path)
                        st.image(sample_img_pil, caption=name, width=100)
                        if st.button(f"Gunakan {name}", key=f"sample_{name}"):
                            uploaded_file = path 
                            selected_sample_path = path
                    except FileNotFoundError:
                        st.error(f"File contoh {path} tidak ditemukan.")

        input_image_pil = None
        if uploaded_file is not None:
            try:
                if isinstance(uploaded_file, str): 
                    input_image_pil = Image.open(uploaded_file)
                else: 
                    input_image_pil = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Error membuka gambar: {e}")
                input_image_pil = None
        
        if input_image_pil:
            st.subheader("Hasil Analisis Gambar:")
            col1, col2 = st.columns(2)
            with col1:
                st.image(input_image_pil, caption="Gambar Asli", use_container_width=True)

            processed_image_tensor, original_img_array_for_gradcam = preprocess_image_for_model(input_image_pil)
            
            try:
                predictions = active_model.predict(processed_image_tensor)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class_name = CLASSES[predicted_class_index]
                confidence = predictions[0][predicted_class_index] * 100

                st.success(f"**Prediksi Kelas:** {predicted_class_name}")
                st.info(f"**Kepercayaan:** {confidence:.2f}%")

                with col2:
                    st.markdown("<h6>Visualisasi Grad-CAM</h6>", unsafe_allow_html=True)
                    try:
                        last_conv_name = get_last_conv_layer_name(model_name_for_gradcam)
                        img_for_gradcam_uint8 = original_img_array_for_gradcam.astype(np.uint8)
                        heatmap = make_gradcam_heatmap(processed_image_tensor, active_model, last_conv_name, pred_index=predicted_class_index)
                        grad_cam_image = generate_gradcam_overlay(img_for_gradcam_uint8, heatmap)
                        st.image(grad_cam_image, caption=f"Grad-CAM untuk '{predicted_class_name}'", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error saat membuat Grad-CAM: {e}")
                        st.warning(f"Detail error: {str(e)}")
                        st.warning("Pastikan nama layer konvolusi terakhir (`get_last_conv_layer_name`) untuk Grad-CAM sudah benar dan terverifikasi dari `model.summary()`.")

            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {e}")

elif page == "📊 Performa Model":
    st.header("📊 Ringkasan Performa Model pada Test Set")
    st.markdown("Berikut adalah hasil evaluasi dari model MobileNetV3 Small dan Large, baik tahap *baseline (feature extraction)* maupun setelah *fine-tuning*.")

    performance_data_actual = {
        "MobileNetV3 Small (Baseline)": {
            "Test Accuracy": "89.27%",
            "Test Loss": "0.2700",
            "Epoch Terakhir Dilaporkan": "28/50 (val_accuracy: 0.8894)",
            "Path Confusion Matrix": PATH_CONFUSION_MATRIX_SMALL_BASELINE,
            "Path Grafik Training": PATH_TRAINING_HISTORY_SMALL_BASELINE,
            "F1-Score (Macro Avg)": "0.89",
        },
        "MobileNetV3 Large (Baseline)": {
            "Test Accuracy": "91.53%",
            "Test Loss": "0.2310",
            "Epoch Terakhir Dilaporkan": "29/50 (val_accuracy: 0.9108)",
            "Path Confusion Matrix": PATH_CONFUSION_MATRIX_LARGE_BASELINE,
            "Path Grafik Training": PATH_TRAINING_HISTORY_LARGE_BASELINE,
            "F1-Score (Macro Avg)": "0.92",
        },
        "MobileNetV3 Small (Fine-tuned)": {
            "Test Accuracy": "92.93%",
            "Test Loss": "0.2106",
            "Epoch Terakhir Dilaporkan": "37/50 (train_accuracy: 0.9605)",
            "Path Confusion Matrix": PATH_CONFUSION_MATRIX_SMALL_FINETUNE,
            "Path Grafik Training": PATH_TRAINING_HISTORY_SMALL_FINETUNE,
            "F1-Score (Macro Avg)": "0.93",
        },
        "MobileNetV3 Large (Fine-tuned)": {
            "Test Accuracy": "93.63%",
            "Test Loss": "0.2376",
            "Epoch Terakhir Dilaporkan": "43/50 (train_accuracy: 0.9944)",
            "Path Confusion Matrix": PATH_CONFUSION_MATRIX_LARGE_FINETUNE,
            "Path Grafik Training": PATH_TRAINING_HISTORY_LARGE_FINETUNE,
            "F1-Score (Macro Avg)": "0.94",
        }
    }

    # --- 1. Membuat Tabel Perbandingan Metrik di Awal ---
    st.subheader("Tabel Ringkasan Performa Model")

    # Siapkan data untuk tabel
    summary_table_data = []
    for model_name, metrics in performance_data_actual.items():
        summary_table_data.append({
            "Model": model_name,
            "Test Accuracy": metrics["Test Accuracy"],
            "Test Loss": metrics["Test Loss"],
            "F1-Score (Macro Avg)": metrics.get("F1-Score (Macro Avg)", "N/A")
        })

    # Tampilkan tabel ringkasan
    st.dataframe(summary_table_data, hide_index=True, use_container_width=True)
    st.markdown("---")

    for model_name, metrics in performance_data_actual.items():
        st.subheader(model_name)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"- **Test Accuracy:** {metrics['Test Accuracy']}")
            st.markdown(f"- **Test Loss:** {metrics['Test Loss']}")
            st.markdown(f"- **Epoch Terakhir Dilaporkan:** {metrics['Epoch Terakhir Dilaporkan']}")
            st.markdown(f"- **F1-Score (Macro Avg):** {metrics.get('F1-Score (Macro Avg)', 'N/A')}")

        colgraph, colconf = st.columns(2)
        with colgraph:
            st.markdown("**Grafik Training History:**")
            if os.path.exists(metrics["Path Grafik Training"]):
                st.image(metrics["Path Grafik Training"], caption=f"Training History - {model_name}", use_container_width=True)
            else:
                st.warning(f"Gambar training history tidak ditemukan di: {metrics['Path Grafik Training']}. Harap perbarui path.")

        with colconf:
            st.markdown("**Confusion Matrix:**")
            if os.path.exists(metrics["Path Confusion Matrix"]):
                st.image(metrics["Path Confusion Matrix"], caption=f"Confusion Matrix - {model_name}", use_container_width=True)
            else:
                st.warning(f"Gambar confusion matrix tidak ditemukan di: {metrics['Path Confusion Matrix']}. Harap perbarui path.")

        st.markdown("---")


elif page == "📚 Tentang Dataset":
    st.header("📚 Tentang Dataset: Intel Image Classification")
    st.markdown("""
    Dataset yang digunakan dalam proyek ini adalah **Intel Image Classification Dataset**.
    Dataset ini awalnya dirilis oleh Intel untuk sebuah kompetisi klasifikasi gambar di platform Kaggle.
    
    **Fitur Utama Dataset:**
    - **Jumlah Kelas:** 6 kategori pemandangan alam utama:
        - `buildings` (bangunan)
        - `forest` (hutan)
        - `glacier` (gletser)
        - `mountain` (gunung)
        - `sea` (laut)
        - `street` (jalan)
    - **Ukuran Dataset:**
        - Sekitar 14,000 gambar untuk data latih (`train`).
        - Sekitar 3,000 gambar untuk data prediksi/uji (`test`).
    - **Ukuran Gambar:** Gambar-gambar memiliki resolusi asli 150x150 piksel. Untuk model MobileNetV3, gambar-gambar ini diubah ukurannya menjadi 224x224 piksel selama pra-pemrosesan.
    - **Sumber:** Gambar-gambar dikumpulkan dari berbagai sumber dan mencakup variasi yang cukup baik dalam setiap kelas.

    Anda dapat menemukan dataset ini di [Kaggle: Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
    """)

    PATH_GAMBAR_DISTRIBUSI_KELAS_TRAINING = "results/data/distribusi_training.png" 
    PATH_GAMBAR_DISTRIBUSI_KELAS_TESTING = "results/data/distribusi_test.png" 
    if os.path.exists(PATH_GAMBAR_DISTRIBUSI_KELAS_TRAINING and os.path.exists(PATH_GAMBAR_DISTRIBUSI_KELAS_TESTING)):
        st.subheader("Distribusi Kelas dalam Dataset")
        st.markdown("Berikut adalah distribusi jumlah gambar untuk setiap kelas dalam dataset:")
        col1, col2 = st.columns(2)
        with col1:
            st.image(PATH_GAMBAR_DISTRIBUSI_KELAS_TRAINING, caption="Distribusi Kelas pada Data Latih")
        with col2:
            st.image(PATH_GAMBAR_DISTRIBUSI_KELAS_TESTING, caption="Distribusi Kelas pada Data Uji")
        st.markdown("Gambar di atas menunjukkan jumlah gambar untuk setiap kelas dalam data latih dan uji. Distribusi yang seimbang antar kelas sangat penting untuk performa model yang baik.")
        st.markdown("---")
    else:
        st.info("Anda dapat menambahkan gambar plot distribusi kelas Anda di sini jika sudah ada.")
    # --- ----------------------------------------- ---