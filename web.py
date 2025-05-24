import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import os
import cv2 # Diperlukan untuk beberapa operasi gambar jika tidak di utils

# (Opsional) Import fungsi Grad-CAM dari utils
# Jika Anda memindahkannya ke app_utils/grad_cam_utils.py
try:
    from app_utils.grad_cam_utils import make_gradcam_heatmap, generate_gradcam_overlay
except ImportError:
    st.error("Pastikan file app_utils/grad_cam_utils.py ada dan benar jika Anda memisahkan fungsi Grad-CAM.")
    # Definisikan fungsi Grad-CAM di sini jika tidak diimpor (seperti di atas)
    # --- PASTE FUNGSI make_gradcam_heatmap dan generate_gradcam_overlay DARI ATAS JIKA TIDAK IMPORT ---
    def make_gradcam_heatmap(img_array_processed, model, last_conv_layer_name, pred_index=None):
        grad_model = tf.keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array_processed)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
        return heatmap.numpy()

    def generate_gradcam_overlay(original_img_array_uint8, heatmap, alpha=0.5):
        heatmap_resized = cv2.resize(heatmap, (original_img_array_uint8.shape[1], original_img_array_uint8.shape[0]))
        heatmap_rgb = np.uint8(255 * heatmap_resized)
        heatmap_rgb = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(heatmap_rgb, alpha, original_img_array_uint8, 1 - alpha, 0)
        return superimposed_img
    # --- END PASTE ---


# --- Konfigurasi Aplikasi ---
st.set_page_config(page_title="Klasifikasi Pemandangan Alam", layout="wide")

# Kelas target (SESUAIKAN DENGAN DATASET ANDA)
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMG_WIDTH, IMG_HEIGHT = 224, 224

# Path ke model (SESUAIKAN)
MODEL_PATH_SMALL = 'saved_models/mobilenet_v3_small_finetune_best.keras'
MODEL_PATH_LARGE = 'saved_models/mobilenet_v3_large_finetune_best.keras'

# Path ke gambar contoh (SESUAIKAN)
SAMPLE_IMAGE_DIR = 'sample_test_images/'
sample_images_data = {
    "Bangunan": os.path.join(SAMPLE_IMAGE_DIR, "buildings_sample.jpg"),
    "Hutan": os.path.join(SAMPLE_IMAGE_DIR, "forest_sample.jpg"),
    "Gletser": os.path.join(SAMPLE_IMAGE_DIR, "glacier_sample.jpg"),
    "Gunung": os.path.join(SAMPLE_IMAGE_DIR, "mountain_sample.jpg"),
    "Laut": os.path.join(SAMPLE_IMAGE_DIR, "sea_sample.jpg"),
    "Jalan": os.path.join(SAMPLE_IMAGE_DIR, "street_sample.jpg"),
}
# Filter sample images yang benar-benar ada filenya
valid_sample_images = {name: path for name, path in sample_images_data.items() if os.path.exists(path)}


# --- Fungsi Bantuan ---
@st.cache_resource # Cache resource agar model tidak di-load ulang setiap interaksi
def load_keras_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        st.success(f"Model berhasil dimuat dari: {model_path}")
        return model
    except Exception as e:
        st.error(f"Error memuat model dari {model_path}: {e}")
        return None

def preprocess_image_for_model(pil_image):
    """ Preprocess PIL image untuk input MobileNetV3 """
    img_resized = pil_image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img_resized) # (height, width, channels), float32
    img_array_expanded = np.expand_dims(img_array, axis=0) # (1, height, width, channels)
    return mobilenet_v3_preprocess_input(img_array_expanded), img_array # Kembalikan juga array original (sebelum preprocess_input) untuk gradcam

def get_last_conv_layer_name(model):
    """ Mencari nama layer konvolusi terakhir sebelum GlobalAveragePooling """
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4 and isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.SeparableConv2D)):
            if 'predictions' not in layer.name.lower() and 'dense' not in layer.name.lower() and 'global' not in layer.name.lower():
                 return layer.name
    st.warning("Tidak dapat menemukan nama layer konvolusi terakhir secara otomatis untuk Grad-CAM. Menggunakan fallback.")
    # Fallback (HARUS DIVERIFIKASI DARI model.summary() Anda)
    if 'small' in model.name.lower() or (hasattr(model, '_name') and 'small' in model._name.lower()): # Cek _name juga
        return 'expanded_conv_10/project/BatchNorm' # CONTOH untuk MobileNetV3Small
    else:
        return 'expanded_conv_15/project/BatchNorm' # CONTOH untuk MobileNetV3Large

# --- Muat Model ---
# Model dimuat sekali saat aplikasi dimulai atau saat dipilih
# (Untuk efisiensi, bisa juga hanya load saat benar-benar akan digunakan)
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
    st.image("https://www.researchgate.net/profile/Adrian-Rosebrock/publication/325559039/figure/fig1/AS:633957117681664@1528158885314/Sample-of-images-from-the-Intel-Image-Classification-dataset.png", caption="Contoh gambar dari dataset Intel Image Classification (Sumber: Rosebrock, PyImageSearch)")


elif page == "🖼️ Demo Klasifikasi":
    st.header("🖼️ Demo Klasifikasi Gambar Pemandangan Alam")

    model_choice = st.selectbox("Pilih Model:", ("MobileNetV3 Small", "MobileNetV3 Large"))

    active_model = None
    model_name_for_gradcam = "" # Untuk fallback Grad-CAM layer name
    if model_choice == "MobileNetV3 Small":
        if model_small is None: # Load jika belum
            model_small = load_keras_model(MODEL_PATH_SMALL)
        active_model = model_small
        model_name_for_gradcam = "small"
    else:
        if model_large is None: # Load jika belum
            model_large = load_keras_model(MODEL_PATH_LARGE)
        active_model = model_large
        model_name_for_gradcam = "large"

    if active_model is None:
        st.error("Model yang dipilih gagal dimuat. Silakan periksa path model atau coba lagi.")
    else:
        # --- Input Gambar ---
        st.subheader("Unggah Gambar Anda atau Pilih Contoh:")
        uploaded_file = st.file_uploader("Seret dan lepas gambar di sini, atau klik untuk memilih file", type=["jpg", "jpeg", "png"])
        
        st.markdown("<h5 style='text-align: center; color: grey;'>ATAU</h5>", unsafe_allow_html=True)
        
        st.subheader("Pilih Gambar Contoh:")
        if not valid_sample_images:
            st.warning("Tidak ada gambar contoh yang valid ditemukan. Pastikan path dan file benar.")
        else:
            cols = st.columns(len(valid_sample_images))
            selected_sample_path = None
            for i, (name, path) in enumerate(valid_sample_images.items()):
                with cols[i]:
                    try:
                        sample_img_pil = Image.open(path)
                        st.image(sample_img_pil, caption=name, width=100) # Tampilkan thumbnail
                        if st.button(f"Gunakan {name}", key=f"sample_{name}"):
                            uploaded_file = path # Set uploaded_file ke path jika tombol ditekan
                            selected_sample_path = path
                    except FileNotFoundError:
                        st.error(f"File contoh {path} tidak ditemukan.")

        input_image_pil = None
        if uploaded_file is not None:
            try:
                # Jika tombol sampel ditekan, uploaded_file adalah path string
                if isinstance(uploaded_file, str): 
                    input_image_pil = Image.open(uploaded_file)
                else: # Jika dari file_uploader, itu adalah BytesIO object
                    input_image_pil = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Error membuka gambar: {e}")
                input_image_pil = None
        
        if input_image_pil:
            st.subheader("Gambar yang Diproses:")
            col1, col2 = st.columns(2)
            with col1:
                st.image(input_image_pil, caption="Gambar Asli", use_column_width=True)

            # Pra-pemrosesan dan Prediksi
            processed_image_tensor, original_img_array_for_gradcam = preprocess_image_for_model(input_image_pil)
            
            try:
                predictions = active_model.predict(processed_image_tensor)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class_name = CLASSES[predicted_class_index]
                confidence = predictions[0][predicted_class_index] * 100

                st.success(f"**Prediksi Kelas:** {predicted_class_name}")
                st.info(f"**Kepercayaan:** {confidence:.2f}%")

                # Visualisasi Grad-CAM
                with col2:
                    st.subheader("Visualisasi Grad-CAM")
                    try:
                        # Dapatkan nama layer konvolusi terakhir dari model yang aktif
                        # Perlu verifikasi nama layer ini dari model.summary()
                        last_conv_name = get_last_conv_layer_name(active_model)
                        
                        # Pastikan original_img_array_for_gradcam adalah uint8
                        img_for_gradcam_uint8 = original_img_array_for_gradcam.astype(np.uint8)

                        heatmap = make_gradcam_heatmap(processed_image_tensor, active_model, last_conv_name, pred_index=predicted_class_index)
                        grad_cam_image = generate_gradcam_overlay(img_for_gradcam_uint8, heatmap)
                        st.image(grad_cam_image, caption=f"Grad-CAM untuk '{predicted_class_name}'", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error saat membuat Grad-CAM: {e}")
                        st.warning("Pastikan nama layer konvolusi terakhir untuk Grad-CAM sudah benar dan terverifikasi dari `model.summary()`.")

            except Exception as e:
                st.error(f"Error saat melakukan prediksi: {e}")

elif page == "📊 Performa Model":
    st.header("📊 Ringkasan Performa Model")
    st.markdown("""
    Berikut adalah ringkasan performa (hipotetis) dari model MobileNetV3 Small dan Large pada *test set* setelah pelatihan.
    *(Catatan: Anda harus mengganti nilai-nilai ini dengan hasil aktual dari evaluasi model Anda)*
    """)

    # --- DATA PERFORMA MODEL (GANTI DENGAN HASIL ANDA) ---
    performance_data = {
        "MobileNetV3 Small": {"Akurasi": "92.50%", "F1-Score (Macro Avg)": "0.92", "Catatan": "Cepat dan ringan."},
        "MobileNetV3 Large": {"Akurasi": "94.20%", "F1-Score (Macro Avg)": "0.94", "Catatan": "Lebih akurat, sedikit lebih berat."},
    }
    # --- ---------------------------------------------- ---

    for model_name, metrics in performance_data.items():
        st.subheader(model_name)
        for metric, value in metrics.items():
            st.markdown(f"- **{metric}:** {value}")
        st.markdown("---")
    
    st.markdown("### Contoh Confusion Matrix (Hipotetis)")
    st.markdown("Anda dapat menampilkan gambar confusion matrix yang telah Anda simpan dari proses evaluasi di sini.")
    # Contoh: st.image("path/to/your/confusion_matrix_small.png", caption="Confusion Matrix - MobileNetV3 Small")
    # st.image("path/to/your/confusion_matrix_large.png", caption="Confusion Matrix - MobileNetV3 Large")
    st.info("Untuk implementasi nyata, muat gambar confusion matrix Anda dari file.")


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
        - Sekitar 7,000 gambar untuk data segregasi (tidak digunakan langsung di proyek ini, tapi bagian dari dataset asli).
    - **Ukuran Gambar:** Gambar-gambar memiliki resolusi 150x150 piksel. Untuk model MobileNetV3, gambar-gambar ini diubah ukurannya menjadi 224x224 piksel.
    - **Sumber:** Gambar-gambar dikumpulkan dari berbagai sumber dan mencakup variasi yang cukup baik dalam setiap kelas.

    Anda dapat menemukan dataset ini di [Kaggle: Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
    """)
    # Anda bisa menambahkan plot distribusi kelas dari EDA di sini jika diinginkan
    # Contoh: st.image("path/to/your/class_distribution_plot.png", caption="Distribusi Kelas dalam Dataset")
    st.info("Untuk implementasi nyata, muat gambar plot distribusi kelas Anda dari file.")