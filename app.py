import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, Dropout, MaxPooling1D, LSTM, Dense

# --- Load resources ---
model = load_model("model1_hybrid_wisata1.h5", compile=False)
embedding_matrix = np.load("embedding_matrix_wisata_norm_fix.npy")
key_norm = pd.read_csv("key_norm.csv")

# --- Preprocessing Functions ---

def casefolding(text):
    return text.lower().strip()

def datacleaning(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'rt[\s]', '', text)
    text = re.sub(r'[?|$|.|@#%^/&*=!_:")(-+,]-', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub('\n',' ',text)
    text = re.sub('\t',' ',text)
    text = re.sub('\s+',' ',text)
    text = text.strip(' ')
    text = re.sub('  +', ' ', text)
    text = re.sub('user',' ',text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)

    emoticon_pattern = re.compile("[" u"\U0001F600-\U0001F64F"
                                  u"\U0001F300-\U0001F5FF"
                                  u"\U0001F680-\U0001F6FF"
                                  u"\U0001F1E0-\U0001F1FF"
                                  u"\U00002700-\U000027BF"
                                  u"\U000024C2-\U0001F251"
                                  u"\U0001F932" u"\U0001F923"
                                  u"\U0001F92D" u"\U0001F914"
                                  u"\U0001F924" u"\U0001F92A"
                                  u"\U0001F929" u"\U0001F928"
                                  u"\U0001F941" u"\U0001F9D0"
                                  u"\U0001F9E7'" "]+", flags=re.UNICODE)
    return emoticon_pattern.sub(r'', text)

def WordNormalization(text):
    text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0]
                     if (key_norm['singkat'] == word).any() else word for word in text.split()])
    return text.lower()

# --- Tokenizer ---
max_words = 1000
max_sequence_length = 35

# Simulasi Tokenizer (atau load dari file jika disimpan)
# HARUS sama dengan saat training
def load_tokenizer():
    data = pd.read_excel("wisata_norm_fix.xlsx")  # gunakan data pelatihan asli
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data['Ulasan'])
    return tokenizer

tokenizer = load_tokenizer()

# --- Label Mapping ---
label_map = {0: "Netral", 1: "Puas", 2: "Tidak Puas"}

# --- Streamlit UI ---
st.title("Prediksi Kepuasan Wisatawan")
st.write("Masukkan ulasan wisatawan dan sistem akan memprediksi tingkat kepuasannya.")

user_input = st.text_area("Masukkan ulasan wisatawan:")

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Mohon masukkan ulasan terlebih dahulu.")
    else:
        # Preprocessing
        text = casefolding(user_input)
        text = datacleaning(text)
        text = WordNormalization(text)
        
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_sequence_length)
        
        prediction = model.predict(padded)
        label = np.argmax(prediction)

        st.success(f"Hasil Prediksi: **{label_map[label]}**")


# 🔹 NEW: Batch Prediction (CSV/XLSX)
st.markdown("---")
st.subheader("Prediksi Massal dari File (CSV/XLSX)")

uploaded_file = st.file_uploader(
    "Unggah file CSV atau XLSX berisi ulasan (satu ulasan per baris).",
    type=["csv", "xlsx"]
)

def preprocess_text_series(text_series: pd.Series) -> pd.Series:
    text_series = text_series.fillna("").astype(str)
    text_series = text_series.apply(casefolding)\
                             .apply(datacleaning)\
                             .apply(WordNormalization)
    return text_series

def predict_texts(list_of_texts):
    """Mengembalikan label prediksi & probabilitas untuk list teks."""
    seqs = tokenizer.texts_to_sequences(list_of_texts)
    padded = pad_sequences(seqs, maxlen=max_sequence_length)
    probs = model.predict(padded, verbose=0)  # shape: (n, num_classes)
    pred_idx = np.argmax(probs, axis=1)
    pred_label = [label_map[i] for i in pred_idx]
    return pred_label, probs

if uploaded_file is not None:
    # Baca file
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df_in = pd.read_csv(uploaded_file)
        else:
            df_in = pd.read_excel(uploaded_file)  # sheet pertama
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        df_in = None

    if df_in is not None:
        st.write("Contoh baris dari data yang diunggah:")
        st.dataframe(df_in.head(5))

        # Deteksi kolom teks; default ke 'Ulasan' jika ada
        text_columns = [
            c for c in df_in.columns
            if df_in[c].dtype == 'object' or str(df_in[c].dtype).startswith('string')
        ]
        default_index = text_columns.index("Ulasan") if "Ulasan" in text_columns else (0 if text_columns else None)

        if not text_columns:
            st.warning("Tidak ada kolom bertipe teks yang terdeteksi. Pastikan ada kolom berisi ulasan.")
        else:
            selected_col = st.selectbox(
                "Pilih kolom yang berisi ulasan:",
                options=text_columns,
                index=default_index if default_index is not None else 0
            )

            if st.button("Prediksi Massal"):
                # Preprocess
                texts_clean = preprocess_text_series(df_in[selected_col])
                # Prediksi
                pred_labels, probs = predict_texts(texts_clean.tolist())

                # Susun hasil
                df_out = df_in.copy()
                df_out["Prediksi"] = pred_labels

                # Tambahkan probabilitas per kelas mengikuti urutan indeks label_map
                class_names_sorted_by_idx = [label_map[i] for i in range(probs.shape[1])]
                for i, cname in enumerate(class_names_sorted_by_idx):
                    df_out[f"Prob_{cname}"] = probs[:, i]

                st.success(f"Prediksi selesai untuk {len(df_out)} baris.")
                st.write("Pratinjau hasil:")
                st.dataframe(df_out.head(10))

                # Unduh hasil CSV
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Unduh Hasil (CSV)",
                    data=csv_bytes,
                    file_name="hasil_prediksi_massal.csv",
                    mime="text/csv"
                )

