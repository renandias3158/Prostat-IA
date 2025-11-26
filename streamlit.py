import os
import zipfile
import subprocess
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ------------------------------------------------------------
# 1. Funções auxiliares
# ------------------------------------------------------------

def download_panda_dataset():
    cmd = [
        "kaggle", "competitions", "download",
        "-c", "prostate-cancer-grade-assessment",
        "-p", ".", "--quiet"
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode, result.stdout, result.stderr


# ACEITA tanto caminho quanto arquivo enviado pelo usuário
def load_image(file_or_path, target_size=(224, 224)):
    if isinstance(file_or_path, str):
        img = Image.open(file_or_path).convert("RGB")
    else:
        img = Image.open(file_or_path).convert("RGB")  # BytesIO

    img = img.resize(target_size)
    img = img_to_array(img)
    return preprocess_input(img)


# ------------------------------------------------------------
# 2. UI Streamlit
# ------------------------------------------------------------

st.title("📌 PANDA – Prostate Cancer Grade Assessment")
st.write("App completo: download, visualização, deep learning e predição.")

# ------------------------------------------------------------
# Kaggle credentials
# ------------------------------------------------------------

st.header("1. Configurar credenciais Kaggle")

kaggle_file = st.file_uploader("Envie seu kaggle.json", type="json")

if kaggle_file:
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")

    with open(kaggle_path, "wb") as f:
        f.write(kaggle_file.read())

    os.chmod(kaggle_path, 0o600)
    st.success("kaggle.json salvo!")

    if st.button("Baixar dataset completo"):
        st.info("Baixando — isso pode levar muito tempo...")
        ret, out, err = download_panda_dataset()

        if ret != 0:
            st.error(f"Erro ao baixar: {err}")
        else:
            st.info("Extraindo arquivo ZIP...")
            with zipfile.ZipFile("prostate-cancer-grade-assessment.zip", "r") as z:
                z.extractall()
            st.success("Dataset extraído com sucesso!")


# ------------------------------------------------------------
# Carregar labels
# ------------------------------------------------------------

if os.path.exists("train.csv"):
    st.header("2. Carregar train.csv")
    labels = pd.read_csv("train.csv")

    if {"image_id", "isup_grade"}.issubset(labels.columns):
        labels = labels[["image_id", "isup_grade"]]
        st.write(labels.head())
    else:
        st.error("train.csv não contém as colunas esperadas.")

else:
    st.stop()


# ------------------------------------------------------------
# Download de algumas imagens
# ------------------------------------------------------------

st.header("3. Baixar imagens individuais")

n = st.slider("Quantas imagens baixar?", 20, 5000, 200)

if st.button("Baixar imagens"):
    os.makedirs("train_images", exist_ok=True)
    downloaded = 0

    for _, row in labels.head(n).iterrows():
        img_id = row["image_id"]
        out_path = f"train_images/{img_id}.png"

        if not os.path.exists(out_path):
            cmd = [
                "kaggle", "competitions", "download",
                "-c", "prostate-cancer-grade-assessment",
                "-f", f"train_images/{img_id}.png",
                "-p", "train_images",
                "--quiet"
            ]
            subprocess.run(cmd)
            downloaded += 1

    st.success(f"{downloaded} imagens baixadas!")


# ------------------------------------------------------------
# TREINAMENTO DO MODELO
# ------------------------------------------------------------

st.header("4. Treinar modelo Deep Learning")

if st.button("Treinar modelo CNN"):
    st.info("Carregando imagens (pode demorar)…")

    imgs = []
    y = []

    for _, row in labels.head(n).iterrows():
        path = f"train_images/{row.image_id}.png"
        if os.path.exists(path):
            imgs.append(load_image(path))
            y.append(row.isup_grade)

    X = np.array(imgs)
    y = np.array(y)

    st.write(f"Dataset carregado: {X.shape}")

    st.info("Construindo modelo MobileNetV2...")

    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False  # transfer learning freeze

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="linear")
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    st.info("Treinando...")

    history = model.fit(
        X, y,
        validation_split=0.1,
        epochs=3,
        batch_size=8
    )

    model.save("modelo_panda.h5")
    st.success("Modelo treinado e salvo!")


# ------------------------------------------------------------
# Predição
# ------------------------------------------------------------

st.header("5. Testar modelo treinado")

if os.path.exists("modelo_panda.h5"):
    model = tf.keras.models.load_model("modelo_panda.h5")
    st.success("Modelo carregado!")

    uploaded = st.file_uploader("Envie uma imagem PNG para análise", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Imagem enviada", width=300)

        x = load_image(uploaded)
        x = np.expand_dims(x, axis=0)

        pred = model.predict(x)[0][0]
        st.subheader(f"🎯 Predição ISUP grade: **{pred:.2f}**")

else:
    st.info("Treine o modelo antes de fazer predições.")
