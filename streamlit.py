# app.py

import os
import zipfile
import subprocess
import pandas as pd
import numpy as np

import streamlit as st

# Função para baixar dataset da competição
def download_panda_dataset():
    # roda comando kaggle para baixar os dados da competição
    cmd = [
        "kaggle", "competitions", "download",
        "-c", "prostate-cancer-grade-assessment",
        "-p", ".",   # pasta atual
        "--quiet"
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode, result.stdout, result.stderr

# --- UI Streamlit ---
st.title("PANDA / Prostate Cancer Grade Assessment")

st.header("1. Configurar credenciais Kaggle")

kaggle_file = st.file_uploader("Envie seu kaggle.json", type="json")
if kaggle_file:
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    with open(kaggle_path, "wb") as f:
        f.write(kaggle_file.read())
    os.chmod(kaggle_path, 0o600)
    st.success("kaggle.json salvo e configurado!")

    if st.button("Baixar e extrair dataset completa"):
        st.write("Baixando dataset da competição … isso pode demorar …")
        ret, out, err = download_panda_dataset()
        if ret != 0:
            st.error(f"Erro ao baixar: {err}")
        else:
            st.write("Download completo. Extraindo …")
            with zipfile.ZipFile("prostate-cancer-grade-assessment.zip", "r") as z:
                z.extractall()
            st.success("Dataset baixado e extraído com sucesso!")

# Se o CSV existir — avançamos para carregar labels
if os.path.exists("train.csv"):
    st.header("2. Carregar labels (train.csv)")
    labels = pd.read_csv("train.csv")
    if "image_id" in labels.columns and "isup_grade" in labels.columns:
        labels = labels[["image_id", "isup_grade"]]
        st.write("Amostra dos dados de label:")
        st.write(labels.head())
    else:
        st.error("train.csv não contém as colunas esperadas ('image_id', 'isup_grade')")

    st.header("3. Baixar / carregar algumas imagens (opcional)")
    n = st.number_input("Quantas imagens baixar / carregar", min_value=10, max_value=5000, value=200)
    if st.button("Baixar e carregar imagens"):
        # cria pasta para imagens
        os.makedirs("train_images", exist_ok=True)
        count = 0
        for i, row in labels.head(n).iterrows():
            img_id = row["image_id"]
            img_file = f"train_images/{img_id}.png"
            if not os.path.exists(img_file):
                # download da imagem via kaggle API
                cmd = [
                    "kaggle", "competitions", "download",
                    "-c", "prostate-cancer-grade-assessment",
                    "-f", f"train_images/{img_id}.png",
                    "-p", "train_images",
                    "--quiet"
                ]
                subprocess.run(cmd)
                count += 1
        st.success(f"{count} novas imagens baixadas (se já existiam, não serão baixadas).")

        st.write("Lista de imagens baixadas:")
        files = os.listdir("train_images")
        st.write(files[:10])

    # Se quiser, adicionar mais etapas, por exemplo carregar imagens, pré-processar, treinar modelo etc.

else:
    st.info("Quando você baixar e extrair o dataset, train.csv aparecerá aqui.")
