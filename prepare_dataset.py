import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import clip
from PIL import Image

DATASET_PATH = r"C:\Users\Abdo\Desktop\SIC\Final Project\dataset"
CSV_OUTPUT = "face_embeddings_clib.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

data = []
labels = []

for label in ["real", "fake"]:
    folder = os.path.join(DATASET_PATH, label)
    for file in tqdm(os.listdir(folder), desc=f"Processing {label}"):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        image_input = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input)
        embedding = embedding.cpu().numpy().flatten()
        data.append(embedding)
        labels.append(label)

df = pd.DataFrame(data)
df["label"] = labels
df.to_csv(CSV_OUTPUT, index=False)
print("CSV saved:", CSV_OUTPUT)
