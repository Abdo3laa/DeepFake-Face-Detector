import os
import cv2
import numpy as np
import joblib
import torch
import clip
from PIL import Image
from flask import Flask, render_template, request, url_for, flash
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device)

adapter_model, le = joblib.load("model.pkl")

THRESHOLD = 0.45

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "supersecretkey123"


def classify_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return "Invalid Image", "gray"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return "No face detected", "gray"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    image_input = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model_clip.encode_image(image_input)
    embedding = embedding.cpu().numpy().reshape(1, -1)

    final_prob = adapter_model.predict_proba(embedding)[0][1]

    pred = int(final_prob >= THRESHOLD)
    result = le.inverse_transform([pred])[0]

    if result.lower() == "fake":
        text = "Prediction: AI Generated"
        color = "#FF0606"
    else:
        text = "Prediction: Real"
        color = "#06FA67"

    return text, color


@app.route("/", methods=["GET", "POST"])
def index():
    file_url = None
    result = None
    result_color = None
    delete_file = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            file_url = url_for("static", filename=f"uploads/{filename}")
            result, result_color = classify_image(file_path)
            delete_file = file_url
        else:
            flash("Please upload a valid image!", "error")

    logo_url = url_for("static", filename="logo.png")
    return render_template("index.html",
                           file_url=file_url,
                           result=result,
                           result_color=result_color,
                           logo_url=logo_url,
                           delete_file=delete_file)


@app.route("/static/uploads/<filename>", methods=["DELETE"])
def delete_uploaded(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
    return "", 200


if __name__ == "__main__":
    app.run(debug=True)
