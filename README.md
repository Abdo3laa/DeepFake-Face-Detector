# 🤖 Deepfake Detection System: CLIP Embeddings + XGBoost Classifier

## 💡 Overview
This project implements a complete machine-learning pipeline to detect **AI-generated faces** (deepfakes) versus **real human faces**. Our solution leverages the **CLIP (Contrastive Language–Image Pre-training) image embeddings** for superior semantic feature extraction, followed by an **XGBoost classifier** for high-performance prediction. A key innovation is the **fine-tuning** mechanism, which allows the system to rapidly adapt and generalize to new image generators (like Gemini, GPT, and Midjourney) without requiring a full model retraining.

### Why Deepfake Detection?

The increasing realism of AI-generated content poses significant risks:
* **Identity Theft** and **Impersonation**.
* Creation of **Fake Evidence** and **Misinformation**.
* **Cybercrime** and **Reputational Damage**.

Traditional visual-feature-based models struggle to maintain accuracy when new generators with different artifact patterns emerge. Our **semantic understanding approach** offers enhanced **robustness** and **generalization**.

---

## 🛠️ System Architecture & Repository Structure

### System Architecture Stages

| Stage | Description | Key Artifact |
| :--- | :--- | :--- |
| **Dataset Preparation** | Convert images into CLIP embeddings and save to a single CSV file. | `dataset_embeddings.csv` |
| **Model Training** | Train the XGBoost classifier on the 511-dimensional CLIP features. | `model_initial.pkl` |
| **Fine-Tuning** | Adapt the existing model to new fake image styles using a lightweight, fast training pass. | `model.pkl` (Final Model) |
| **Deployment** | Flask web application for real-time image upload and classification. | Web Interface |

### Repository Structure
├── prepare_dataset.py # Generate CLIP embeddings CSV
├── train_model.py # Train XGBoost on embeddings
├── fine_tuning.py # Fine-tune model on new data
├── requirements.txt # Dependencies
└── deployment
├── app.py # Flask web application
├── model.pkl # Final fine-tuned model
├── static
│ ├── uploads # Temporary image storage
│ └── css/style.css # Web styling
└── templates
└── index.html # Web interface

---

## 📊 Dataset & Model Overview

### Dataset Summary

The dataset comprises approximately **~450,000 images** sourced from various public deepfake challenges and manually generated content.

| Source | Real Images | Fake Images |
| :--- | :--- | :--- |
| **Kaggle** (FFHQ / GRAVEX-200K / Celeb-DF / DFDC, etc.) | ✅ Yes | ✅ Yes |
| **Manual Generation** (Gemini / GPT / Midjourney / Stable Diffusion) | ❌ No | ✅ Yes |

### Model Components

#### 1. CLIP (Feature Extraction)
* **Function:** Converts each face image into a **511-dimensional feature embedding**.
* **Rationale:** Pre-trained by OpenAI, CLIP's embeddings capture subtle, semantic artifacts often missed by raw pixel-based models, such as **texture consistency**, **lighting artifacts**, **facial symmetry errors**, and **background blending issues**.

#### 2. XGBoost (Classifier)
* **Function:** A highly efficient and accurate Gradient Boosting classifier.
* **Training:** Trained on the CLIP embeddings to output a probability score, classifying the face as **Real Face** or **AI-Generated Face**.

#### 3. Fine-Tuning
* **Method:** Allows fast, lightweight adaptation to new datasets/generators, significantly improving generalization.
* **Code Snippet:** `adapter_model.fit(..., xgb_model=old_model.get_booster())`
* **Benefit:** Enables rapid deployment against emerging AI models.

---

## 🔥 Model Performance

The fine-tuning step proved crucial for maintaining high accuracy against the latest, highly realistic generators.

| Generator | Accuracy Before Fine-Tuning | Accuracy After Fine-Tuning |
| :--- | :--- | :--- |
| StyleGAN | 94% | **94%** |
| Stable Diffusion | 92% | **94%** |
| AiGenImage | 91% | **94%** |
| **GPT** | ~70% | **~93%** |
| **Gemini** | ~65% | **~95%** |

**Final Model Accuracy:** **~94.5%** across 5+ diverse AI generators.

---

## 🚀 Deployment (Flask Web App)

The system is deployed via a simple, real-time web application.

### Features
* **Secure Upload:** Uploaded images are automatically deleted after classification.
* **Pre-Processing:** Mandatory **face detection** is performed before the classification step.
* **Prediction:** Outputs a clear, probability-based result (e.g., "Prediction: AI-Generated").
* **Threshold:** Uses a customizable probability threshold (default: 0.45).

### Running Locally

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Flask App:**
    ```bash
    cd deployment
    python app.py
    ```

3.  **Access the Application:**
    Open your browser and navigate to:
    ```
    [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
    ```

---

## 🤝 Team & Conclusion

**Team Members (SIC 702 - Group 7)**
* Abdelrahman Alaa
* Omar Yasser
* Ahmed Hany

### Conclusion

We have successfully developed a **generalized deepfake detection system** that provides a practical, high-accuracy solution:
* Achieves **~94.5% real-world accuracy** across multiple, diverse AI generators.
* Employs an **efficient fine-tuning mechanism** to adapt to new deepfake styles.
* Deploys a user-friendly **Flask web interface** for real-time usage.

This project is a crucial contribution to **combating misinformation**, **protecting digital identities**, and supporting **cybersecurity**.

---

## ⏭️ Future Improvements Roadmap

| Short-Term | Medium-Term | Long-Term |
| :--- | :--- | :--- |
| **Multi-face detection** in a single image | Explainable model predictions (XAI) | Real-time **webcam detection** |
| Browser extension for instant checking | Batch verification tool for media analysts | Full **Video deepfake detection** |
| Dataset auto-update pipeline | Live server API for third-party integration | **Mobile application** development |
