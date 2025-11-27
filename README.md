# 🤖 Deepfake Detection System: CLIP Embeddings + XGBoost Classifier

## 💡 Overview
This project implements a complete machine-learning pipeline to detect **AI-generated faces** (deepfakes) versus **real human faces**. The system leverages **CLIP (Contrastive Language–Image Pre-training) image embeddings** for advanced semantic feature extraction, followed by an **XGBoost classifier** for robust prediction. A key innovation is the **fine-tuning mechanism**, which enables rapid adaptation to new image generators (Gemini, GPT, Midjourney, etc.) without full model retraining.

### Motivation
The increasing realism of AI-generated content poses serious risks:  
- **Identity theft** and impersonation.  
- Creation of **fake evidence** and misinformation.  
- **Cybercrime** and reputational damage.  

Traditional pixel-based models struggle with generalization to new generators, while our semantic-based approach ensures enhanced **robustness** and **accuracy**.

---

## 🛠️ System Architecture

### Pipeline Stages

| Stage | Description | Key Artifact |
| :--- | :--- | :--- |
| **Dataset Preparation** | Convert images to CLIP embeddings and store as CSV. | `dataset_embeddings.csv` |
| **Model Training** | Train XGBoost classifier on 511-dimensional CLIP features. | `model_initial.pkl` |
| **Fine-Tuning** | Adapt model to new fake image styles via lightweight retraining. | `model.pkl` (Final Model) |
| **Deployment** | Flask web application for real-time classification. | Web Interface |

---

## 📊 Dataset & Model Overview

### Dataset
Approximately **450,000 images** sourced from public deepfake datasets and manually generated content.

| Source | Real Images | Fake Images |
| :--- | :--- | :--- |
| **Kaggle** (FFHQ, GRAVEX-200K, Celeb-DF, DFDC, etc.) | ✅ Yes | ✅ Yes |
| **Manual Generation** (Gemini, GPT, Midjourney, Stable Diffusion) | ❌ No | ✅ Yes |

### Model Components

#### 1. CLIP (Feature Extraction)
- Converts each face image into a **511-dimensional embedding**.
- Captures subtle artifacts missed by pixel-based models, including **texture consistency**, **lighting artifacts**, **facial symmetry errors**, and **background blending issues**.

#### 2. XGBoost (Classifier)
- Gradient Boosting classifier trained on CLIP embeddings.
- Outputs probability scores to classify faces as **Real** or **AI-Generated**.

#### 3. Fine-Tuning
- Enables **rapid adaptation** to new generators without full retraining.
- Improves generalization and maintains high accuracy against emerging AI models.
- Example:  
```python
adapter_model.fit(..., xgb_model=old_model.get_booster())
```
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
## Try the Live Demo!

You can test the final, fine-tuned model instantly using the hosted Gradio app:

[**Deepfake Detector Gradio App**](https://blaxinosss-ai-face-detector.hf.space/?__theme=system&deep_link=WPhhnEHdJp8)

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
