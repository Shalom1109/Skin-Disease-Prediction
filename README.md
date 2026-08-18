# SkinSense AI - Clinical Skin Disease Diagnosis System

SkinSense AI is a high-performance deep learning web application designed for the accurate prediction and classification of skin diseases. It combines a state-of-the-art visual deep learning triple ensemble model with a contextual Retrieval-Augmented Generation (RAG) pipeline for clinical summary synthesis.

## 🚀 Key Features

* **Triple Ensemble Classifier**: Combines **ConvNeXt Base**, **Vision Transformer (ViT) Base**, and **MaxViT Base** to fuse local, global, and hybrid attention features for diagnosis.
* **Over 96% Classification Accuracy**: Fine-tuned on the **HAM10000** dataset with advanced training strategies (AMP, AdamW, Label Smoothing, parameter freezing, and weighted random sampling).
* **Clinical RAG System**: Dynamically retrieves medical descriptions, recommendations, and important Do's & Don'ts for the predicted condition.
* **Modern Web Dashboard**: Features a responsive glassmorphic user interface built with HTML5, CSS3, and JavaScript, showing real-time diagnosis graphs and clinical summaries.
* **SQL Console**: Interactive dashboard tool for checking and managing user data/prediction logs.

---

## 🛠️ Repository Setup

To keep the repository clean and comply with GitHub's file size limit (100MB), the large model weights, SQLite database logs, and raw dataset images are excluded from Git. Follow the setup below to restore them.

### 1. Prerequisites
Make sure you have Python 3.10+ installed.

### 2. Install Dependencies
Navigate to the `backend` directory and install the required libraries:
```bash
cd backend
pip install -r requirements.txt
```

### 3. Model Weights & Dataset Setup
* **Model Weights**: Place your pre-trained `skin_sense_ensemble_best.pth` file into the `backend/` directory:
  `backend/skin_sense_ensemble_best.pth`
* **Dataset**: If you plan to retrain or run validation, place the HAM10000 images in the `dataset/` folder in the project root:
  `dataset/`

---

## 🏁 Running the Application

For a quick setup, you can launch the app directly:

### On Windows
1. Double-click the **`LAUNCH_SKINSENSE_AI.bat`** file in the root folder.
2. The unified FastAPI server will launch at `http://127.0.0.1:8000` and automatically open in your default browser.

### Manually (Terminal)
Navigate to the backend directory and run the Uvicorn server:
```bash
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Then open `http://127.0.0.1:8000` in your web browser.

---

## 🏗️ Architecture Summary

```
                       [ Frontend Web UI ]
                       (HTML5, CSS3, JS)
                               │ (API Requests / JWT Auth)
                               ▼
                      [ FastAPI Backend ]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 [ Triple Ensemble Model ]            [ RAG / Ollama System ]
 (ConvNeXt + ViT + MaxViT)             (Offline RAG Database)
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
                     [ SQLite Database ]
                      (Logs & Accounts)
```

---

## 📝 License
This project is licensed under the MIT License.
