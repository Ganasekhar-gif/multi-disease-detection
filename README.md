# 🧠 Multi-Disease Detection using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" alt="status"/>
  <img src="https://img.shields.io/badge/Models-VGG16_&_NeuralNet-blue"/>
  <img src="https://img.shields.io/github/languages/top/Ganasekhar-gif/multi-disease-detection" alt="languages"/>
</p>

This project is a comprehensive AI-based medical assistant that can **predict multiple types of diseases** using two separate deep learning models:

- 🧬 **Image-based Disease Detection Model** powered by **VGG16**
- ❤️ **Heart Disease Detection Model** powered by a custom **Neural Network**

The project features two interactive web interfaces:
- A **Flask frontend** for uploading and predicting image-based diseases.
- A **FastAPI-powered backend** integrated with **Fitbit API** to fetch health metrics for heart disease prediction.

---

## 🏥 Diseases Covered

### 🔬 Image-Based Disease Detection (VGG16)

Predicts the following diseases from medical images:

- COVID
- NORMAL (Healthy)
- PNEUMONIA
- basal_cell_carcinoma
- melanoma
- glioma_tumor
- no_tumor
- pituitary_tumor

### ❤️ Heart Disease Detection (Neural Network)

Predicts heart disease risk based on clinical features fetched via **Fitbit API** and/or form input.

---

## 📂 Project Structure

```
multi-disease-detection/
├── static/
│   └── images/                    # Sample input images
├── templates/
│   └── index.html                 # Flask UI template
│── main.py                    # Flask app for prediction            
├── models/
│   └── MULTI_DISEASE_DETECTION_MODEL.keras # VGG16 image model (manual add)
|   |_ heart_disease_detection_model.keras
|   |_ heart_scalar.pkl
|──notebooks/
|   |── disease_detection.ipynb
|   |── heart_disease_detection.ipynb                
└── README.md                     # Project documentation
```

---

## 🚀 Features

- 🔍 Image classification with VGG16
- 🫀 Clinical prediction with neural network
- 🖼️ Flask-based image upload and diagnosis
- ⚡ FastAPI backend with Fitbit API integration
- 📊 Real-time disease risk prediction
- 🧩 Modular code structure

---

## 📦 Setup Instructions

> 📝 Make sure you have **Python ≥ 3.8**, and **pip** installed.

### 1. Clone the Repository

```bash
git clone https://github.com/Ganasekhar-gif/multi-disease-detection.git
cd multi-disease-detection
```

---

## 📥 Model Placement

Since model files are large, they are **not included** in the repository.

Manually place your trained models like this:

```
models/
└── MULTI_DISEASE_DETECTION_MODEL.keras       # Image classification model (~158MB)
└── heart_disease_detection_model.keras


```

## ⚡ Run Multi Disease Prediction App (FastAPI)

```bash
cd 
uvicorn main:app --reload
```

Then open the UI: [http://127.0.0.1:8000](http://127.0.0.1:8000)

Or access the interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔗 Fitbit API Integration

The FastAPI backend includes integration with the **Fitbit API** to automatically fetch health metrics (like heart rate, sleep, etc.), which are then used for heart disease risk prediction.

To enable this:

1. Register for a Fitbit developer account: [https://dev.fitbit.com](https://dev.fitbit.com)
2. Generate access tokens and paste them in `fitbit_integration.py`
3. Enable Fitbit syncing before using prediction endpoint

---

## 🧠 Model Details

### 🖼️ VGG16 - Multi-Disease Image Classifier

- Base Model: VGG16 (transfer learning)
- Layers: GlobalAveragePooling + Dense layers
- Loss: Categorical Crossentropy
- Optimizer: Adam
- Input: 224x224 medical images

### ❤️ Heart Disease Model

- Type: Feedforward Neural Network
- Input: Clinical + Fitbit features (e.g., resting bp, cholesterol)
- Hidden layers: 2 Dense + ReLU
- Output: Sigmoid (binary classification)

---

## ✨ Screenshots

- `static/images/` contains example input images for testing
- Add sample results for visual clarity

---

## 🛠️ Tools & Libraries

- **Flask** - Web UI for image model
- **FastAPI** - Heart disease backend
- **TensorFlow / Keras** - Model development
- **OpenCV / Pillow** - Image processing
- **Fitbit API** - Health data integration
- **Uvicorn** - FastAPI server

---

## 🤝 Contribution

Pull requests are welcome! If you have suggestions or find issues, feel free to:

- Fork the repo
- Create a feature branch
- Submit a PR

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 📬 Contact

- 👤 **Ganasekhar G**
- 🌐 [LinkedIn](https://www.linkedin.com/in/ganasekhar-gif/)
- 📧 ganasekharkalla@gmail.com

---

> ⭐ If you found this project useful, give it a star and share it!
