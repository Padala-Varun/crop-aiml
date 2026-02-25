# 🌾 AI Smart Agriculture Assistant

An AI-powered Smart Agriculture Web Application that helps farmers with crop recommendation, yield forecasting, rainfall prediction, plant disease detection, and an AI chatbot assistant.

## 🚀 Features

- **Crop Recommendation** — XGBoost classifier predicting the best crop based on soil & weather data
- **Yield Forecasting** — Random Forest regressor estimating expected crop yield
- **Rainfall Prediction** — Artificial Neural Network predicting rainfall amounts
- **Plant Disease Detection** — CNN (MobileNet) detecting plant diseases from leaf images
- **Gemini AI Chatbot** — Agriculture-focused AI assistant powered by Google Gemini
- **Voice Assistant** — Speech-to-text and text-to-speech for hands-free interaction

## 📦 Setup Instructions

### 1. Clone & Navigate
```bash
cd Preethi-Friend-Project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
copy .env.example .env
# Edit .env and add your Gemini API key
```

### 5. Train ML Models
```bash
python train_models.py
```
This generates synthetic datasets and trains all 4 models.

### 6. Run the Application
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

## 📊 Dataset Format

### crop.csv
| Column | Description |
|--------|-------------|
| N | Nitrogen content in soil |
| P | Phosphorus content in soil |
| K | Potassium content in soil |
| temperature | Temperature in °C |
| humidity | Relative humidity % |
| ph | Soil pH value |
| rainfall | Rainfall in mm |
| label | Crop name (target) |

### yield.csv
| Column | Description |
|--------|-------------|
| crop | Crop type |
| area | Area in hectares |
| rainfall | Rainfall in mm |
| soil_quality | Soil quality score (1-10) |
| yield | Crop yield in tons (target) |

### rainfall.csv
| Column | Description |
|--------|-------------|
| month | Month number (1-12) |
| humidity | Humidity % |
| temperature | Temperature °C |
| pressure | Atmospheric pressure hPa |
| wind_speed | Wind speed km/h |
| rainfall | Rainfall in mm (target) |

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict-crop` | POST | Crop recommendation |
| `/predict-yield` | POST | Yield prediction |
| `/predict-rainfall` | POST | Rainfall prediction |
| `/predict-disease` | POST | Plant disease detection |
| `/chat` | POST | AI chatbot |

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **ML**: scikit-learn, XGBoost, TensorFlow/Keras
- **AI**: Google Gemini API
- **Voice**: Web Speech API
