"""
Train and save all ML models for the Smart Agriculture AI Assistant.
Generates synthetic datasets if not present, then trains:
1. Crop Recommendation - XGBoost Classifier
2. Yield Forecasting - Random Forest Regressor
3. Rainfall Prediction - ANN (Keras Sequential)
4. Plant Disease Detection - CNN (MobileNet-based placeholder)
"""

import os
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. GENERATE SYNTHETIC DATASETS
# ============================================================

def generate_crop_dataset(path='datasets/crop.csv'):
    """Generate synthetic crop recommendation dataset."""
    if os.path.exists(path):
        print(f"[INFO] {path} already exists, skipping generation.")
        return
    
    np.random.seed(42)
    crops = {
        'rice':       {'N': (60, 100), 'P': (35, 65), 'K': (35, 55), 'temp': (20, 28), 'humidity': (75, 95), 'ph': (5.0, 7.0), 'rainfall': (180, 260)},
        'wheat':      {'N': (80, 130), 'P': (40, 70), 'K': (15, 35), 'temp': (15, 25), 'humidity': (50, 75), 'ph': (6.0, 7.5), 'rainfall': (50, 120)},
        'maize':      {'N': (60, 100), 'P': (35, 60), 'K': (25, 50), 'temp': (18, 30), 'humidity': (55, 80), 'ph': (5.5, 7.0), 'rainfall': (60, 110)},
        'cotton':     {'N': (100, 150), 'P': (40, 70), 'K': (18, 35), 'temp': (22, 32), 'humidity': (60, 85), 'ph': (6.0, 8.0), 'rainfall': (60, 110)},
        'jute':       {'N': (60, 100), 'P': (35, 60), 'K': (35, 55), 'temp': (23, 32), 'humidity': (70, 95), 'ph': (6.0, 7.5), 'rainfall': (150, 250)},
        'coffee':     {'N': (90, 130), 'P': (15, 35), 'K': (25, 45), 'temp': (22, 30), 'humidity': (50, 75), 'ph': (6.0, 7.0), 'rainfall': (100, 200)},
        'sugarcane':  {'N': (70, 120), 'P': (40, 65), 'K': (18, 40), 'temp': (25, 35), 'humidity': (70, 90), 'ph': (5.5, 7.5), 'rainfall': (80, 150)},
        'banana':     {'N': (80, 120), 'P': (70, 100), 'K': (45, 60), 'temp': (25, 32), 'humidity': (75, 90), 'ph': (5.5, 7.0), 'rainfall': (90, 150)},
        'mango':      {'N': (15, 35), 'P': (15, 35), 'K': (25, 50), 'temp': (27, 37), 'humidity': (45, 70), 'ph': (5.5, 7.5), 'rainfall': (40, 100)},
        'grapes':     {'N': (15, 35), 'P': (120, 150), 'K': (190, 210), 'temp': (20, 35), 'humidity': (75, 85), 'ph': (5.5, 6.5), 'rainfall': (60, 80)},
        'apple':      {'N': (15, 35), 'P': (120, 145), 'K': (195, 210), 'temp': (20, 25), 'humidity': (90, 95), 'ph': (5.5, 6.5), 'rainfall': (100, 130)},
        'coconut':    {'N': (15, 30), 'P': (8, 18), 'K': (28, 38), 'temp': (25, 30), 'humidity': (90, 98), 'ph': (5.5, 6.5), 'rainfall': (130, 200)},
    }
    
    rows = []
    samples_per_crop = 180
    for crop, ranges in crops.items():
        for _ in range(samples_per_crop):
            row = {
                'N': np.random.uniform(*ranges['N']),
                'P': np.random.uniform(*ranges['P']),
                'K': np.random.uniform(*ranges['K']),
                'temperature': np.random.uniform(*ranges['temp']),
                'humidity': np.random.uniform(*ranges['humidity']),
                'ph': round(np.random.uniform(*ranges['ph']), 2),
                'rainfall': np.random.uniform(*ranges['rainfall']),
                'label': crop
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] Generated {path} with {len(df)} samples")


def generate_yield_dataset(path='datasets/yield.csv'):
    """Generate synthetic yield prediction dataset."""
    if os.path.exists(path):
        print(f"[INFO] {path} already exists, skipping generation.")
        return
    
    np.random.seed(43)
    crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'banana', 'coffee', 'jute']
    rows = []
    
    for _ in range(1200):
        crop = np.random.choice(crops)
        area = np.random.uniform(0.5, 50)
        rainfall = np.random.uniform(30, 300)
        soil_quality = np.random.uniform(1, 10)
        
        base_yield = {
            'rice': 3.5, 'wheat': 3.0, 'maize': 4.5, 'cotton': 1.5,
            'sugarcane': 70.0, 'banana': 30.0, 'coffee': 1.0, 'jute': 2.0
        }
        
        y = base_yield[crop] * area * (0.5 + soil_quality / 20) * (0.7 + rainfall / 500)
        y += np.random.normal(0, y * 0.1)
        y = max(0.1, y)
        
        rows.append({
            'crop': crop, 'area': round(area, 2), 'rainfall': round(rainfall, 2),
            'soil_quality': round(soil_quality, 2), 'yield': round(y, 2)
        })
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] Generated {path} with {len(df)} samples")


def generate_rainfall_dataset(path='datasets/rainfall.csv'):
    """Generate synthetic rainfall prediction dataset."""
    if os.path.exists(path):
        print(f"[INFO] {path} already exists, skipping generation.")
        return
    
    np.random.seed(44)
    rows = []
    
    for _ in range(600):
        month = np.random.randint(1, 13)
        humidity = np.random.uniform(30, 95)
        temperature = np.random.uniform(10, 42)
        pressure = np.random.uniform(990, 1030)
        wind_speed = np.random.uniform(2, 35)
        
        rainfall = (humidity * 1.5 + (30 - abs(temperature - 25)) * 2 
                    + (1015 - pressure) * 0.8 + wind_speed * 0.5
                    + np.random.normal(0, 15))
        rainfall = max(0, round(rainfall, 2))
        
        rows.append({
            'month': month, 'humidity': round(humidity, 2),
            'temperature': round(temperature, 2), 'pressure': round(pressure, 2),
            'wind_speed': round(wind_speed, 2), 'rainfall': rainfall
        })
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK] Generated {path} with {len(df)} samples")


# ============================================================
# 2. TRAIN MODELS
# ============================================================

def train_crop_model():
    """Train XGBoost Classifier for crop recommendation."""
    print("\n" + "="*50)
    print("Training Crop Recommendation Model (XGBoost)")
    print("="*50)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score
    
    df = pd.read_csv('datasets/crop.csv')
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[OK] Accuracy: {acc:.4f}")
    
    os.makedirs('models', exist_ok=True)
    with open('models/crop_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'label_encoder': le, 'features': list(X.columns)}, f)
    print("[OK] Saved models/crop_model.pkl")


def train_yield_model():
    """Train Random Forest Regressor for yield prediction."""
    print("\n" + "="*50)
    print("Training Yield Prediction Model (Random Forest)")
    print("="*50)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    
    df = pd.read_csv('datasets/yield.csv')
    le = LabelEncoder()
    df['crop_encoded'] = le.fit_transform(df['crop'])
    
    X = df[['crop_encoded', 'area', 'rainfall', 'soil_quality']]
    y = df['yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"[OK] MAE: {mae:.4f}, R²: {r2:.4f}")
    
    os.makedirs('models', exist_ok=True)
    with open('models/yield_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'label_encoder': le, 'crops': list(le.classes_)}, f)
    print("[OK] Saved models/yield_model.pkl")


def train_rainfall_model():
    """Train ANN for rainfall prediction."""
    print("\n" + "="*50)
    print("Training Rainfall Prediction Model (ANN)")
    print("="*50)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    
    df = pd.read_csv('datasets/rainfall.csv')
    
    X = df[['month', 'humidity', 'temperature', 'pressure', 'wind_speed']]
    y = df['rainfall']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(5,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, 
              validation_split=0.2, verbose=0)
    
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"[OK] Test MAE: {mae:.4f}")
    
    os.makedirs('models', exist_ok=True)
    model.save('models/rainfall_model.keras')
    
    with open('models/rainfall_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("[OK] Saved models/rainfall_model.keras and rainfall_scaler.pkl")


def train_disease_model():
    """Create and save a CNN model for plant disease detection."""
    print("\n" + "="*50)
    print("Creating Plant Disease Detection Model (MobileNet CNN)")
    print("="*50)
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import MobileNetV2
    
    # Disease classes (PlantVillage dataset standard classes - subset)
    disease_classes = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Corn_(maize)___Cercospora_leaf_spot', 'Corn_(maize)___Common_rust', 'Corn_(maize)___healthy',
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
        'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Rice___Brown_spot', 'Rice___Leaf_blast', 'Rice___healthy'
    ]
    
    num_classes = len(disease_classes)
    
    # Build MobileNetV2 transfer learning model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    os.makedirs('models', exist_ok=True)
    model.save('models/plant_disease_model.keras')
    
    # Save class names
    with open('models/disease_classes.pkl', 'wb') as f:
        pickle.dump(disease_classes, f)
    
    print(f"[OK] Saved models/plant_disease_model.keras ({num_classes} classes)")
    print(f"[OK] Saved models/disease_classes.pkl")
    print("[NOTE] This is an untrained architecture. For production, train on the PlantVillage dataset.")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("🌾 Smart Agriculture AI - Model Training Pipeline")
    print("=" * 60)
    
    # Generate datasets
    print("\n📊 Step 1: Generating Datasets...")
    generate_crop_dataset()
    generate_yield_dataset()
    generate_rainfall_dataset()
    
    # Train models
    print("\n🧠 Step 2: Training Models...")
    train_crop_model()
    train_yield_model()
    train_rainfall_model()
    train_disease_model()
    
    print("\n" + "=" * 60)
    print("✅ All models trained and saved successfully!")
    print("📁 Models saved in: models/")
    print("📁 Datasets saved in: datasets/")
    print("\nYou can now run: python app.py")
