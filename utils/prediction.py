"""
Prediction utilities for all ML models.
Handles model loading, preprocessing, and inference.
"""

import os
import pickle
import numpy as np
from PIL import Image
import io

# Global model cache
_models = {}


def _get_model(model_name):
    """Load and cache a model."""
    if model_name in _models:
        return _models[model_name]
    
    model_paths = {
        'crop': 'models/crop_model.pkl',
        'yield': 'models/yield_model.pkl',
        'rainfall_model': 'models/rainfall_model.keras',
        'rainfall_scaler': 'models/rainfall_scaler.pkl',
        'disease_model': 'models/plant_disease_model.keras',
        'disease_classes': 'models/disease_classes.pkl',
    }
    
    path = model_paths.get(model_name)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}. Run 'python train_models.py' first.")
    
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            _models[model_name] = pickle.load(f)
    elif path.endswith('.keras'):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        _models[model_name] = tf.keras.models.load_model(path)
    
    return _models[model_name]


def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    """
    Predict the best crop based on soil and weather parameters.
    
    Returns:
        dict: {crop, confidence, all_predictions}
    """
    try:
        data = _get_model('crop')
        model = data['model']
        le = data['label_encoder']
        
        features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
        
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        crop_name = le.inverse_transform([prediction])[0]
        confidence = float(max(probabilities)) * 100
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = [
            {'crop': le.inverse_transform([idx])[0], 'confidence': round(float(probabilities[idx]) * 100, 2)}
            for idx in top_indices
        ]
        
        return {
            'success': True,
            'crop': crop_name,
            'confidence': round(confidence, 2),
            'top_predictions': top_predictions
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def predict_yield(crop, area, rainfall, soil_quality):
    """
    Predict crop yield based on parameters.
    
    Returns:
        dict: {yield_value, unit}
    """
    try:
        data = _get_model('yield')
        model = data['model']
        le = data['label_encoder']
        crops = data['crops']
        
        if crop.lower() not in [c.lower() for c in crops]:
            return {
                'success': False, 
                'error': f'Unknown crop: {crop}. Available: {", ".join(crops)}'
            }
        
        crop_encoded = le.transform([crop.lower()])[0]
        features = np.array([[crop_encoded, area, rainfall, soil_quality]])
        
        yield_value = model.predict(features)[0]
        
        return {
            'success': True,
            'yield_value': round(float(yield_value), 2),
            'unit': 'tons',
            'crop': crop,
            'area': area,
            'available_crops': list(crops)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def predict_rainfall(month, humidity, temperature, pressure, wind_speed):
    """
    Predict rainfall amount using the ANN model.
    
    Returns:
        dict: {rainfall, unit}
    """
    try:
        model = _get_model('rainfall_model')
        scaler = _get_model('rainfall_scaler')
        
        features = np.array([[month, humidity, temperature, pressure, wind_speed]])
        features_scaled = scaler.transform(features)
        
        predicted_rainfall = model.predict(features_scaled, verbose=0)[0][0]
        predicted_rainfall = max(0, float(predicted_rainfall))
        
        # Classify intensity
        if predicted_rainfall < 20:
            intensity = 'Low'
        elif predicted_rainfall < 60:
            intensity = 'Moderate'
        elif predicted_rainfall < 120:
            intensity = 'High'
        else:
            intensity = 'Very High'
        
        return {
            'success': True,
            'rainfall': round(predicted_rainfall, 2),
            'unit': 'mm',
            'intensity': intensity
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def predict_disease(image_file):
    """
    Predict plant disease from a leaf image using CNN.
    
    Args:
        image_file: File-like object or path to image
        
    Returns:
        dict: {disease, confidence, treatment, is_healthy}
    """
    try:
        from utils.chatbot import get_treatment
        
        model = _get_model('disease_model')
        classes = _get_model('disease_classes')
        
        # Load and preprocess image
        if isinstance(image_file, str):
            img = Image.open(image_file)
        else:
            img = Image.open(io.BytesIO(image_file.read()))
        
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx]) * 100
        
        disease_name = classes[predicted_class_idx]
        is_healthy = 'healthy' in disease_name.lower()
        
        # Format disease name for display
        display_name = disease_name.replace('___', ' - ').replace('_', ' ')
        
        treatment = get_treatment(disease_name)
        
        # Top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        top_predictions = [
            {
                'disease': classes[idx].replace('___', ' - ').replace('_', ' '),
                'confidence': round(float(predictions[idx]) * 100, 2)
            }
            for idx in top_indices
        ]
        
        return {
            'success': True,
            'disease': display_name,
            'confidence': round(confidence, 2),
            'treatment': treatment,
            'is_healthy': is_healthy,
            'top_predictions': top_predictions
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_available_crops():
    """Get list of crops available for yield prediction."""
    try:
        data = _get_model('yield')
        return list(data['crops'])
    except:
        return ['rice', 'wheat', 'maize', 'cotton', 'sugarcane', 'banana', 'coffee', 'jute']
