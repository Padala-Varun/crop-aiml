"""
Smart Agriculture AI Assistant - Flask Application
Main application file with all API endpoints.
"""

import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ============================================================
# PAGE ROUTES
# ============================================================

@app.route('/')
def home():
    """Render the home/landing page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Render the main dashboard."""
    return render_template('dashboard.html')

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    """
    Predict the best crop based on soil and weather parameters.
    
    Expected JSON:
    {
        "N": 90, "P": 42, "K": 43,
        "temperature": 20.8, "humidity": 82,
        "ph": 6.5, "rainfall": 202.9
    }
    """
    try:
        from utils.prediction import predict_crop as _predict_crop
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        required = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
        
        result = _predict_crop(
            n=float(data['N']),
            p=float(data['P']),
            k=float(data['K']),
            temperature=float(data['temperature']),
            humidity=float(data['humidity']),
            ph=float(data['ph']),
            rainfall=float(data['rainfall'])
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict-yield', methods=['POST'])
def predict_yield():
    """
    Predict crop yield.
    
    Expected JSON:
    {
        "crop": "rice",
        "area": 10.5,
        "rainfall": 200,
        "soil_quality": 7.5
    }
    """
    try:
        from utils.prediction import predict_yield as _predict_yield
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        required = ['crop', 'area', 'rainfall', 'soil_quality']
        for field in required:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
        
        result = _predict_yield(
            crop=str(data['crop']),
            area=float(data['area']),
            rainfall=float(data['rainfall']),
            soil_quality=float(data['soil_quality'])
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict-rainfall', methods=['POST'])
def predict_rainfall():
    """
    Predict rainfall amount.
    
    Expected JSON:
    {
        "month": 7,
        "humidity": 80,
        "temperature": 28,
        "pressure": 1008,
        "wind_speed": 12
    }
    """
    try:
        from utils.prediction import predict_rainfall as _predict_rainfall
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        required = ['month', 'humidity', 'temperature', 'pressure', 'wind_speed']
        for field in required:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
        
        result = _predict_rainfall(
            month=int(data['month']),
            humidity=float(data['humidity']),
            temperature=float(data['temperature']),
            pressure=float(data['pressure']),
            wind_speed=float(data['wind_speed'])
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    """
    Predict plant disease from uploaded leaf image.
    
    Expected: multipart/form-data with 'image' file field
    """
    try:
        from utils.prediction import predict_disease as _predict_disease
        
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'success': False, 'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, webp'}), 400
        
        result = _predict_disease(file)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """
    AI Chatbot endpoint.
    
    Expected JSON:
    {
        "message": "What fertilizer is best for rice?",
        "history": [{"role": "user", "content": "..."}, ...]
    }
    """
    try:
        from utils.chatbot import get_chat_response
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        message = data['message'].strip()
        if not message:
            return jsonify({'success': False, 'error': 'Empty message'}), 400
        
        history = data.get('history', [])
        response = get_chat_response(message, history)
        
        return jsonify({
            'success': True,
            'response': response
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get-crops', methods=['GET'])
def get_crops():
    """Get list of available crops for yield prediction dropdown."""
    try:
        from utils.prediction import get_available_crops
        crops = get_available_crops()
        return jsonify({'success': True, 'crops': crops})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size: 16MB'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("🌾 Smart Agriculture AI Assistant")
    print("=" * 40)
    print("🌐 Server: http://localhost:5000")
    print("📊 Dashboard: http://localhost:5000/dashboard")
    print("=" * 40)
    app.run(debug=True, host='0.0.0.0', port=5000)
