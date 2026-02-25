"""
Chatbot utility using Google Gemini API for agricultural assistance.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Disease treatment database
DISEASE_TREATMENTS = {
    'Apple___Apple_scab': 'Apply fungicides like Captan or Mancozeb. Remove and destroy fallen leaves. Prune trees for better air circulation.',
    'Apple___Black_rot': 'Remove mummified fruits and cankers. Apply fungicides during bloom. Maintain proper tree hygiene.',
    'Apple___Cedar_apple_rust': 'Apply fungicides like Myclobutanil. Remove nearby juniper trees if possible. Use resistant apple varieties.',
    'Apple___healthy': 'Your apple plant looks healthy! Continue with regular watering, fertilization, and pest monitoring.',
    'Corn_(maize)___Cercospora_leaf_spot': 'Apply foliar fungicides like Pyraclostrobin. Rotate crops. Use resistant hybrids.',
    'Corn_(maize)___Common_rust': 'Apply fungicides if severe. Plant resistant hybrids. Early planting can help avoid peak rust periods.',
    'Corn_(maize)___healthy': 'Your corn plant looks healthy! Maintain proper irrigation and nutrient management.',
    'Grape___Black_rot': 'Apply Mancozeb or Myclobutanil fungicides. Remove mummified berries. Improve canopy management.',
    'Grape___Esca_(Black_Measles)': 'No cure available. Remove infected vines. Apply wound protectants after pruning.',
    'Grape___healthy': 'Your grapevine looks healthy! Continue with regular pruning and disease monitoring.',
    'Potato___Early_blight': 'Apply Chlorothalonil or Mancozeb fungicides. Rotate crops every 2-3 years. Remove infected plant debris.',
    'Potato___Late_blight': 'Apply metalaxyl-based fungicides immediately. Destroy infected plants. Ensure proper drainage.',
    'Potato___healthy': 'Your potato plant looks healthy! Maintain proper hilling and moisture management.',
    'Tomato___Bacterial_spot': 'Apply copper-based bactericides. Use disease-free seeds. Avoid overhead irrigation.',
    'Tomato___Early_blight': 'Apply Chlorothalonil fungicide. Mulch around plants. Remove infected lower leaves.',
    'Tomato___Late_blight': 'Apply metalaxyl fungicides. Remove infected plants immediately. Avoid wetting foliage.',
    'Tomato___Leaf_Mold': 'Improve ventilation. Apply fungicides like Chlorothalonil. Reduce humidity in greenhouse.',
    'Tomato___Septoria_leaf_spot': 'Apply Mancozeb fungicides. Remove infected leaves. Avoid splashing water on leaves.',
    'Tomato___Spider_mites': 'Apply miticides or neem oil. Increase humidity. Introduce predatory mites.',
    'Tomato___Target_Spot': 'Apply Chlorothalonil fungicide. Improve air circulation. Remove infected leaves.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whitefly vectors with insecticides. Use resistant varieties. Remove infected plants.',
    'Tomato___Tomato_mosaic_virus': 'No chemical cure. Remove infected plants. Disinfect tools. Use resistant varieties.',
    'Tomato___healthy': 'Your tomato plant looks healthy! Continue with regular care and stake support.',
    'Pepper,_bell___Bacterial_spot': 'Apply copper sprays. Use certified disease-free seeds. Rotate crops.',
    'Pepper,_bell___healthy': 'Your pepper plant looks healthy! Maintain consistent watering and fertilization.',
    'Strawberry___Leaf_scorch': 'Remove infected leaves. Apply fungicides. Ensure proper spacing for air flow.',
    'Strawberry___healthy': 'Your strawberry plant looks healthy! Continue with mulching and proper watering.',
    'Rice___Brown_spot': 'Apply Mancozeb fungicide. Use balanced fertilization. Treat seeds before planting.',
    'Rice___Leaf_blast': 'Apply Tricyclazole fungicide. Use resistant varieties. Avoid excess nitrogen fertilizer.',
    'Rice___healthy': 'Your rice plant looks healthy! Maintain proper water management and nutrient supply.',
}

def get_treatment(disease_name):
    """Get treatment recommendation for a detected disease."""
    return DISEASE_TREATMENTS.get(disease_name, 
        'Please consult a local agricultural expert for specific treatment recommendations for this condition.')


def get_chat_response(message, history=None):
    """
    Get a response from the Gemini AI chatbot.
    
    Args:
        message: User's message
        history: List of previous messages [{'role': 'user'/'assistant', 'content': '...'}]
    
    Returns:
        str: AI response text
    """
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key or api_key == 'your_gemini_api_key_here':
        return get_fallback_response(message)
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        system_prompt = """You are an expert agricultural assistant helping farmers with:
- Crop advice and recommendations
- Disease prevention and treatment  
- Soil health and fertilizer suggestions
- Irrigation and water management
- Pest control guidance
- Seasonal farming advice
- Sustainable farming practices
- Weather impact on farming
- Market trends for crops

Provide practical, actionable advice. Be friendly and supportive. 
If you're unsure, recommend consulting a local agricultural extension officer.
Keep responses concise but informative (2-3 paragraphs max)."""
        
        # Build conversation context
        prompt_parts = [system_prompt + "\n\n"]
        
        if history:
            for msg in history[-6:]:  # Last 6 messages for context
                role = "Farmer" if msg.get('role') == 'user' else "Assistant"
                prompt_parts.append(f"{role}: {msg.get('content', '')}\n")
        
        prompt_parts.append(f"Farmer: {message}\nAssistant:")
        
        full_prompt = "".join(prompt_parts)
        response = model.generate_content(full_prompt)
        
        return response.text
        
    except Exception as e:
        print(f"[ERROR] Gemini API error: {e}")
        return get_fallback_response(message)


def get_fallback_response(message):
    """Provide fallback responses when Gemini API is not available."""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        return ("🌾 Hello! I'm your Smart Agriculture Assistant. I can help you with crop recommendations, "
                "disease detection, yield predictions, and farming advice. How can I help you today?")
    
    if any(word in message_lower for word in ['crop', 'plant', 'grow', 'sow']):
        return ("🌱 For crop selection, consider your soil type, climate, and water availability. "
                "Use our **Crop Recommendation** tool on the dashboard — it analyzes your soil nutrients (N, P, K), "
                "temperature, humidity, pH, and rainfall to suggest the best crop. "
                "Would you like guidance on any specific crop?")
    
    if any(word in message_lower for word in ['disease', 'sick', 'infection', 'spot', 'blight', 'rot']):
        return ("🔬 For plant disease detection, use our **Disease Detection** tool — simply upload a photo of the "
                "affected leaf and our AI will identify the disease and suggest treatments. "
                "Common preventive measures include crop rotation, proper spacing, and timely fungicide application.")
    
    if any(word in message_lower for word in ['fertilizer', 'nutrient', 'npk', 'nitrogen', 'phosphorus']):
        return ("🧪 Fertilizer selection depends on your soil test results and crop needs:\n"
                "• **Nitrogen (N)**: Promotes leaf growth — use Urea or Ammonium Sulfate\n"
                "• **Phosphorus (P)**: Supports root development — use DAP or SSP\n"
                "• **Potassium (K)**: Improves disease resistance — use MOP\n"
                "Get your soil tested to determine exact requirements.")
    
    if any(word in message_lower for word in ['water', 'irrigation', 'rain', 'drought']):
        return ("💧 Efficient irrigation is crucial for farming. Consider these methods:\n"
                "• **Drip irrigation**: Best for water conservation (90% efficiency)\n"
                "• **Sprinkler**: Good for large fields\n"
                "• **Flood/Surface**: Traditional but less efficient\n"
                "Use our **Rainfall Prediction** tool to plan your irrigation schedule.")
    
    if any(word in message_lower for word in ['pest', 'insect', 'bug', 'worm']):
        return ("🐛 Integrated Pest Management (IPM) strategies:\n"
                "1. **Cultural**: Crop rotation, resistant varieties, proper spacing\n"
                "2. **Biological**: Introduce natural predators (ladybugs, lacewings)\n"
                "3. **Mechanical**: Traps, barriers, handpicking\n"
                "4. **Chemical**: Use pesticides as last resort, follow safety guidelines\n"
                "Identify the specific pest for targeted treatment.")
    
    if any(word in message_lower for word in ['yield', 'harvest', 'production']):
        return ("📊 To maximize crop yield:\n"
                "• Ensure proper soil preparation and nutrient management\n"
                "• Use quality seeds and appropriate varieties\n"
                "• Follow recommended spacing and planting depth\n"
                "• Monitor for pests and diseases regularly\n"
                "Use our **Yield Prediction** tool to estimate your expected harvest.")
    
    if any(word in message_lower for word in ['soil', 'ph', 'organic']):
        return ("🌍 Soil health is the foundation of good farming:\n"
                "• **pH 6.0-7.0**: Ideal for most crops\n"
                "• Add organic matter (compost, manure) to improve structure\n"
                "• Practice crop rotation to prevent nutrient depletion\n"
                "• Get regular soil tests every 2-3 years\n"
                "• Avoid over-tilling to preserve soil microorganisms")
    
    return ("🌾 I'm your Smart Agriculture Assistant! I can help with:\n"
            "• 🌱 **Crop Recommendation** — Best crop for your conditions\n"
            "• 📊 **Yield Prediction** — Expected harvest estimation\n"
            "• 🌧️ **Rainfall Forecast** — Weather-based planning\n"
            "• 🔬 **Disease Detection** — Upload leaf images for diagnosis\n\n"
            "Try asking about crops, fertilizers, pests, irrigation, or soil health!\n\n"
            "💡 *Tip: Set up your Gemini API key in .env for AI-powered responses.*")
