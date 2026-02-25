/**
 * Smart Agriculture AI Assistant - Frontend Application
 * Handles page navigation, form submissions, chatbot, voice, and result display.
 */

// ============================================================
// STATE
// ============================================================
const state = {
    currentPage: 'overview',
    chatHistory: [],
    isRecording: false,
    recognition: null,
};

// ============================================================
// INITIALIZATION
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initForms();
    initChatbot();
    initVoice();
    initImageUpload();
    initMobileMenu();
    loadCropsList();
});

// ============================================================
// NAVIGATION
// ============================================================
function initNavigation() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            navigateTo(page);
        });
    });

    // Quick action cards
    document.querySelectorAll('.quick-action-card').forEach(card => {
        card.addEventListener('click', () => {
            const page = card.dataset.page;
            if (page) navigateTo(page);
        });
    });

    // Overview stat cards
    document.querySelectorAll('.stat-card').forEach(card => {
        card.addEventListener('click', () => {
            const page = card.dataset.page;
            if (page) navigateTo(page);
        });
    });
}

function navigateTo(page) {
    state.currentPage = page;

    // Update nav active states
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.page === page);
    });

    // Show/hide page sections
    document.querySelectorAll('.page-section').forEach(section => {
        section.classList.toggle('active', section.id === `page-${page}`);
    });

    // Update header title
    const titles = {
        'overview': { title: 'Dashboard', sub: 'Welcome to Smart Agriculture AI' },
        'crop': { title: 'Crop Recommendation', sub: 'Find the best crop for your soil and climate' },
        'yield': { title: 'Yield Prediction', sub: 'Estimate your expected crop yield' },
        'rainfall': { title: 'Rainfall Forecast', sub: 'Predict rainfall for better planning' },
        'disease': { title: 'Disease Detection', sub: 'Detect plant diseases from leaf images' },
        'chatbot': { title: 'AI Assistant', sub: 'Ask anything about agriculture' },
    };

    const t = titles[page] || titles['overview'];
    const headerTitle = document.querySelector('.dash-header-title h1');
    const headerSub = document.querySelector('.dash-header-title p');
    if (headerTitle) headerTitle.textContent = t.title;
    if (headerSub) headerSub.textContent = t.sub;

    // Close mobile menu
    closeMobileMenu();
}

// ============================================================
// MOBILE MENU
// ============================================================
function initMobileMenu() {
    const menuBtn = document.querySelector('.mobile-menu-btn');
    const overlay = document.querySelector('.sidebar-overlay');

    if (menuBtn) {
        menuBtn.addEventListener('click', toggleMobileMenu);
    }
    if (overlay) {
        overlay.addEventListener('click', closeMobileMenu);
    }
}

function toggleMobileMenu() {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.querySelector('.sidebar-overlay');
    sidebar.classList.toggle('open');
    if (overlay) overlay.classList.toggle('show');
}

function closeMobileMenu() {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.querySelector('.sidebar-overlay');
    if (sidebar) sidebar.classList.remove('open');
    if (overlay) overlay.classList.remove('show');
}

// ============================================================
// FORM SUBMISSIONS
// ============================================================
function initForms() {
    // Crop Recommendation
    const cropForm = document.getElementById('crop-form');
    if (cropForm) {
        cropForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await predictCrop();
        });
    }

    // Yield Prediction
    const yieldForm = document.getElementById('yield-form');
    if (yieldForm) {
        yieldForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await predictYield();
        });
    }

    // Rainfall Prediction
    const rainfallForm = document.getElementById('rainfall-form');
    if (rainfallForm) {
        rainfallForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await predictRainfall();
        });
    }
}

// ---- CROP RECOMMENDATION ----
async function predictCrop() {
    const data = {
        N: parseFloat(document.getElementById('crop-n').value),
        P: parseFloat(document.getElementById('crop-p').value),
        K: parseFloat(document.getElementById('crop-k').value),
        temperature: parseFloat(document.getElementById('crop-temp').value),
        humidity: parseFloat(document.getElementById('crop-humidity').value),
        ph: parseFloat(document.getElementById('crop-ph').value),
        rainfall: parseFloat(document.getElementById('crop-rainfall').value),
    };

    showLoader('Analyzing soil parameters...');
    hideError('crop-error');
    hideResult('crop-result');

    try {
        const response = await fetch('/predict-crop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        const result = await response.json();
        hideLoader();

        if (result.success) {
            displayCropResult(result);
        } else {
            showError('crop-error', result.error || 'Prediction failed');
        }
    } catch (err) {
        hideLoader();
        showError('crop-error', 'Failed to connect to server. Ensure the app is running.');
    }
}

function displayCropResult(result) {
    const container = document.getElementById('crop-result');
    if (!container) return;

    const cropEmojis = {
        'rice': '🌾', 'wheat': '🌾', 'maize': '🌽', 'cotton': '🏵️',
        'jute': '🧵', 'coffee': '☕', 'sugarcane': '🎋', 'banana': '🍌',
        'mango': '🥭', 'grapes': '🍇', 'apple': '🍎', 'coconut': '🥥',
    };
    const emoji = cropEmojis[result.crop] || '🌱';

    container.querySelector('.result-value').textContent = `${emoji} ${result.crop.charAt(0).toUpperCase() + result.crop.slice(1)}`;
    container.querySelector('.confidence-fill').style.width = `${result.confidence}%`;
    container.querySelector('.confidence-text').textContent = `${result.confidence}% confidence`;

    // Top predictions
    const list = container.querySelector('.predictions-list');
    list.innerHTML = result.top_predictions.map((p, i) => `
        <li>
            <span class="crop-name">
                <span class="rank-badge">${i + 1}</span>
                ${cropEmojis[p.crop] || '🌱'} ${p.crop.charAt(0).toUpperCase() + p.crop.slice(1)}
            </span>
            <span class="crop-score">${p.confidence}%</span>
        </li>
    `).join('');

    showResult('crop-result');
}

// ---- YIELD PREDICTION ----
async function predictYield() {
    const data = {
        crop: document.getElementById('yield-crop').value,
        area: parseFloat(document.getElementById('yield-area').value),
        rainfall: parseFloat(document.getElementById('yield-rainfall').value),
        soil_quality: parseFloat(document.getElementById('yield-soil').value),
    };

    showLoader('Predicting crop yield...');
    hideError('yield-error');
    hideResult('yield-result');

    try {
        const response = await fetch('/predict-yield', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        const result = await response.json();
        hideLoader();

        if (result.success) {
            displayYieldResult(result);
        } else {
            showError('yield-error', result.error || 'Prediction failed');
        }
    } catch (err) {
        hideLoader();
        showError('yield-error', 'Failed to connect to server. Ensure the app is running.');
    }
}

function displayYieldResult(result) {
    const container = document.getElementById('yield-result');
    if (!container) return;

    container.querySelector('.result-value').textContent = `${result.yield_value} ${result.unit}`;

    const details = container.querySelector('.result-details');
    details.innerHTML = `
        <div class="result-detail-item">
            <div class="label">Crop</div>
            <div class="value">${result.crop.charAt(0).toUpperCase() + result.crop.slice(1)}</div>
        </div>
        <div class="result-detail-item">
            <div class="label">Area</div>
            <div class="value">${result.area} hectares</div>
        </div>
        <div class="result-detail-item">
            <div class="label">Expected Yield</div>
            <div class="value">${result.yield_value} ${result.unit}</div>
        </div>
        <div class="result-detail-item">
            <div class="label">Yield per Hectare</div>
            <div class="value">${(result.yield_value / result.area).toFixed(2)} ${result.unit}/ha</div>
        </div>
    `;

    showResult('yield-result');
}

// ---- RAINFALL PREDICTION ----
async function predictRainfall() {
    const data = {
        month: parseInt(document.getElementById('rain-month').value),
        humidity: parseFloat(document.getElementById('rain-humidity').value),
        temperature: parseFloat(document.getElementById('rain-temp').value),
        pressure: parseFloat(document.getElementById('rain-pressure').value),
        wind_speed: parseFloat(document.getElementById('rain-wind').value),
    };

    showLoader('Predicting rainfall...');
    hideError('rainfall-error');
    hideResult('rainfall-result');

    try {
        const response = await fetch('/predict-rainfall', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });

        const result = await response.json();
        hideLoader();

        if (result.success) {
            displayRainfallResult(result);
        } else {
            showError('rainfall-error', result.error || 'Prediction failed');
        }
    } catch (err) {
        hideLoader();
        showError('rainfall-error', 'Failed to connect to server. Ensure the app is running.');
    }
}

function displayRainfallResult(result) {
    const container = document.getElementById('rainfall-result');
    if (!container) return;

    const intensityEmojis = {
        'Low': '🌤️', 'Moderate': '🌦️', 'High': '🌧️', 'Very High': '⛈️'
    };

    const emoji = intensityEmojis[result.intensity] || '🌧️';
    container.querySelector('.result-value').textContent = `${emoji} ${result.rainfall} mm`;

    const details = container.querySelector('.result-details');
    details.innerHTML = `
        <div class="result-detail-item">
            <div class="label">Rainfall</div>
            <div class="value">${result.rainfall} mm</div>
        </div>
        <div class="result-detail-item">
            <div class="label">Intensity</div>
            <div class="value">${result.intensity}</div>
        </div>
        <div class="result-detail-item">
            <div class="label">Advisory</div>
            <div class="value">${getRainfallAdvisory(result.intensity)}</div>
        </div>
    `;

    showResult('rainfall-result');
}

function getRainfallAdvisory(intensity) {
    const advisories = {
        'Low': 'Consider irrigation',
        'Moderate': 'Good for most crops',
        'High': 'Ensure drainage ready',
        'Very High': 'Flood risk — take precautions',
    };
    return advisories[intensity] || 'Monitor conditions';
}

// ============================================================
// IMAGE UPLOAD & DISEASE DETECTION
// ============================================================
function initImageUpload() {
    const zone = document.querySelector('.upload-zone');
    const input = document.getElementById('disease-image');

    if (!zone || !input) return;

    // Drag and drop
    ['dragenter', 'dragover'].forEach(evt => {
        zone.addEventListener(evt, (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        zone.addEventListener(evt, (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
        });
    });

    zone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            input.files = files;
            showImagePreview(files[0]);
        }
    });

    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            showImagePreview(input.files[0]);
        }
    });

    // Submit button
    const detectBtn = document.getElementById('detect-disease-btn');
    if (detectBtn) {
        detectBtn.addEventListener('click', predictDisease);
    }
}

function showImagePreview(file) {
    const preview = document.querySelector('.image-preview');
    const img = preview.querySelector('img');

    if (preview && img) {
        const reader = new FileReader();
        reader.onload = (e) => {
            img.src = e.target.result;
            preview.classList.add('show');
        };
        reader.readAsDataURL(file);
    }
}

async function predictDisease() {
    const input = document.getElementById('disease-image');
    if (!input || input.files.length === 0) {
        showError('disease-error', 'Please upload a leaf image first.');
        return;
    }

    const formData = new FormData();
    formData.append('image', input.files[0]);

    showLoader('Analyzing leaf image...');
    hideError('disease-error');

    const diseaseResult = document.querySelector('.disease-result');
    if (diseaseResult) diseaseResult.classList.remove('show');

    try {
        const response = await fetch('/predict-disease', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();
        hideLoader();

        if (result.success) {
            displayDiseaseResult(result);
        } else {
            showError('disease-error', result.error || 'Detection failed');
        }
    } catch (err) {
        hideLoader();
        showError('disease-error', 'Failed to connect to server.');
    }
}

function displayDiseaseResult(result) {
    const container = document.querySelector('.disease-result');
    if (!container) return;

    // Status
    const status = container.querySelector('.disease-status');
    status.className = `disease-status ${result.is_healthy ? 'healthy' : 'infected'}`;
    status.querySelector('.status-icon').textContent = result.is_healthy ? '✅' : '⚠️';
    status.querySelector('.status-text h3').textContent = result.disease;
    status.querySelector('.status-text p').textContent = `Confidence: ${result.confidence}%`;

    // Confidence
    container.querySelector('.confidence-fill').style.width = `${result.confidence}%`;

    // Treatment
    const treatmentText = container.querySelector('.treatment-text');
    if (treatmentText) {
        treatmentText.textContent = result.treatment;
    }

    // Top predictions
    const list = container.querySelector('.predictions-list');
    if (list && result.top_predictions) {
        list.innerHTML = result.top_predictions.map((p, i) => `
            <li>
                <span class="crop-name">
                    <span class="rank-badge">${i + 1}</span>
                    ${p.disease}
                </span>
                <span class="crop-score">${p.confidence}%</span>
            </li>
        `).join('');
    }

    container.classList.add('show');
}

// ============================================================
// CHATBOT
// ============================================================
function initChatbot() {
    const sendBtn = document.getElementById('chat-send');
    const input = document.getElementById('chat-input');

    if (sendBtn) {
        sendBtn.addEventListener('click', sendChatMessage);
    }
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message) return;

    // Add user message
    addChatMessage('user', message);
    input.value = '';

    // Add to history
    state.chatHistory.push({ role: 'user', content: message });

    // Show typing indicator
    showTypingIndicator();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                history: state.chatHistory.slice(-10),
            }),
        });

        const result = await response.json();
        hideTypingIndicator();

        if (result.success) {
            addChatMessage('bot', result.response);
            state.chatHistory.push({ role: 'assistant', content: result.response });

            // Text-to-speech
            speak(result.response);
        } else {
            addChatMessage('bot', '❌ ' + (result.error || 'Something went wrong.'));
        }
    } catch (err) {
        hideTypingIndicator();
        addChatMessage('bot', '❌ Failed to connect to server. Please make sure the app is running.');
    }
}

function addChatMessage(role, content) {
    const container = document.getElementById('chat-messages');
    if (!container) return;

    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message ${role}`;

    const avatar = role === 'bot' ? '🌾' : '👤';

    msgDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-bubble">${escapeHtml(content)}</div>
    `;

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function showTypingIndicator() {
    const container = document.getElementById('chat-messages');
    if (!container) return;

    const indicator = document.createElement('div');
    indicator.className = 'chat-message bot';
    indicator.id = 'typing-indicator';
    indicator.innerHTML = `
        <div class="message-avatar">🌾</div>
        <div class="message-bubble">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    container.appendChild(indicator);
    container.scrollTop = container.scrollHeight;
}

function hideTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================
// VOICE ASSISTANT
// ============================================================
function initVoice() {
    const voiceBtn = document.getElementById('chat-voice');
    if (!voiceBtn) return;

    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        voiceBtn.style.display = 'none';
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    state.recognition = new SpeechRecognition();
    state.recognition.continuous = false;
    state.recognition.interimResults = false;
    state.recognition.lang = 'en-US';

    state.recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        const input = document.getElementById('chat-input');
        if (input) {
            input.value = transcript;
            sendChatMessage();
        }
    };

    state.recognition.onend = () => {
        state.isRecording = false;
        voiceBtn.classList.remove('recording');
    };

    state.recognition.onerror = () => {
        state.isRecording = false;
        voiceBtn.classList.remove('recording');
    };

    voiceBtn.addEventListener('click', toggleVoice);
}

function toggleVoice() {
    const voiceBtn = document.getElementById('chat-voice');

    if (state.isRecording) {
        state.recognition.stop();
        state.isRecording = false;
        voiceBtn.classList.remove('recording');
    } else {
        state.recognition.start();
        state.isRecording = true;
        voiceBtn.classList.add('recording');
    }
}

function speak(text) {
    if (!('speechSynthesis' in window)) return;

    // Clean text for speech
    const cleanText = text.replace(/[*#•]/g, '').replace(/\n+/g, '. ').substring(0, 500);

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 0.95;
    utterance.pitch = 1;
    utterance.volume = 0.8;

    // Try to find a good English voice
    const voices = speechSynthesis.getVoices();
    const englishVoice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Google'));
    if (englishVoice) utterance.voice = englishVoice;

    speechSynthesis.cancel(); // stop any current speech
    speechSynthesis.speak(utterance);
}

// ============================================================
// LOAD CROPS LIST
// ============================================================
async function loadCropsList() {
    try {
        const response = await fetch('/get-crops');
        const result = await response.json();

        if (result.success && result.crops) {
            const select = document.getElementById('yield-crop');
            if (select) {
                select.innerHTML = result.crops.map(crop =>
                    `<option value="${crop}">${crop.charAt(0).toUpperCase() + crop.slice(1)}</option>`
                ).join('');
            }
        }
    } catch (err) {
        console.log('Could not load crops list, using defaults.');
    }
}

// ============================================================
// HELPERS
// ============================================================
function showLoader(text) {
    const loader = document.querySelector('.loader-overlay');
    if (loader) {
        const msg = loader.querySelector('p');
        if (msg) msg.textContent = text || 'Processing...';
        loader.classList.add('show');
    }
}

function hideLoader() {
    const loader = document.querySelector('.loader-overlay');
    if (loader) loader.classList.remove('show');
}

function showResult(id) {
    const el = document.getElementById(id);
    if (el) el.classList.add('show');
}

function hideResult(id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove('show');
}

function showError(id, message) {
    const el = document.getElementById(id);
    if (el) {
        el.innerHTML = `⚠️ ${message}`;
        el.classList.add('show');
    }
}

function hideError(id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove('show');
}
