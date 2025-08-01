import os
import logging
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import io
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Audio processing constants
SAMPLE_RATE = 22050
DURATION = 30
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

model_lock = Lock()  # Thread safety for concurrent users

# Genre labels
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

class MusicGenreCNN(nn.Module):
    """
    Convolutional Neural Network for Music Genre Classification
    Same architecture as in the notebook
    """
    
    def __init__(self, num_classes=10, input_channels=1):
        super(MusicGenreCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Convolutional layers with batch normalization, activation, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout1(x)
        
        # Adaptive pooling to handle variable input sizes
        x = self.adaptive_pool(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, 256 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

# Use best available device for optimal performance
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Global model management
model = None
model_loaded = False

def load_model_once():
    """Load model only once and keep in memory"""
    global model, model_loaded
    if model_loaded and model is not None:
        return model
    
    try:
        logger.info("Loading model...")
        model = MusicGenreCNN(num_classes=len(GENRES))
        checkpoint = torch.load('gtzan_genre_classifier.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        model_loaded = True
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False
        return None

# Load model at startup
model = load_model_once()

def load_and_preprocess_audio(file_path_or_data, duration=30, sr=22050, is_bytes=False):
    """
    Load and preprocess audio file or bytes data
    """
    try:
        if is_bytes:
            # Load from bytes data
            audio, _ = librosa.load(io.BytesIO(file_path_or_data), sr=sr, duration=duration)
        else:
            # Load from file path
            audio, _ = librosa.load(file_path_or_data, sr=sr, duration=duration)
        
        # Pad or trim to fixed length
        target_length = sr * duration
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            
        return audio
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return None

def extract_mel_spectrogram(audio, sr=22050):
    """
    Extract mel-spectrogram features from audio
    """
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=N_MELS, 
            hop_length=HOP_LENGTH, n_fft=N_FFT
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    except Exception as e:
        logger.error(f"Error extracting mel-spectrogram: {e}")
        return None

def predict_genre(audio_data, top_k=5, is_bytes=False):
    """
    Prediction with thread safety
    Handles various audio formats and provides detailed error messages
    """
    with model_lock:  # Ensure thread safety for concurrent users
        try:
            if model is None:
                return {"error": "Model not loaded. Please check server logs."}

            # Load and preprocess audio with multiple attempts if needed
            audio = None
            error_messages = []
            
            for attempt in range(3):  # Try up to 3 different loading methods
                try:
                    if attempt == 0:
                        # First try with normal parameters
                        audio = load_and_preprocess_audio(audio_data, DURATION, SAMPLE_RATE, is_bytes)
                    elif attempt == 1:
                        # Try loading without duration limit
                        if is_bytes:
                            audio, orig_sr = librosa.load(io.BytesIO(audio_data), sr=None)
                        else:
                            audio, orig_sr = librosa.load(audio_data, sr=None)
                        # Resample if needed
                        if orig_sr != SAMPLE_RATE:
                            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
                        # Pad/trim to target length
                        target_length = SAMPLE_RATE * DURATION
                        if len(audio) > target_length:
                            audio = audio[:target_length]
                        else:
                            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
                    elif attempt == 2:
                        # Final attempt with mono conversion and resampling
                        if is_bytes:
                            audio, orig_sr = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
                        else:
                            audio, orig_sr = librosa.load(audio_data, sr=None, mono=True)
                        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
                        target_length = SAMPLE_RATE * DURATION
                        audio = librosa.util.fix_length(audio, target_length)
                    
                    if audio is not None:
                        break
                        
                except Exception as e:
                    error_messages.append(f"Attempt {attempt + 1} failed: {str(e)}")
                    continue

            if audio is None:
                return {
                    "error": "Failed to process audio file",
                    "details": error_messages,
                    "suggestion": "Please try a different audio format (WAV or MP3 recommended)"
                }

            # Extract mel-spectrogram features
            mel_spec = extract_mel_spectrogram(audio, SAMPLE_RATE)
            if mel_spec is None:
                return {"error": "Feature extraction failed"}

            # Convert to tensor and move to device
            try:
                features = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)
            except Exception as e:
                return {
                    "error": "Failed to convert features to tensor",
                    "details": str(e)
                }

            # Make prediction
            try:
                with torch.no_grad():
                    output = model(features)
                    probabilities = F.softmax(output, dim=1)
                    
                    # Get top-k predictions
                    top_probs, top_indices = torch.topk(probabilities, min(top_k, len(GENRES)))
                    
                    predictions = []
                    for i in range(top_probs.size(1)):
                        genre_idx = top_indices[0][i].item()
                        confidence = top_probs[0][i].item()
                        predictions.append({
                            "genre": GENRES[genre_idx],
                            "confidence": round(confidence, 4),
                            "percentage": round(confidence * 100, 2)
                        })
                    
                    return {
                        "success": True,
                        "predictions": predictions,
                        "top_prediction": predictions[0] if predictions else None,
                        "audio_info": {
                            "duration_seconds": len(audio) / SAMPLE_RATE,
                            "sample_rate": SAMPLE_RATE,
                            "normalized": True
                        }
                    }
                    
            except Exception as e:
                return {
                    "error": "Prediction failed",
                    "details": str(e)
                }
            
        except Exception as e:
            logger.error(f"Unexpected error in predict_genre: {e}")
            return {
                "error": "Unexpected error during processing",
                "details": str(e)
            }

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/debug')
def debug():
    """Serve the debug page"""
    return send_from_directory('.', 'debug_ui.html')

@app.route('/audio/<genre>/<filename>')
def serve_audio(genre, filename):
    """Serve audio files for playback"""
    try:
        audio_path = f"test_data/{genre}"
        return send_from_directory(audio_path, filename)
    except Exception as e:
        logger.error(f"Error serving audio file: {e}")
        return {"error": "Audio file not found"}, 404

@app.route('/predict', methods=['POST'])
def predict():
    """Handle audio file upload and prediction"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_genre(filepath)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/predict_sample/<genre>/<filename>')
def predict_sample(genre, filename):
    """Predict genre for a sample file from the dataset"""
    try:
        # Validate genre
        if genre not in GENRES:
            return jsonify({"error": "Invalid genre"}), 400
        
        # Construct file path
        file_path = f"test_data/{genre}/{filename}"
        
        if not os.path.exists(file_path):
            return jsonify({"error": "Sample file not found"}), 404
        
        # Make prediction
        result = predict_genre(file_path)
        
        if "error" in result:
            return jsonify(result), 500
        
        # Add actual genre for comparison
        result["actual_genre"] = genre
        result["is_correct"] = result["top_prediction"]["genre"] == genre if result["top_prediction"] else False
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict_sample endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/samples/<genre>')
def get_samples(genre):
    """Get list of sample files for a genre"""
    try:
        if genre not in GENRES:
            return jsonify({"error": "Invalid genre"}), 400
        
        genre_path = f"test_data/{genre}"
        if not os.path.exists(genre_path):
            return jsonify({"error": "Genre folder not found"}), 404
        
        # Get first 10 sample files
        files = [f for f in os.listdir(genre_path) if f.endswith('.wav')][:10]
        files.sort()
        
        return jsonify({
            "genre": genre,
            "samples": files
        })
        
    except Exception as e:
        logger.error(f"Error in get_samples endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/genres')
def get_genres():
    """Get list of available genres"""
    return jsonify({"genres": GENRES})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    })

if __name__ == '__main__':
    import sys
    
    # Default port
    port = 5000
    
    # Check for port argument
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Usage: python app.py [port]")
            print("Example: python app.py 8080")
            sys.exit(1)
    
    logger.info("Starting Genre Classification Flask App")
    logger.info(f"Available genres: {GENRES}")
    logger.info(f"Starting server on port {port}")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=port)
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {port} is already in use!")
            logger.info("Try a different port: python app.py 8080")
            logger.info("Or on macOS, disable AirPlay Receiver in System Preferences")
        else:
            logger.error(f"Error starting server: {e}")