# Music Genre Classification System

## Overview

A Flask web application that classifies music genres using a Convolutional Neural Network (CNN) trained on the GTZAN dataset. Upload audio files or record directly to get genre predictions with confidence scores.

## Features

- Audio file upload (MP3, WAV, FLAC)
- Real-time microphone recording
- Genre classification with confidence scores
- REST API endpoints
- Sample audio testing

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- GPU support optional but recommended
- Kaggle account (for dataset download)

### Step 1: Clone and Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd music-genre-classification

# Install required packages
pip install -r requirements.txt
pip install kaggle  # For dataset download
```

### Step 2: Download Dataset

Download the GTZAN dataset from Kaggle:

1. Get Kaggle API token from [kaggle.com](https://www.kaggle.com) → Account → API
2. Setup: `mkdir ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
3. Download: `kaggle datasets download andradaolteanu/gtzan-dataset-music-genre-classification`
4. Extract: `unzip gtzan-dataset-music-genre-classification.zip && mv Data Data-2`


### Step 3: Verify Installation

```bash
# Run the test suite to verify installation
python test_app.py

# Quick test - should show "Model loaded successfully"
python -c "from app import model; print('✓ Setup complete!' if model else '✗ Setup failed')"
```

### Application Startup

#### Simple Start

```bash
# Start with default port (5000)
python app.py

# Start with custom port
python app.py 8080
python app.py 3000

# Access the web interface
# Navigate to: http://localhost:5000 (or your configured port)
```

If port 5000 is in use: `python app.py 8080`

## Usage

1. **Upload Audio**: Drag/drop audio files (MP3, WAV, FLAC)
2. **Record Audio**: Use microphone for real-time recording  
3. **Get Results**: View genre predictions with confidence scores
4. **Test Samples**: Try pre-loaded dataset examples

## API Endpoints

- `POST /predict` - Upload and classify audio file
- `GET /health` - Check system status
- `GET /genres` - List available genres
- `GET /predict_sample/{genre}/{filename}` - Test with sample audio

## Performance

- **Test Accuracy**: 80.67%
- **Model Size**: 10.2 MB  
- **Inference Time**: ~1-2 seconds per file

## Research Opportunities

**Improve Model Performance**: Current baseline is 80.67% accuracy - can you beat it?

**Experiment Ideas**:
- Try different architectures (ResNet, Transformers)
- Implement data augmentation techniques  
- Experiment with short-duration classification (5-second clips)
- Add real-time processing capabilities

