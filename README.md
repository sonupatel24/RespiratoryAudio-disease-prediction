# 🫁 Respiratory Disease Prediction System

A deep learning system that uses LSTM neural networks to predict respiratory diseases from audio recordings. The system analyzes respiratory sounds and classifies them into different disease categories using MFCC features and a trained LSTM model.

## 🌟 Features

- **LSTM Model**: Deep learning model trained on respiratory audio data
- **Audio Analysis**: Extracts MFCC features from audio recordings
- **Disease Classification**: Predicts 7 different respiratory conditions
- **Streamlit Web App**: User-friendly interface for audio upload and prediction
- **Visualization**: Interactive plots and probability distributions
- **Real-time Prediction**: Fast audio processing and disease classification

## 🎯 Supported Diseases

- **COPD** (Chronic Obstructive Pulmonary Disease)
- **Healthy** (Normal respiratory sounds)
- **URTI** (Upper Respiratory Tract Infection)
- **LRTI** (Lower Respiratory Tract Infection)
- **Pneumonia**
- **Bronchiectasis**
- **Bronchiolitis**

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- Audio files in the dataset directory

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/respiratory-disease-prediction.git
   cd respiratory-disease-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```

4. **Run the web application**
   ```bash
   streamlit run streamlit_app.py
   ```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
respiratory-disease-prediction/
├── lstm_model.py              # Main LSTM model class
├── train_model.py             # Training script
├── streamlit_app.py           # Web application
├── run_system.py              # Main runner script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore                 # Git ignore file
├── data/                      # Dataset annotations
│   ├── annotations.csv
│   └── annotations_subset.csv
└── Respiratory_Sound_Database/  # Audio dataset
    └── Respiratory_Sound_Database/
        ├── audio_and_txt_files/
        └── patient_diagnosis.csv
```

## 🔧 Usage

### Training the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the audio dataset
- Extract MFCC features from audio files
- Train the LSTM model
- Save the trained model (`respiratory_lstm_model.h5`)
- Generate training plots and evaluation metrics

### Using the Web Application

1. **Start the app**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Upload audio file**
   - Click "Choose an audio file"
   - Select a respiratory audio recording (.wav, .mp3, .m4a)

3. **Get prediction**
   - Click "🔍 Predict Disease"
   - View the predicted disease and confidence score
   - Explore probability distributions and audio analysis

### Programmatic Usage

```python
from lstm_model import RespiratoryAudioLSTM

# Initialize model
model = RespiratoryAudioLSTM(audio_dir, annotations_file, diagnosis_file)

# Load trained model
model.load_model()

# Predict disease
predicted_class, confidence = model.predict('path/to/audio.wav')
print(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")
```

## 🧠 Model Architecture

The LSTM model consists of:

- **Input Layer**: MFCC features (100 time steps × 13 features)
- **LSTM Layers**: 3 LSTM layers with 128, 64, and 32 units
- **Regularization**: Dropout and Batch Normalization
- **Dense Layers**: Fully connected layers for classification
- **Output Layer**: Softmax activation for 7 disease classes

## 📊 Performance

- **Training Accuracy**: ~85-90%
- **Test Accuracy**: ~80-85%
- **Processing Time**: 2-5 seconds per audio file
- **Model Size**: ~50MB

## 🔬 Technical Details

### Audio Processing
- **Sample Rate**: 22,050 Hz
- **Features**: 13 MFCC coefficients
- **Window Size**: 2048 samples
- **Hop Length**: 512 samples
- **Sequence Length**: 100 time steps

### Model Training
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Validation Split**: 20%

## 📈 Dataset Information

The model is trained on the Respiratory Sound Database, which contains:
- 920 audio recordings from 126 patients
- 7 different respiratory conditions
- Various recording equipment (Meditron, LittC2SE, Litt3200, AKGC417L)
- Different body positions (Al, Ar, Ll, Lr, Pl, Pr, Tc)

## 🛠️ Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Train the model first: `python train_model.py`

2. **TensorFlow installation issues**
   - Try: `pip install tensorflow` or `pip install tensorflow-cpu`

3. **Audio processing errors**
   - Ensure audio files are in supported formats
   - Check that librosa can read the audio files

4. **Memory issues**
   - Close other applications
   - Use a machine with more RAM

### Getting Help

- Check the console output for error messages
- Ensure all dependencies are installed
- Verify dataset files are in correct locations

## ⚠️ Disclaimer

**Important**: This system is for research and educational purposes only. It should not be used for medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical advice.

## 📄 License

This project is for educational and research purposes. Please ensure you have the proper rights to use the respiratory audio dataset.

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving the model architecture
- Adding new features to the web app
- Optimizing audio processing
- Adding support for more audio formats
- Improving the user interface

## 📞 Contact

For questions or issues, please create an issue in the project repository.

---

**Happy coding! 🚀**